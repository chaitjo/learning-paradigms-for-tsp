import time
import argparse
import pprint as pp

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle

import pickle
import os


def save_dataset(dataset, filename):
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        
def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class GoogleTSPReader(object):
    """Iterator that reads TSP dataset files and yields mini-batches.
    
    Format expected as in Vinyals et al., 2015: https://arxiv.org/abs/1506.03134, http://goo.gl/NDcOIG
    
    References:
        - https://github.com/chaitjo/graph-convnet-tsp/blob/master/utils/google_tsp_reader.py
    """

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath):
        """
        Args:
            num_nodes: Number of nodes in TSP tours
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.filedata = open(filepath, "r").readlines()
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_tour_nodes = []
        batch_tour_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            
            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            
            # Compute node and edge representation of tour + tour_len
            tour_len = 0
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                tour_len += W_val[i][j]
            
            # Add final connection of tour in edge target
            nodes_target[j] = len(tour_nodes) - 1
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1
            tour_len += W_val[j][tour_nodes[0]]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_tour_nodes.append(tour_nodes)
            batch_tour_len.append(tour_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.tour_nodes = np.stack(batch_tour_nodes, axis=0)
        batch.tour_len = np.stack(batch_tour_len, axis=0)
        return batch

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_neighbors", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    
    opts = parser.parse_args()
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    if opts.type is None:
        filename = f"data/tsp/tsp{opts.num_nodes}_concorde"
    else:
        filename = f"data/tsp/tsp{opts.num_nodes}_{opts.type}_concorde"
    dataset = GoogleTSPReader(opts.num_nodes, opts.num_neighbors, opts.batch_size, filename+".txt")
    print("Number of batches of size {}: {}".format(opts.batch_size, dataset.max_iter))

    for idx, batch in enumerate(dataset):
        if idx==0:
            dataset_nodes = batch.nodes
            dataset_nodes_coord = batch.nodes_coord
            dataset_nodes_target = batch.nodes_target
            dataset_edges = batch.edges
            dataset_edges_values = batch.edges_values
            dataset_edges_target = batch.edges_target
            dataset_tour_nodes = batch.tour_nodes
            dataset_tour_len = batch.tour_len
        else:
            dataset_nodes = np.concatenate((dataset_nodes, batch.nodes), axis=0)
            dataset_nodes_coord = np.concatenate((dataset_nodes_coord, batch.nodes_coord), axis=0)
            dataset_nodes_target = np.concatenate((dataset_nodes_target, batch.nodes_target), axis=0)
            dataset_edges = np.concatenate((dataset_edges, batch.edges), axis=0)
            dataset_edges_values = np.concatenate((dataset_edges_values, batch.edges_values), axis=0)
            dataset_edges_target = np.concatenate((dataset_edges_target, batch.edges_target), axis=0)
            dataset_tour_nodes = np.concatenate((dataset_tour_nodes, batch.tour_nodes), axis=0)
            dataset_tour_len = np.concatenate((dataset_tour_len, batch.tour_len), axis=0)

    print("nodes:", dataset_nodes.shape)
    print("nodes_coord:", dataset_nodes_coord.shape)
    print("nodes_target:", dataset_nodes_target.shape)
    print("edges:", dataset_edges.shape)
    print("edges_values:", dataset_edges_values.shape)
    print("edges_targets:", dataset_edges_target.shape)
    print("tour_nodes:", dataset_tour_nodes.shape)
    print("tour_len:", dataset_tour_len.shape)
    
    save_dataset(dataset_nodes_coord.tolist(), filename+".pkl")

    # Uncomment to save more detailed dataset objects to .pkl
    # save_dataset(
    #     {
    #         "nodes": dataset_nodes.tolist(),
    #         "nodes_coord": dataset_nodes_coord.tolist(),
    #         "nodes_target": dataset_nodes_target.tolist(),
    #         "edges": dataset_edges.tolist(),
    #         "edges_values": dataset_edges_values.tolist(),
    #         "edges_target": dataset_edges_target.tolist(),
    #         "tour_nodes": dataset_tour_nodes.tolist(),
    #         "tour_len": dataset_tour_len.tolist()
    #     }, 
    #     filename+".pkl")

    print("Saved: ", filename+".pkl")
