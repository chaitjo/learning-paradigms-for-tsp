import os
import argparse
import pprint as pp
import numpy as np
import pickle
from datetime import timedelta
from torch.utils.data import DataLoader

from problems.tsp.problem_tsp import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="tsp")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model_names", nargs='+', required=True)
    parser.add_argument("--dataset_names", nargs='+', required=True)
    parser.add_argument("--dataset_sizes", nargs='+', required=True)
    
    opts = parser.parse_args()
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    for dataset_name, dataset_size in zip(opts.dataset_names, opts.dataset_sizes):
        dataset_path = f"data/tsp/{dataset_name}.txt"

        dataset = TSPSL.make_dataset(filename=dataset_path, num_samples=dataset_size)
        dataloader = DataLoader(dataset, batch_size=opts.batch_size)
        gt_costs = []
        for batch_idx, batch in enumerate(dataloader):
            nodes_coord = batch["nodes_coord"]
            tour_nodes = batch["tour_nodes"]
            gt_costs.append(TSPSL.get_costs(nodes_coord, tour_nodes)[0])
        gt_costs = np.stack(gt_costs).flatten()
        
        for model_name in opts.model_names:
            for ext in ['greedy', 'sample250', 'bs250', 'sample1280', 'bs1280']:
                model_name_ext = model_name + '-' + ext
                
                try:
                    if opts.problem == 'tspsl':
                        res_file = f"results/{opts.problem}/{dataset_name}/{dataset_name}-{model_name_ext}-t1-0-{dataset_size}.txt.pkl"
                    else:
                        res_file = f"results/{opts.problem}/{dataset_name}/{dataset_name}-{model_name_ext}-t1-0-{dataset_size}.pkl"
                    results, parallelism = pickle.load(open(res_file, 'rb'))
                    costs, tours, durations = zip(*results)

                    print(f"Model: {model_name_ext}")
                    print(f"Dataset: {dataset_name}")
                    print("Average cost: {:.3f} +- {:.3f}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
                    print("Groundtruth cost: {:.3f} +- {:.3f}".format(np.mean(gt_costs), 2 * np.std(gt_costs) / np.sqrt(len(gt_costs))))
                    print("Average Optimality Gap: {:.3f}%".format((np.mean(costs)/np.mean(gt_costs) - 1)*100))
                    print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
                    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
                    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))
                    print()
                
                except:
                    print(f"File not found: results/{opts.problem}/{dataset_name}/{dataset_name}-{model_name_ext}-t1-0-{dataset_size}.txt.pkl")
                    print()

