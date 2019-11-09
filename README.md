# On Learning Paradigms for the Travelling Salesman Problem

This repository contains code for the paper 
[**"On Learning Paradigms for the Travelling Salesman Problem"**](https://arxiv.org/abs/1910.07210)
by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson, presented at the [NeurIPS 2019 Graph Representation Learning Workshop](https://grlearning.github.io/overview/).

We explore the impact of learning paradigms on training deep neural networks for the Travelling Salesman Problem. 
We design controlled experiments to train supervised learning (SL) and reinforcement learning (RL) models on fixed graph sizes up to 100 nodes, and evaluate them on variable sized graphs up to 500 nodes.
Beyond not needing labelled data, our results reveal favorable properties of RL over SL:
RL training leads to better *emergent* generalization to variable graph sizes and 
is a key component for learning scale-invariant solvers for novel combinatorial problems.

**Acknowledgements:** Our work is a modified clone of [Wouter Kool's excellent code repository](https://github.com/wouterkool/attention-learn-to-route) for the paper ["Attention, Learn to Solve Routing Problems!"](https://openreview.net/forum?id=ByxBFsRqYm) by Wouter Kool, Herke van Hoof and Max Welling (ICLR 2019). 
Please refer to the original repository for usage instructions beyond reporoducing our experimental results.

## Installation

```
# Clone the repository. 
git clone https://github.com/chaitjo/learning-paradigms-for-tsp.git
cd learning-paradigms-for-tsp

# Set up a new conda environment and activate it.
conda create -n tsp-env python=3.6.7
source activate tsp-env

# Install all dependencies and Jupyter Lab (for using notebooks).
conda install pytorch=1.2.0 cudatoolkit=10.0 -c pytorch
conda install numpy scipy tqdm matplotlib
pip install tensorboard_logger
conda install -c conda-forge jupyterlab
```

## Data preparation

For Supervised Learning, download and prepare TSP20, TSP50 and TSP100 datasets by following instructions from [our previous work](https://github.com/chaitjo/graph-convnet-tsp):
1. Download TSP datasets from [this link](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp). Extract the `.tar.gz` file and place each `.txt` file in the `/data/tsp/` directory.
```
tar -xvzf tsp-data.tar.gz ./data/tsp/
```

2. For TSP50 and TSP100, the 1M training set needs to be split into 10K validation samples and 999K training samples. Use the `split_train_val.py` script to do so: 
```
python data/split_train_val.py --num_nodes <num-nodes>
```

3. Performing validation/evaluation for Reinforcement Learning models on a fixed validation/test set requires converting the dataset from `.txt` files to `.pkl` files using the `data/google_data_to_pkl.py` script: 
```
python data/google_data_to_pkl.py --type <val/test> --num_nodes <20/50/100>
```

## Usage

Refer to [Wouter Kool's repository](https://github.com/wouterkool/attention-learn-to-route) for comprehensive instructions for running the codebase.
For reproducing experiments, we provide a set of scripts for training and evaluation in the `/scripts` directory.
Pre-trained models and TensorBoard logs for all experiments described in the paper can be found in the `/pretrained` directory.

High-level commands:
```
# Training
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python run.py 
    --problem <tsp/tspsl> 
    --model attention 
    --encoder <gat/gcn> 
    --baseline <rollout/critic> 
    --graph_size <20/50/100> 
    --batch_size 512 
    --train_dataset data/tsp/tsp<20/50/100>_train_concorde.txt 
    --val_dataset data/tsp/tsp<20/50/100>_val_concorde.txt 
    --lr_model 1e-4 
    --log_step 100 
    --run_name ar_<gat/gcn>_<rl/sl>
    
# Evaluation
CUDA_VISIBLE_DEVICES=<available-gpu-ids> python eval.py data/tsp/tsp<20/50/100>_test_concorde.<txt/pkl>
    --model outputs/<tsp/tspsl>_<20/50/100>/ar_<gat/gcn>_<rl/sl>_<datetime>/ 
    --decode_strategy <greedy/sample/bs> 
    --eval_batch_size <1/200/...>
    --width 1280 
```


## Resources

- [Optimal TSP Datasets generated with Concorde](https://drive.google.com/open?id=1-5W-S5e7CKsJ9uY9uVXIyxgbcZZNYBrp)
- [Paper on arXiv](https://arxiv.org/abs/1910.07210)
- [Wouter Kool's repository](https://github.com/wouterkool/attention-learn-to-route) and [ICLR paper](https://openreview.net/forum?id=ByxBFsRqYm)
