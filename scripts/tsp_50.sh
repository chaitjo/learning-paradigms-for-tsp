# Training
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --model attention --encoder gat --baseline rollout --graph_size 50 --batch_size 512 --val_dataset data/tsp/tsp50_val_concorde.pkl --lr_model 1e-4 --log_step 100 --run_name ar_gat_rl
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --model attention --encoder gcn --baseline rollout --graph_size 50 --batch_size 512 --val_dataset data/tsp/tsp50_val_concorde.pkl --lr_model 1e-4 --log_step 100 --run_name ar_gcn_rl
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --model attention --encoder gat --baseline critic --graph_size 50 --batch_size 512 --val_dataset data/tsp/tsp50_val_concorde.pkl --lr_model 1e-4 --log_step 100 --run_name ar_gat_rl_ac
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --model attention --encoder mlp --embedding_dim 256 --baseline rollout --graph_size 50 --batch_size 512 --val_dataset data/tsp/tsp50_val_concorde.pkl --lr_model 1e-4 --log_step 100 --run_name ar_mlp_rl


# Evaluation

### GAT Model
CCUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp50_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp50_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp50_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 1280 --eval_batch_size 250

CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 1280 --eval_batch_size 500

CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp100_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp100_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp100_test_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 1280 --eval_batch_size 100

### GCN Model
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gcn_rl_20190831T163814/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gcn_rl_20190831T163814/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gcn_rl_20190831T163814/ --decode_strategy bs --width 1280 --eval_batch_size 100

### MLP Model
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_mlp_rl_20190906T161220/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_mlp_rl_20190906T161220/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_mlp_rl_20190906T161220/ --decode_strategy bs --width 1280 --eval_batch_size 50

### AC Model
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gat_rl_ac_20190906T133034/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gat_rl_ac_20190906T133034/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=1 python eval.py data/tsp/tsp20_test_concorde.pkl data/tsp/tsp50_test_concorde.pkl data/tsp/tsp100_test_concorde.pkl --model outputs/tsp_50/ar_gat_rl_ac_20190906T133034/ --decode_strategy bs --width 1280 --eval_batch_size 100


# Transfer to large-scale TSP

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.pkl --model pretrained/tsp_50/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.pkl --model pretrained/tsp_50/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.pkl --model pretrained/tsp_50/ --decode_strategy bs --width 250 --eval_batch_size 10
