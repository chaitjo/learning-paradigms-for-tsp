# Training
CUDA_VISIBLE_DEVICES=0 python run.py --problem tspsl --model attention --encoder gat --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --lr_model 1e-4 --log_step 100 --run_name ar_gat_sl
CUDA_VISIBLE_DEVICES=0 python run.py --problem tspsl --model attention --encoder gcn --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --lr_model 1e-4 --log_step 100 --run_name ar_gcn_sl
CUDA_VISIBLE_DEVICES=0 python run.py --problem tspsl --model attention --encoder mlp --embedding_dim 256 --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --lr_model 1e-4 --log_step 100 --run_name ar_mlp_sl


# Evaluation

### GAT Model
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 1280 --eval_batch_size 500

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp50_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp50_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp50_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 1280 --eval_batch_size 250

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 1280 --eval_batch_size 100

### GCN Model
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_gcn_sl_20190831T164126/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_gcn_sl_20190831T164126/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_gcn_sl_20190831T164126/ --decode_strategy bs --width 1280 --eval_batch_size 100

### MLP Model
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_mlp_sl_20190906T161157/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_mlp_sl_20190906T161157/ --decode_strategy sample --width 1280 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/ar_mlp_sl_20190906T161157/ --decode_strategy bs --width 1280 --eval_batch_size 100


# Transfer to large-scale TSP

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp150_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp200_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp300_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp400_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp500_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp600_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp700_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp800_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy greedy --eval_batch_size 100
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy sample --width 250 --eval_batch_size 1
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp900_concorde.txt --model outputs/tspsl_20/tsp20_sl_20190816T173555/ --decode_strategy bs --width 250 --eval_batch_size 10

