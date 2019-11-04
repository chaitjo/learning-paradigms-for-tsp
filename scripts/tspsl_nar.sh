# Training
CUDA_VISIBLE_DEVICES=0 python run.py --problem tspsl --model mlp --encoder gat --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --log_step 100 --lr_model 1e-3 --n_encode_layers 4 --max_grad_norm 0 --run_name narmlp_gat_sl

CUDA_VISIBLE_DEVICES=1 python run.py --problem tspsl --model nar_attention --encoder gat --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --log_step 100 --lr_model 1e-3 --n_encode_layers 4 --max_grad_norm 0 --run_name narattn_gat_sl

### BigNet
CUDA_VISIBLE_DEVICES=0 python run.py --problem tspsl --model nar_attention --encoder gat --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.txt --val_dataset data/tsp/tsp20_val_concorde.txt --log_step 100 --lr_model 1e-3 --n_encode_layers 10 --hidden_dim 256 --hidden_dim 256  --max_grad_norm 0 --run_name narattn_gat_sl_bignet

### RL
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --model nar_attention --encoder gat --graph_size 20 --batch_size 512 --train_dataset data/tsp/tsp20_train_concorde.pkl --log_step 100 --lr_model 1e-3 --n_encode_layers 4 --max_grad_norm 0 --run_name narattn_gat_rl


CUDA_VISIBLE_DEVICES=1 python run.py --problem tspsl --model mlp --encoder gat --graph_size 50 --batch_size 512 --train_dataset data/tsp/tsp50_train_concorde.txt --val_dataset data/tsp/tsp50_val_concorde.txt --lr_model 1e-4 --log_step 100 --run_name nar_gat_sl
CUDA_VISIBLE_DEVICES=2,3 python run.py --problem tspsl --model mlp --encoder gat --graph_size 100 --batch_size 512 --train_dataset data/tsp/tsp100_train_concorde.txt --val_dataset data/tsp/tsp100_val_concorde.txt --lr_model 1e-4 --log_step 100 --run_name nar_gat_sl

# Evaluation

### GAT Model
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/xyz/ --decode_strategy greedy --eval_batch_size 200
CUDA_VISIBLE_DEVICES=0 python eval.py data/tsp/tsp20_test_concorde.txt data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt --model outputs/tspsl_20/xyz/ --decode_strategy bs --width 1280 --eval_batch_size 100
