conda activate []

checkpoint_dir=[]

python3 train.py \
--checkpoint_dir ${checkpoint_dir} \
--max_epoch 1000 \
--lr 1e-3 \
--weight_decay 0.00005 \
--batch_size 4 \
--val_batch_size 4 \
--iter_per_epoch 500 \
--print_freq 5000 \
--summary_freq 5000 \
--patch_size 512 \
--AWGN_sigma 300 

