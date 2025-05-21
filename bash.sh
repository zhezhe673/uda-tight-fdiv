CUDA_VISIBLE_DEVICES=0 python main.py data/office31 \
        -d Office31 -b 32 -s D -t A --workers 2 -a resnet50 \
        --epochs 40 --seed 42   --learner_type fdaad \
        --learnable --transform_type affine --init_params '{"a": 1,"b": 0}'  \
        --divergence kl --pretrained  --lr 0.004 --weight_decay 0.0005 \
        --bottleneck-dim 1024 --iter_per_epoch 2000 --lr_gamma 0.0002 \
        --reg_coef 1.0 \
        --log_dir ./logs/office31/fdaad/affine/d-a &

