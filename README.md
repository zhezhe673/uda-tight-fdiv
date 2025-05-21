# Code for Adaptive \$f\$-Divergence Domain Adaptation

**GitHub Repository:** [https://github.com/zhezhe673/uda-tight-fdiv](https://github.com/zhezhe673/uda-tight-fdiv)

This example demonstrates how to run the fDAAD domain adaptation method for image classification using the [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) framework.

## Repository Structure

```
Transfer-Learning-Library/
└── examples/
    └── domain_adaptation/
        └── image_classification/
            ├── main.py           # Entry point for training and evaluation
            ├── fDAAD.py          # fDAAD implementation
            ├── utils/            # Utility modules provided by the library
            ├── model/            # Pre-defined model architectures (ResNet, LeNet, etc.)
            └── ...               # Other library files
```

> **Note:** Copy `main.py` and `fDAAD.py` from this example into `examples/domain_adaptation/image_classification/` of your local clone of the Transfer-Learning-Library.

## Requirements

* Python 3.7+
* PyTorch 1.7+
* torchvision
* numpy
* tllib (Transfer-Learning-Library core)

Install dependencies with:

```bash
pip install torch torchvision numpy
# Then install the Transfer-Learning-Library:
cd Transfer-Learning-Library
pip install -e .
```

## Usage

After placing `main.py` and `fDAAD.py` into the `examples/domain_adaptation/image_classification/` directory, you can run training or evaluation:

```bash
# Example: Office31 D → A adaptation
CUDA_VISIBLE_DEVICES=0 python main.py data/office31 \
    -d Office31 -b 32 -s D -t A --workers 2 -a resnet50 \
    --epochs 40 --seed 42 --learner_type fdaad \
    --learnable --transform_type affine --init_params '{"a":1,"b":0}' \
    --divergence kl --pretrained --lr 0.004 --weight_decay 0.0005 \
    --bottleneck-dim 1024 --iter_per_epoch 2000 --lr_gamma 0.0002 \
    --reg_coef 1.0 \
    --log_dir ./logs/office31/fdaad/affine/d-a
```

### Parameters

* `--data`, `-s`, `-t`: dataset name and source/target domains
* `--arch`: backbone network (e.g., `resnet50`, `resnet101`, `lenet`)
* `--learner_type`: `fdaad` selects the fDAAD learner
* `--transform_type` & `--init_params`: configure the adaptive τ-transform
* Standard training options: learning rate, weight decay, batch size, epochs, etc.

## Output

* Logs are saved under `--log_dir`
* Model checkpoints under `--save_dir`
* t-SNE visualizations (if `--phase analysis`) under `--visual_dir`

## Citation

If you use fDAAD in your research, please cite:

> Yi et al., "Feature Decomposed Adaptive Alignment Divergence for Unsupervised Domain Adaptation", *NeurIPS 2025*.

## License

This example inherits the MIT License of the Transfer-Learning-Library. See [LICENSE](https://github.com/thuml/Transfer-Learning-Library/blob/master/LICENSE) for details.
