# -*- coding: utf-8 -*-
import random
import time
import warnings
import argparse
import logging
import os
import json

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn.utils.spectral_norm as sn
import numpy as np

import utils
from utils import seed_all, test_accuracy, scheduler
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter
from tllib.utils.analysis import collect_feature, tsne, a_distance

from fDAAD import fDAADLearner
from model.resnet import *
from model.LeNet import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    seed_all(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    log_filename = os.path.join(
        args.log_dir,
        f"{args.data}_{args.learner_type}_{args.divergence}_{args.source[0]}_"
        f"{args.target[0]}_{args.transform_type}_{args.init_params}_"
        f"seed{args.seed}_coef{args.reg_coef}.log"
    )
    
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_filename, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Arguments: {args}")
    logger.info("Loading data and building data loaders...")

    # Prepare data transforms
    train_transform = utils.get_train_transform(
        args.train_resizing, scale=args.scale, ratio=args.ratio,
        random_horizontal_flip=not args.no_hflip,
        random_color_jitter=False, resize_size=args.resize_size,
        norm_mean=args.norm_mean, norm_std=args.norm_std
    )
    val_transform = utils.get_val_transform(
        args.val_resizing, resize_size=args.resize_size,
        norm_mean=args.norm_mean, norm_std=args.norm_std
    )
    print("Train transform:", train_transform)
    print("Validation transform:", val_transform)

    # Load datasets
    (train_source_dataset, train_target_dataset,
     val_dataset, test_dataset,
     num_classes, args.class_names) = utils.get_dataset(
        args.data, args.root, args.source, args.target,
        train_transform, val_transform
    )
    train_source_loader = DataLoader(
        train_source_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True
    )
    train_target_loader = DataLoader(
        train_target_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers
    )

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    logger.info("Building model...")
    print(f"=> Using model architecture '{args.arch}'")

    # Select feature extractor
    if args.arch == 'lenet':
        feature_extractor = LeNet()
        for param in feature_extractor.parameters():
            param.requires_grad = True
    elif args.arch == 'resnet50':
        feature_extractor = resnet50(pretrained=True)
        for param in feature_extractor.parameters():
            param.requires_grad = True  
    elif args.arch == 'resnet101':
        feature_extractor = resnet101(pretrained=True)
        for param in feature_extractor.parameters():
            param.requires_grad = True

    # Bottleneck layer
    bottleneck = nn.Sequential(
        nn.Linear(feature_extractor.out_features, args.bottleneck_dim),
        nn.BatchNorm1d(args.bottleneck_dim),
        nn.ReLU(),
        nn.Dropout(0.5)
    )
    backbone = nn.Sequential(OrderedDict([
        ('feature_extractor', feature_extractor),
        ('bottleneck', bottleneck)
    ]))

    # Classification head
    taskhead = nn.Sequential(
        sn(nn.Linear(args.bottleneck_dim, args.bottleneck_dim)),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        sn(nn.Linear(args.bottleneck_dim, num_classes))
    )

    # Initialize weights
    bottleneck[0].weight.data.normal_(0, 0.005)
    bottleneck[0].bias.data.fill_(0.1)
    taskhead[0].weight.data.normal_(0, 0.01)
    taskhead[0].bias.data.fill_(0.0)
    taskhead[-1].weight.data.normal_(0, 0.01)
    taskhead[-1].bias.data.fill_(0.0)
    
    logger.info(f"Learner type: {args.learner_type}")
    logger.info("Initializing learner...")
    LearnerClass = fDAADLearner
    taskloss = nn.CrossEntropyLoss()
    
    learner = LearnerClass(
        backbone, taskhead, taskloss,
        divergence=args.divergence, reg_coef=args.reg_coef,
        n_classes=num_classes, grl_params={
            "max_iters": args.iter_per_epoch,
            "hi": 0.1, "auto_step": True
        },
        learnable=args.learnable,
        transform_type=args.transform_type,
        init_params=args.init_params,
    ).to(device)

    optimizer = SGD(
        [
            {"params": learner.backbone.feature_extractor.parameters(), "lr_mult": 0.1},
            {"params": learner.backbone.bottleneck.parameters(),       "lr_mult": 1.0},
            {"params": learner.taskhead.parameters(),                   "lr_mult": 1.0},
            {"params": learner.auxhead.parameters(),                    "lr_mult": 1.0},
            {"params": learner.fdaad_divhead.fdaad_loss.parameters(),     "lr_mult": 1.0},
        ],
        lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )
    lr_scheduler = scheduler(optimizer, args.lr,
                             decay_step_=0.75,
                             gamma_=args.lr_gamma)
    
    # Analysis phase: feature extraction and visualization
    if args.phase == 'analysis':
        best_model_filename = os.path.join(
            args.save_dir,
            f"{args.data}_{args.learner_type}_{args.divergence}_"
            f"{args.source[0]}_{args.target[0]}_{args.transform_type}_"
            f"{args.init_params}_seed{args.seed}_coef{args.reg_coef}_best.pt"
        )
        network, _ = learner.get_reusable_model(False)
        checkpoint = torch.load(best_model_filename)
        keys = ["feature_extractor", "bottleneck"]
        filtered = {
            k[len("0."):]: v for k, v in checkpoint.items()
            if any(k.startswith(f"0.{key}") for key in keys)
        }
        network.load_state_dict(filtered, strict=True)
        network.to(device)

        source_feature = collect_feature(train_source_loader, network, device)
        target_feature = collect_feature(train_target_loader, network, device)
        tSNE_filename = os.path.join(
            args.visual_dir,
            f"{args.data}_{args.learner_type}_{args.divergence}_"
            f"{args.source[0]}_{args.target[0]}_{args.transform_type}_"
            f"{args.init_params}_seed{args.seed}_coef{args.reg_coef}.pdf"
        )
        tsne.visualize(source_feature, target_feature, tSNE_filename, vis_ratio=args.vis_ratio)
        print("Saved t-SNE plot to", tSNE_filename)
        return

    # Test-only phase
    if args.phase == 'test':
        best_model_filename = os.path.join(
            args.save_dir,
            f"{args.data}_{args.learner_type}_{args.divergence}_"
            f"{args.source[0]}_{args.target[0]}_{args.transform_type}_"
            f"{args.init_params}_seed{args.seed}_coef{args.reg_coef}_best.pt"
        )
        network = learner.get_reusable_model(True)
        network.load_state_dict(torch.load(best_model_filename))
        acc1 = utils.validate(test_loader, network, args, device)
        print(f"Test accuracy: {acc1:.4f}")
        return
    
    logger.info("Starting training...")
    best_acc = -np.inf
    for epoch in range(args.epochs):
        train(train_source_iter, train_target_iter, learner,
              optimizer, lr_scheduler, epoch, args, logger)
        
        val_acc, val_loss = test_accuracy(
            learner.get_reusable_model(True), val_loader, taskloss, device
        )
        latest_model_filename = os.path.join(
            args.save_dir,
            f"{args.data}_{args.learner_type}_{args.divergence}_"
            f"{args.source[0]}_{args.target[0]}_{args.transform_type}_"
            f"{args.init_params}_seed{args.seed}_coef{args.reg_coef}_latest.pt"
        )
        torch.save(learner.get_reusable_model(True).state_dict(), latest_model_filename)
        
        if val_acc > best_acc:
            best_model_filename = os.path.join(
                args.save_dir,
                f"{args.data}_{args.learner_type}_{args.divergence}_"
                f"{args.source[0]}_{args.target[0]}_{args.transform_type}_"
                f"{args.init_params}_seed{args.seed}_coef{args.reg_coef}.pt"
            )
            torch.save(learner.get_reusable_model(True).state_dict(), best_model_filename)
            best_acc = val_acc

        logger.info(
            f"[Epoch {epoch}] Val Acc: {val_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Best Val Acc: {best_acc:.4f}"
        )

    # Final test
    network = learner.get_reusable_model(True)
    network.load_state_dict(torch.load(best_model_filename))
    test_acc, _ = test_accuracy(network, test_loader, taskloss, device)
    logger.info(f"Model saved to: {best_model_filename}")
    logger.info(f"Final best test accuracy: {test_acc:.4f}")
    logger.info("Training complete.")

    # Save run results
    results_file = os.path.join(args.log_dir, "results.txt")
    record = {
        "params": vars(args),
        "best_acc": best_acc
    }
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Results recorded in {results_file}")


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          learner, optimizer: SGD, lr_scheduler, epoch: int,
          args: argparse.Namespace, logger):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')

    learner.train()
    end = time.time()
    for i in range(args.iter_per_epoch):
        x_s, labels_s = next(train_source_iter)[:2]
        x_t, = next(train_target_iter)[:1]
        x_s, x_t, labels_s = x_s.to(device), x_t.to(device), labels_s.to(device)

        data_time.update(time.time() - end)

        loss, others = learner((x_s, x_t), labels_s)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 10)
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(
                f"[Epoch {epoch}/{args.epochs}] Iter {i}/{args.iter_per_epoch} | "
                f"BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f}) | "
                f"DataTime {data_time.val:.3f} ({data_time.avg:.3f}) | "
                f"TaskLoss {others['taskloss']:.4f} | "
                f"f-DAAD Src {others.get('fdaad_src', -1):.4f} | "
                f"f-DAAD Trg {others.get('fdaad_trg', -1):.4f}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fDAAD for Unsupervised Domain Adaptation')
    # Dataset settings
    parser.add_argument('--learner_type', type=str, default='fdd',
                        choices=['fdal', 'fdaad', 'fdd'],
                        help='Type of learner to use')
    parser.add_argument('--divergence', type=str, default='kl',
                        help='Type of divergence (e.g. kl, pearson)')
    parser.add_argument('root', metavar='DIR',
                        help='Root directory of the dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        choices=utils.get_dataset_names(),
                        help='Dataset name')
    parser.add_argument('-s', '--source', nargs='+', required=True,
                        help='Source domain(s)')
    parser.add_argument('-t', '--target', nargs='+', required=True,
                        help='Target domain(s)')
    parser.add_argument('--train-resizing', type=str, default='default',
                        help='Resizing method for training data')
    parser.add_argument('--val-resizing', type=str, default='default',
                        help='Resizing method for validation data')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='Image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0],
                        help='Random resize scale range')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.],
                        help='Random resize aspect ratio range')
    parser.add_argument('--no-hflip', action='store_true',
                        help='Disable random horizontal flip during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406),
                        help='Normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225),
                        help='Normalization std')
    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='Backbone architecture')
    parser.add_argument('--bottleneck-dim', type=int, default=256,
                        help='Dimension of bottleneck layer')
    parser.add_argument('--trade-off', type=float, default=1.0,
                        help='Trade-off hyperparameter for transfer loss')
    # Training parameters
    parser.add_argument('--reg_coef', type=float, default=1.0,
                        help='Regularization coefficient')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('--init_params', type=json.loads,
                        default='{"a": 1.0, "b": 0.0}',
                        help='Initialization parameters for transform')
    parser.add_argument('--learnable', action='store_true',
                        help='Enable learnable transform parameters')
    parser.add_argument('--transform_type', type=str, default='affine',
                        choices=['affine', 'power', 'exponential', 'sigmoid'],
                        help='Type of τ-transform')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.001,
                        help='LR scheduler gamma')
    parser.add_argument('--lr_decay', type=float, default=0.75,
                        help='LR scheduler decay factor')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay')
    parser.add_argument('-j', '--workers', type=int, default=2,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('-i', '--iter_per_epoch', type=int, default=1000,
                        help='Iterations per epoch')
    parser.add_argument('-p', '--print-freq', type=int, default=100,
                        help='Logging frequency')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--per_class_eval', action='store_true',
                        help='Output per-class accuracy during evaluation')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone if set')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for log files')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--visual_dir', type=str, default='./visual',
                        help='Directory for saving visualizations')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test', 'analysis'],
                        help="Phase: 'train', 'test', or 'analysis'")
    parser.add_argument('--vis_ratio', type=float, default=1.0,
                        help='Visualization sampling ratio')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for fDAAD')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for fDAAD')
    args = parser.parse_args()
    main(args)
