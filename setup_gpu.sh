#!/bin/bash
echo "Installing main framework..."
pip install -e .

echo "Fixing dependency conflicts (wandb, tensorboard, numpy, scipy)..."
pip install --upgrade pip setuptools wheel
pip install wandb==0.16.6 tensorboard==2.15.1 numpy==1.26.4 scipy==1.12.0 "protobuf==4.25.3" "onnx==1.16.1" wandb

echo "Environment ready for training!"
