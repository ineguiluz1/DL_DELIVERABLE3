# Fine-tuning StyleGAN2 for The Simpsons Dataset

This guide explains how to set up and fine-tune StyleGAN2-ADA for generating Simpsons-style images. More detailed information in 

## Environment Setup

First, create and activate a new Conda environment with the required dependencies:

```bash
# Create and activate conda environment
conda create -n stylegan2ada python=3.8 -y
conda activate stylegan2ada

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Clone the StyleGAN2-ADA repository
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
cd stylegan2-ada-pytorch

# Install additional requirements
pip install -r requirements.txt
pip install click requests tqdm pyspng ninja click imageio-ffmpeg==0.4.3
```

Verify CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Data Preparation

Prepare your dataset using the provided tool:

```bash
python dataset_tool.py --source=/path/to/your/images --dest=./datasets/simpsons64.zip
```

## Training Process

The training is performed in multiple stages:

1. Initial fine-tuning from FFHQ pre-trained model:
```bash
python train.py --outdir=./training-runs \
                --data=./datasets/simpsons64.zip \
                --cfg=stylegan2 \
                --batch=8 \
                --snap=4 \
                --kimg=400 \
                --resume=ffhq.pkl \
                --gpus=1
```

2. Continue training from the previous checkpoint:
```bash
python train.py --outdir=./training-runs \
                --data=./datasets/simpsons64.zip \
                --cfg=stylegan2 \
                --batch=8 \
                --snap=4 \
                --kimg=800 \
                --resume=model_2.pkl \
                --gpus=1
```

## Generating Images

Generate images using the trained model:

```bash
python generate.py --outdir=out2 \
                  --trunc=1 \
                  --seeds=0-9 \
                  --network=/path/to/your/network-snapshot.pkl
```

## Evaluation

### Computing FID Score
```bash
python calc_metrics.py --metrics=fid50k_full \
                      --data=./datasets/simpsons64.zip \
                      --network=./training-runs/your-model-snapshot.pkl
```

### Computing Inception Score
```bash
python calc_metrics.py --metrics=is50k \
                      --data=./datasets/simpsons64.zip \
                      --network=./training-runs/your-model-snapshot.pkl
```

## Notes

- The training process uses a batch size of 8, suitable for most GPUs with 8GB+ VRAM
- Training is performed in stages, with checkpoints saved every 4 snapshots
- The model is fine-tuned from the FFHQ pre-trained model for better results
- Image generation can be controlled using the truncation parameter (--trunc)