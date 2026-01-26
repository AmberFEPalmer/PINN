Running Python Projects on Isambard AI with Docker

A comprehensive guide for deploying and running Dockerized Python applications on Isambard AI HPC using Singularity/Apptainer.

Overview

This repository demonstrates:

Packaging Python projects in Docker
Building multi-platform Docker images for HPC
Deploying to Isambard AI using Singularity
Submitting GPU-accelerated batch jobs
Example Project: SEIR Physics-Informed Neural Network

Python 3.11.8, TensorFlow 2.20.0, Keras 3.12.0
Part 1: Prepare Docker Image

1.1 Create Dockerfile

FROM python:3.11.8-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Default command
CMD ["python", "env_check.py"]
1.2 Create .dockerignore

.git
__pycache__/
*.pyc
data/
*.npy
*.png
.DS_Store
1.3 Build Multi-Platform Image

Critical: Isambard requires linux/amd64. Apple Silicon Macs need Docker Buildx:

# One-time setup
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap

# Build for AMD64 (Isambard) and ARM64 (Mac)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t <dockerhub-username>/<repo-name>:1.0 \
  --push \
  .
Why? Without --platform linux/amd64, Singularity fails:

FATAL: no child with platform linux/amd64 in index
Part 2: Deploy on Isambard

2.1 Login

ssh username@login.isambard.ac.uk
2.2 Convert Docker to Singularity

mkdir -p ~/project
cd ~/project

# Pull from Docker Hub → convert to .sif
singularity build app.sif docker://<dockerhub-username>/<repo-name>:1.0
2.3 Test Interactively

# Request compute node
srun --gpus=1 --time=00:30:00 --pty /bin/bash --login

# Check environment
singularity exec --pwd /app app.sif python env_check.py

# Run with mounted data
singularity exec --nv \
  --bind data:/app/data \
  --pwd /app \
  app.sif python script.py
Key flags:

--pwd /app - Set working directory
--bind host:container - Mount directories
--nv - Enable NVIDIA GPU
Part 3: Batch Jobs

3.1 Create job.sh

#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --output=output-%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00

module load singularity

singularity exec --nv \
  --bind $HOME/project/data:/app/data \
  --pwd /app \
  $HOME/project/app.sif \
  python script.py
3.2 Submit

sbatch job.sh
squeue --me
tail -f output-*.out
Part 4: Update Workflow

# 1. Update code locally
git commit -am "Update model"
git push

# 2. Rebuild Docker image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t username/repo:1.1 \
  --push \
  .

# 3. Update on Isambard
singularity build app-v1.1.sif docker://username/repo:1.1
Docker vs Singularity

Feature	Docker	Singularity
Root required	Yes	No
Image format	Layers	Single .sif
Working dir	WORKDIR	Host dir (use --pwd)
GPU	--gpus	--nv
Volumes	-v	--bind
Common Issues

Platform mismatch

docker buildx build --platform linux/amd64,linux/arm64 ...
File not found

singularity exec --pwd /app app.sif python script.py
Data inaccessible

singularity exec --bind /path/to/data:/app/data ...
GPU not detected

Add #SBATCH --gpus=1 in job script
Use --nv flag with singularity
Quick Reference

# Build multi-platform
docker buildx build --platform linux/amd64,linux/arm64 -t user/img:tag --push .

# Deploy on Isambard
singularity build app.sif docker://user/img:tag

# Interactive
srun --gpus=1 --time=00:30:00 --pty /bin/bash
singularity exec --nv --bind data:/app/data --pwd /app app.sif python script.py

# Batch
sbatch job.sh
Resources

Singularity Docs
Isambard Guide
Docker Buildx
About This Project

SEIR PINN for COVID-19 modeling using physics-informed neural networks.

Repository: https://github.com/zixuan-liu-17/PINN
Docker Hub: https://hub.docker.com/r/zixuanliu17/pinn-env