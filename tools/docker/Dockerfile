# =============================================================================
# Docker file to create a container with everything needed to build AI models
# Base: Ubuntu 22.04 + CUDA 11.8 + cuDNN 8 + Python 3.11 + full AI dependencies
#
# Build the container (once):
#  sudo docker build -t ai-base -f utils/docker/Dockerfile.ai-base .
# Build the container with a custom base image (optional):
#  sudo docker build -t ai-base -f utils/docker/Dockerfile.ai-base --build-arg CUDA_BASE_IMAGE=custom/image:tag .
#
# Use the container (recommended shm settings for PyTorch DataLoader):
#  sudo docker run --gpus all --shm-size=8g -u $(id -u):$(id -g) -v $(pwd):/workspace -p 8888:8888 -it --rm ai-base
# Start Jupyter manually inside container when needed:
#  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Use the container for TensorBoard (if running TensorBoard inside container):
#  sudo docker run --gpus all --shm-size=8g -u $(id -u):$(id -g) -v $(pwd):/workspace -p 6006:6006 -it --rm ai-base
# =============================================================================

ARG CUDA_BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

LABEL maintainer="Ngoc Ho" \
description="Container with everything needed to build AI models"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        locales tzdata \
        curl gnupg lsb-release git wget \
        software-properties-common \
        build-essential \
        ca-certificates \
        ffmpeg \
        swig \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libsdl2-2.0-0 \
        libsdl2-dev \
        libjpeg-dev \
        zlib1g-dev \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        proj-bin \
        proj-data \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get remove -y python3-sympy 2>/dev/null || true && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --upgrade pip setuptools wheel

RUN python3.11 -m pip install \
    jupyter notebook ipykernel \
    numpy pandas matplotlib plotly tqdm \
    h5py pillow ipywidgets

RUN python3.11 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN python3.11 -m pip install \
    gymnasium[box2d] pygame \
    opencv-python umap-learn torchsummary cma nexusformat

RUN python3.11 -m pip install \
    tensorboard wandb iopath pyyaml submitit braceexpand webdataset \
    timm transformers peft decord einops beartype psutil fire python-box \
    scikit-image ftfy \
    squaternion \
    empy==3.3.4 \
    catkin-pkg \
    lark

# For downloading models/checkpoints from Hugging Face Hub - Drive-JEPA
RUN python3.11 -m pip install "huggingface_hub[cli]"

# For running the evaluation script - Drive-JEPA
RUN python3.11 -m pip install hydra-core

# For NavSim / nuPlan based evaluation pipelines
# ──────────────────────────────────────────────────────────────────────────────
# nuplan-devkit v1.2 transitively pulls navsim==2.0.0 from PyPI which hard-pins
# numpy==1.23.4; that conflicts with opencv-python>=4.9 on Python 3.11 which
# requires numpy>=1.23.5. Fix: install everything with --no-deps so pip never
# attempts to resolve the navsim/numpy constraint. navsim itself MUST be
# installed at container runtime from the mounted source:
#   cd /workspace/vf-drive-jepa/navsim && pip install -e .
# ──────────────────────────────────────────────────────────────────────────────
# Only nuplan-devkit is installed --no-deps because its requirements.txt
# pulls navsim==2.0.0 which hard-pins numpy==1.23.4, conflicting with
# opencv-python 4.9+ on Python 3.11 (needs numpy>=1.23.5).
# All other packages are installed normally so their own deps (boto3, etc.) resolve.
RUN python3.11 -m pip install --no-deps \
    "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

RUN python3.11 -m pip install \
    aioboto3 aiofiles \
    bokeh casadi control \
    Fiona geopandas guppy3 \
    nest_asyncio \
    pyarrow pyinstrument pyogrio pyquaternion \
    rasterio "ray[default]" retry rtree \
    selenium \
    Shapely SQLAlchemy sympy tornado ujson \
    scikit-learn "positional-encodings==6.0.1" \
    "pytorch-lightning==2.2.1"

RUN python3.11 -m pip install pytest

RUN python3.11 -m pip install pybullet

# For FlowFormer++ usage
RUN python3.11 -m pip install \
    yacs \
    loguru

WORKDIR /workspace

RUN useradd -ms /bin/bash developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER developer

CMD ["bash", "--login"]
