# Docker file to create a container with everything needed to build AI models
#
# Build the container (once):
#
#  sudo docker build -t ai-models -f tools/docker/Dockerfile .
#
# Use the container (recommended shm settings for PyTorch DataLoader):
#
#  sudo docker run --gpus all --shm-size=8g -u $(id -u):$(id -g) -v $(pwd):/workspace -p 8888:8888 -it --rm ai-models
#
# Start Jupyter manually inside container when needed:
#
#  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=
#
# Optional alternative for Linux only (shares host IPC namespace):
#
#  sudo docker run --gpus all --ipc=host -u $(id -u):$(id -g) -v $(pwd):/workspace -it --rm ai-models bash
#
# Note: "docker run" must run from the top-level directory

FROM python:3.11-slim-bookworm

LABEL maintainer="Ngoc Ho" \
description="Container with everything needed to build AI models"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python-is-python3 \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    curl \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
    jupyter notebook ipykernel \
    numpy pandas matplotlib plotly tqdm \
    h5py pillow ipywidgets

RUN python -m pip install \
    torch torchvision torchaudio \
    gymnasium[box2d] pygame \
    opencv-python umap-learn torchsummary cma nexusformat

RUN python -m pip install \
    tensorboard wandb iopath pyyaml submitit braceexpand webdataset \
    timm transformers peft decord einops beartype psutil fire python-box \
    scikit-image ftfy

RUN python -m pip install "huggingface_hub[cli]"

EXPOSE 8888

# To reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER developer

CMD ["bash"]
