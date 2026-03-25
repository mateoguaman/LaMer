# Single Docker image for LaMer + language-table on SageMaker.
# Contains two conda environments:
#   - ltvenv (Python 3.10): language-table env servers (JAX/TF/PyBullet)
#   - lamer  (Python 3.12): LaMer training (PyTorch/vLLM/Ray/FSDP)
#
# Build:
#   docker build -t lamer-langtable -f scripts/sagemaker/Dockerfile .

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages (PyBullet needs GL libs for headless rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        netcat-openbsd \
        libgl1-mesa-glx \
        libegl1-mesa \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget -q -O /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/miniforge3 \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/miniforge3/bin:${PATH}"

# --- language-table environment (Python 3.10, JAX/TF) ---
RUN conda create -y -n ltvenv python=3.10 \
    && conda clean -afy

COPY language-table /opt/language-table

RUN /opt/miniforge3/envs/ltvenv/bin/pip install --no-cache-dir \
        -r /opt/language-table/requirements.txt \
    && /opt/miniforge3/envs/ltvenv/bin/pip install --no-cache-dir --no-deps \
        git+https://github.com/google-research/scenic.git@ae21d9e884015aa7bc7cf1d489af53d16c249726 \
    && /opt/miniforge3/envs/ltvenv/bin/pip cache purge 2>/dev/null || true

# --- LaMer environment (Python 3.12, PyTorch/vLLM) ---
RUN conda create -y -n lamer python=3.12 \
    && conda clean -afy

COPY requirements.txt /tmp/lamer-requirements.txt

RUN /opt/miniforge3/envs/lamer/bin/pip install --no-cache-dir \
        -r /tmp/lamer-requirements.txt \
    && /opt/miniforge3/envs/lamer/bin/pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    && /opt/miniforge3/envs/lamer/bin/pip install --no-cache-dir setuptools==74.1.3 \
    && /opt/miniforge3/envs/lamer/bin/pip cache purge 2>/dev/null || true

# sagemaker-training toolkit: handles source code download from S3 and
# entry_point execution when SageMaker runs `docker run <image> train`.
# LaMer code is uploaded via source_dir at launch time (not baked into image),
# so you only need to rebuild when dependencies change.
RUN pip install --no-cache-dir sagemaker-training

WORKDIR /opt/ml/code
