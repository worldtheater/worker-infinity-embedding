FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume

# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# install uv
RUN pip install uv

# install python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system

# install torch from the stable CUDA 12.4 wheel index. The test index can lag or
# miss transitive NVIDIA wheels such as nvidia-cudnn-cu12.
RUN pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# Add src files
ADD src .

# Add test input
COPY test_input.json /test_input.json

# start the handler
CMD python -u /handler.py
