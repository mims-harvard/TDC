FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        wget \
        curl \
        bash \
        bzip2 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MiniConda
RUN curl -LO --silent https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -p /miniconda -b && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh

ENV PATH=/miniconda/bin:${PATH}

# RDKit
RUN conda install -y -q -c rdkit rdkit=2018.09.1.0

# python deps
RUN pip install joblib \
                tqdm \
                scipy
