# Ubuntu 20.04を使用
FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

# 基本的なパッケージを事前インストール
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
       python3 \
       python3-pip \
       python3-tk \
       python3-dev \
       python-is-python3 \
       cmake \
       zlib1g-dev \
       fceux \
       swig \
       xvfb \
       x11-utils \
       python3-opengl \
       libgl1-mesa-glx \
       libgl1-mesa-dev \
       libglu1-mesa \
       libglu1-mesa-dev \
       libglew-dev \
       freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*