#!/bin/bash

# エラー発生時に終了
set -e

# パッケージ管理ツールのアップデート
apt-get update -y
apt-get upgrade -y

# システムパッケージのインストール（X11関連のパッケージを追加）
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-tk \
    cmake \
    zlib1g-dev \
    swig \
    fceux \
    xvfb \
    x11-utils \
    python3-opengl \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglew-dev \
    freeglut3-dev \
    xdpyinfo

# Pythonパッケージをインストール
pip3 install --upgrade pip
pip3 install matplotlib
pip3 install gym==0.21.0
pip3 install gym[atari]==0.21.0
pip3 install gym-super-mario-bros
pip3 install tensorflow
pip3 install numpy

# キャッシュクリア
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "OpenAI Gym環境のセットアップが完了しました"