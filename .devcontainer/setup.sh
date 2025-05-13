#!/bin/bash

# エラー発生時に終了
set -e

# パッケージ管理ツールのアップデート
apt-get update -y
apt-get upgrade -y

# システムパッケージのインストール
apt-get install -y --no-install-recommends \
    python3-pip \
    python3-tk \
    cmake \
    zlib1g-dev \
    swig \
    fceux

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