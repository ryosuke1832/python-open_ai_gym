#!/bin/bash

# Pipをアップグレード
pip install --upgrade pip

# 必要なPythonパッケージをインストール
pip install --no-cache-dir \
    numpy \
    matplotlib \
    gymnasium \
    stable-baselines3[extra] \
    torch \
    tensorboard \
    gym \
    gym[all] \
    gym-retro \
    ppaquette_gym_super_mario

# 追加のパッケージが必要な場合はここに追加