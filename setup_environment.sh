#!/bin/bash

# 必要なパッケージの確認とインストール
check_and_install_packages() {
    local packages=(
        "xvfb"
        "x11-utils"
        "python3-opengl"
        "libgl1-mesa-glx"
        "libgl1-mesa-dev"
        "xdpyinfo"
    )
    
    local missing_packages=()
    
    for pkg in "${packages[@]}"; do
        if ! dpkg -s "$pkg" &> /dev/null; then
            missing_packages+=("$pkg")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo "次のパッケージをインストールします: ${missing_packages[*]}"
        sudo apt-get update
        sudo apt-get install -y "${missing_packages[@]}"
        echo "パッケージのインストールが完了しました。"
    else
        echo "必要なパッケージはすべてインストール済みです。"
    fi
}

# 仮想ディスプレイのクリーンアップと設定
setup_virtual_display() {
    # 既存のXvfbプロセスを終了
    pkill Xvfb || true
    
    # ロックファイルを削除
    rm -f /tmp/.X1-lock
    rm -f /tmp/.X11-unix/X1
    
    # 新しいXvfbを起動
    Xvfb :1 -screen 0 1024x768x24 &
    export DISPLAY=:1
    
    echo "仮想ディスプレイが設定されました: DISPLAY=:1"
}

# メイン処理
echo "環境セットアップを開始します..."
check_and_install_packages
setup_virtual_display
echo "環境セットアップが完了しました。"