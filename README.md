# OpenAI Gym 強化学習環境

OpenAI Gym（現在はGymnasium）を使用して強化学習の実験を行うための開発環境です。
Dev Containerを使用して、Docker上で簡単に環境構築できます。

## 環境

| 項目 | バージョン |
|-----|-----------|
| Python | 3.10 |
| Gymnasium | 最新版 |
| Stable Baselines3 | 最新版 |

## 含まれるサンプルプログラム

- `src/cartpole_sample.py` - CartPole-v1の実行サンプル
- `src/mario_sample.py` - スーパーマリオブラザーズの実行サンプル

## 使い方

1. VS Codeで「Dev Containerで開く」を選択
2. コンテナのビルドが完了するまで待機
3. サンプルプログラムを実行:
   ```bash
   python src/cartpole_sample.py
   ```

## インストール済みのGym環境

- 基本的なGym環境（CartPole, MountainCar, Pendulumなど）
- Atari Games（Breakout, MsPacmanなど）
- レトロゲーム（Gym Retro）
- スーパーマリオブラザーズ

## 拡張機能

- Python
- Python Indent
- autoDocstring
- Ruff (Linter)
- IntelliCode

## 注意点

- GUI表示のために、リモート接続時はポート転送が必要です
- スーパーマリオブラザーズの実行には別途設定が必要な場合があります