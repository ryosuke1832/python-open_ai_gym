# src/__init__.py

# バージョン情報やその他の基本情報を提供
__version__ = "0.1.0"
__author__ = "Ryosuke Yamamoto"

# サブパッケージを明示的にインポート
from src import agents, common, environments

__all__ = ["agents", "common", "environments"]
