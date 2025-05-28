#!/bin/bash

# 実験用環境セットアップスクリプト
echo "Setting up Parler TTS experiment environment..."

# venvを有効化 (既に作成済みと仮定)
# source venv/bin/activate

# 必要なライブラリをインストール
echo "Installing required packages..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core ML Tools
pip install coremltools>=7.0

# プロファイリング用
pip install torch-tb-profiler

# オーディオ処理
pip install librosa soundfile

# 可視化
pip install matplotlib seaborn

# その他のユーティリティ
pip install tqdm

echo "Environment setup complete!"
echo ""
echo "To activate the environment:"
echo "cd /Users/yoshiaki/Projects/parler_tts_experiment"
echo "source venv/bin/activate"
