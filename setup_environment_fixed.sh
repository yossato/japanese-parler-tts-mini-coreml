#!/bin/bash

# Parler TTS ANE実験用の修正された環境セットアップスクリプト
echo "Setting up Parler TTS experiment environment (Fixed Version)..."

# 現在のPython環境を確認
echo "Current Python version:"
python --version

echo "Current pip version:"
pip --version

# 既存の問題のあるパッケージをアンインストール
echo "Cleaning up existing packages..."
pip uninstall -y torch torchvision torchaudio coremltools transformers

# 互換性のあるバージョンをインストール
echo "Installing compatible PyTorch (2.5.0)..."
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core ML Tools (互換性のあるバージョン)
echo "Installing Core ML Tools..."
pip install coremltools==7.2

# Transformers (最新安定版)
echo "Installing Transformers..."
pip install transformers==4.45.0

# 他の必要なライブラリ
echo "Installing other required packages..."
pip install numpy==1.24.3
pip install matplotlib seaborn
pip install librosa soundfile
pip install tqdm
pip install protobuf==3.20.3

# オーディオ処理用（必要に応じて）
pip install scipy

# プロファイリング用
pip install torch-tb-profiler

echo "✅ Environment setup completed with compatible versions!"
echo ""
echo "Installed versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import coremltools; print(f'Core ML Tools: {coremltools.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "To activate the environment:"
echo "cd /Users/yoshiaki/Projects/parler_tts_experiment"
echo "source .venv/bin/activate"
