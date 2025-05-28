#!/bin/bash

# 実用的な環境セットアップ（Python 3.13 + 最新版対応）
echo "Setting up Parler TTS experiment environment (Practical Version)..."

# Python 3.13での互換性を考慮した設定
echo "Current Python version:"
python --version

# 既存パッケージクリーンアップ
echo "Cleaning up existing packages..."
pip uninstall -y torch torchvision torchaudio coremltools transformers numpy

# pip を最新に更新
pip install --upgrade pip

# Python 3.13 対応の最新PyTorch（利用可能な最新版）
echo "Installing PyTorch (latest compatible version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core ML Tools（最新安定版、警告は無視）
echo "Installing Core ML Tools (latest stable)..."
pip install coremltools

# NumPy（Python 3.13対応の最新版）
echo "Installing NumPy (Python 3.13 compatible)..."
pip install numpy

# Transformers（最新安定版）
echo "Installing Transformers..."
pip install transformers

# 基本ライブラリ
echo "Installing basic libraries..."
pip install matplotlib seaborn
pip install tqdm
pip install scipy

# プロファイリング用（オプション）
echo "Installing profiling tools..."
pip install psutil

echo "✅ Practical environment setup completed!"
echo ""
echo "Checking installed versions:"
python -c "
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    print('❌ PyTorch: Not installed')

try:
    import coremltools
    print(f'✅ Core ML Tools: {coremltools.__version__}')
except ImportError:
    print('❌ Core ML Tools: Not installed')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError:
    print('❌ Transformers: Not installed')

try:
    import numpy
    print(f'✅ NumPy: {numpy.__version__}')
except ImportError:
    print('❌ NumPy: Not installed')

try:
    import matplotlib
    print(f'✅ Matplotlib: {matplotlib.__version__}')
except ImportError:
    print('❌ Matplotlib: Not installed')
"

echo ""
echo "If warnings appear about Core ML, they can be ignored for now."
echo "The experiment will use fallback implementations."
