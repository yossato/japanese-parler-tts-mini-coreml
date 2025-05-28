#!/bin/bash

# Parler TTS ANE実験の最終セットアップ
echo "Setting up Parler TTS ANE Experiment..."

# 実行権限を設定
chmod +x setup_environment.sh
chmod +x run_experiment.py

# Phase別スクリプトにも実行権限
find . -name "*.py" -exec chmod +x {} \;

echo "✅ Setup completed!"
echo ""
echo "🚀 Quick Start:"
echo "1. Install dependencies: ./setup_environment.sh"
echo "2. Run all phases: python run_experiment.py --all"
echo "3. Analyze results: python run_experiment.py --analyze"
echo ""
echo "📖 For detailed instructions, see README.md"
