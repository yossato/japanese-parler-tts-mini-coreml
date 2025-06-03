#!/usr/bin/env python3
"""
Phase 1: TTSデコーダーの固定長出力への簡略化（最小限版）
依存関係を最小限に抑えた純粋PyTorch実装
"""

import torch
import torch.nn as nn
import time
import json
import os
import sys

# matplotlib は optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available, skipping plots")

class MinimalFixedLengthDecoder(nn.Module):
    """最小限の固定長デコーダー"""
    
    def __init__(self, input_size=1024, output_size=500, num_codebooks=9, vocab_size=1088):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        
        # シンプルな全結合ネットワーク
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size * num_codebooks * vocab_size)
        )
        
        print(f"✅ Minimal decoder initialized: {input_size} -> {output_size}x{num_codebooks}x{vocab_size}")
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, input_size = x.shape
        
        # Global average pooling
        pooled = x.mean(dim=1)  # [batch_size, input_size]
        
        # Forward pass
        output = self.encoder(pooled)  # [batch_size, output_size * num_codebooks * vocab_size]
        
        # Reshape
        output = output.view(batch_size, self.output_size, self.num_codebooks, self.vocab_size)
        
        # Get token IDs
        tokens = torch.argmax(output, dim=-1)  # [batch_size, output_size, num_codebooks]
        
        return {
            'sequences': tokens.view(batch_size * self.num_codebooks, self.output_size),
            'logits': output,
            'tokens': tokens
        }

class SimpleTextEncoder(nn.Module):
    """シンプルなテキストエンコーダー"""
    
    def __init__(self, vocab_size=32000, hidden_size=1024, max_length=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )
        
        print(f"✅ Simple text encoder initialized: vocab={vocab_size}, hidden={hidden_size}")
    
    def encode_text(self, text):
        """テキストを簡単にエンコード"""
        # 文字レベルエンコーディング
        tokens = []
        for char in text[:self.max_length]:
            token_id = ord(char) % self.vocab_size
            tokens.append(token_id)
        
        # パディング
        while len(tokens) < self.max_length:
            tokens.append(0)
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # Transformer encoding
        encoded = self.encoder(embedded)
        
        return encoded

class MinimalTTSSystem:
    """最小限のTTSシステム"""
    
    def __init__(self):
        self.device = torch.device("cpu")
        
        # コンポーネント初期化
        self.text_encoder = SimpleTextEncoder()
        self.fixed_decoder = MinimalFixedLengthDecoder()
        
        print("✅ Minimal TTS System initialized")
    
    def synthesize(self, text):
        """テキストから音声コード生成"""
        
        with torch.no_grad():
            # テキストエンコーディング
            input_ids = self.text_encoder.encode_text(text)
            text_features = self.text_encoder(input_ids)
            
            # 固定長デコーディング
            audio_codes = self.fixed_decoder(text_features)
            
        return audio_codes
    
    def benchmark(self, texts, num_runs=3):
        """ベンチマーク実行"""
        
        results = {
            'texts': texts,
            'times': [],
            'outputs': []
        }
        
        print(f"\n🔥 Benchmarking with {len(texts)} texts, {num_runs} runs each")
        
        for i, text in enumerate(texts):
            print(f"\n  Text {i+1}: \"{text}\"")
            
            run_times = []
            for run in range(num_runs):
                start_time = time.time()
                output = self.synthesize(text)
                end_time = time.time()
                
                run_time = end_time - start_time
                run_times.append(run_time)
                
                if run == 0:
                    results['outputs'].append(output)
                    print(f"    Output shape: {output['sequences'].shape}")
            
            avg_time = sum(run_times) / len(run_times)
            results['times'].append(avg_time)
            print(f"    Average time: {avg_time:.4f}s")
        
        results['total_avg_time'] = sum(results['times']) / len(results['times'])
        print(f"\n📊 Overall average time: {results['total_avg_time']:.4f}s")
        
        return results

def compare_autoregressive_vs_fixed():
    """自己回帰 vs 固定長の概念的比較"""
    
    print("\n=== Autoregressive vs Fixed Length Comparison ===")
    
    # 自己回帰生成のシミュレーション
    def simulate_autoregressive(seq_length, num_codebooks):
        """自己回帰生成の時間をシミュレート"""
        total_ops = 0
        for step in range(seq_length):
            # 各ステップで全codebookを処理
            total_ops += num_codebooks * step  # 累積的な依存関係
        return total_ops
    
    # 固定長生成のシミュレーション
    def simulate_fixed_length(seq_length, num_codebooks):
        """固定長生成の時間をシミュレート"""
        # 並列処理可能
        total_ops = seq_length * num_codebooks
        return total_ops
    
    # 比較実行
    seq_lengths = [50, 100, 200, 500]
    num_codebooks = 9
    
    print(f"Comparing with {num_codebooks} codebooks:")
    print(f"{'Length':<8} {'Autoregressive':<15} {'Fixed Length':<15} {'Speedup':<10}")
    print("-" * 50)
    
    comparison_data = []
    
    for length in seq_lengths:
        auto_ops = simulate_autoregressive(length, num_codebooks)
        fixed_ops = simulate_fixed_length(length, num_codebooks)
        speedup = auto_ops / fixed_ops if fixed_ops > 0 else 0
        
        print(f"{length:<8} {auto_ops:<15} {fixed_ops:<15} {speedup:<10.2f}x")
        
        comparison_data.append({
            'length': length,
            'autoregressive': auto_ops,
            'fixed_length': fixed_ops,
            'speedup': speedup
        })
    
    return comparison_data

def create_simple_visualization(benchmark_results, comparison_data):
    """シンプルな可視化"""
    
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Skipping visualization (matplotlib not available)")
        return
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ベンチマーク結果
        if benchmark_results and benchmark_results['times']:
            texts = [f"Text {i+1}" for i in range(len(benchmark_results['times']))]
            ax1.bar(texts, benchmark_results['times'])
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Fixed Length Generation Times')
            ax1.tick_params(axis='x', rotation=45)
        
        # 理論的比較
        if comparison_data:
            lengths = [d['length'] for d in comparison_data]
            speedups = [d['speedup'] for d in comparison_data]
            
            ax2.plot(lengths, speedups, 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Theoretical Speedup (x)')
            ax2.set_title('Fixed Length vs Autoregressive (Theoretical)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('phase1_minimal_results.png', dpi=150, bbox_inches='tight')
        print("📈 Visualization saved to phase1_minimal_results.png")
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

def main():
    print("Phase 1: Minimal Fixed Length TTS Decoder Experiment")
    print("=" * 60)
    
    # システム情報
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {torch.device('cpu')}")
    
    try:
        # 1. TTSシステム初期化
        tts_system = MinimalTTSSystem()
        
        # 2. テストテキスト
        test_texts = [
            "Hello",
            "Hello world",
            "This is a test of the fixed length TTS system"
        ]
        
        # 3. ベンチマーク実行
        print("\n" + "="*50)
        benchmark_results = tts_system.benchmark(test_texts, num_runs=5)
        
        # 4. 理論的比較
        comparison_data = compare_autoregressive_vs_fixed()
        
        # 5. 結果保存
        results = {
            'benchmark': {
                'texts': benchmark_results['texts'],
                'times': benchmark_results['times'],
                'total_avg_time': benchmark_results['total_avg_time']
            },
            'comparison': comparison_data,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__
        }
        
        with open('phase1_minimal_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to phase1_minimal_results.json")
        
        # 6. 可視化
        create_simple_visualization(benchmark_results, comparison_data)
        
        # 7. 結果サマリー
        print(f"\n📋 Summary:")
        print(f"  Average generation time: {benchmark_results['total_avg_time']:.4f}s")
        print(f"  Theoretical speedup (500 length): {comparison_data[-1]['speedup']:.1f}x")
        print(f"  Successfully demonstrated fixed-length concept!")
        
        print(f"\n✅ Phase 1 (Minimal) completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Phase 1 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
