#!/usr/bin/env python3
"""
Phase 1: TTSデコーダーの固定長出力への簡略化（修正版）
Core ML変換なしで、PyTorch実装のみでテスト
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

import torch
import torch.nn as nn
import time
import json
import matplotlib.pyplot as plt

# 安全なインポート
try:
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    PARLER_TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Parler TTS not available: {e}")
    PARLER_TTS_AVAILABLE = False

# プロファイリング用の安全なインポート
try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    print("⚠️  Torch profiler not available, using basic timing")
    PROFILER_AVAILABLE = False

class SimpleFixedLengthTTSDecoder(nn.Module):
    """シンプルな固定長TTSデコーダー（最小限実装）"""
    
    def __init__(self, hidden_size=1024, num_codebooks=9, vocab_size=1088, max_length=500):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # シンプルな固定長生成のための線形層
        self.encoder_proj = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        
        # 各codebook用の出力ヘッド
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, max_length * vocab_size)
            ) for _ in range(num_codebooks)
        ])
        
        # 長さ予測器
        self.length_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            dict with 'sequences', 'predicted_length'
        """
        # Global average pooling
        pooled = encoder_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Encoder projection
        projected = self.activation(self.encoder_proj(pooled))
        
        # Length prediction
        predicted_length = self.length_predictor(projected) * self.max_length
        
        # Generate tokens for each codebook
        codebook_outputs = []
        for head in self.output_heads:
            # [batch_size, max_length * vocab_size]
            head_output = head(projected)
            
            # Reshape and get token IDs
            logits = head_output.view(-1, self.max_length, self.vocab_size)
            tokens = torch.argmax(logits, dim=-1)
            codebook_outputs.append(tokens)
        
        # Stack codebook outputs
        output_tokens = torch.stack(codebook_outputs, dim=1)  # [batch, codebooks, length]
        
        return {
            'sequences': output_tokens.view(-1, self.max_length),  # [batch*codebooks, length]
            'predicted_length': predicted_length,
            'codebook_outputs': output_tokens
        }

class MockParlerTTSGenerator:
    """Parler TTSが利用できない場合のモック"""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.hidden_size = 1024
        self.max_text_length = 50
        
        # シンプルなテキストエンコーダー（モック）
        self.text_encoder = nn.Sequential(
            nn.Embedding(32128, self.hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=8,
                    dim_feedforward=2048,
                    batch_first=True
                ),
                num_layers=2
            )
        )
        
        # 固定長デコーダー
        self.fixed_decoder = SimpleFixedLengthTTSDecoder(
            hidden_size=self.hidden_size,
            max_length=500
        )
        
        print("✅ Mock TTS Generator initialized")
    
    def tokenize_text(self, text):
        """テキストを簡単にトークン化"""
        # 文字レベルの簡単なトークン化
        tokens = [ord(c) % 32128 for c in text[:self.max_text_length]]
        # パディング
        while len(tokens) < self.max_text_length:
            tokens.append(0)
        
        return torch.tensor(tokens).unsqueeze(0)  # [1, max_text_length]
    
    def generate_fixed_length(self, text):
        """固定長生成（モック版）"""
        
        # テキストトークン化
        input_ids = self.tokenize_text(text)
        
        with torch.no_grad():
            # テキストエンコーディング
            encoder_outputs = self.text_encoder(input_ids)
            
            # 固定長デコーディング
            decoder_outputs = self.fixed_decoder(encoder_outputs)
        
        return decoder_outputs

def run_performance_comparison():
    """パフォーマンス比較実行"""
    
    print("=== Performance Comparison Test ===")
    
    # テストテキスト
    test_texts = [
        "Hello world",
        "This is a test of the text to speech system",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    results = {}
    
    if PARLER_TTS_AVAILABLE:
        try:
            print("Testing with real Parler TTS...")
            # 実際のParler TTS
            real_generator = SimplifiedTTSGenerator()
            results['real_parler'] = test_generator(real_generator, test_texts, "Real Parler TTS")
        except Exception as e:
            print(f"Real Parler TTS failed: {e}")
            results['real_parler'] = None
    
    # モック版のテスト
    print("Testing with Mock TTS Generator...")
    mock_generator = MockParlerTTSGenerator()
    results['mock_tts'] = test_mock_generator(mock_generator, test_texts, "Mock TTS")
    
    return results

def test_mock_generator(generator, test_texts, name):
    """モックジェネレーターのテスト"""
    
    print(f"\n🧪 Testing {name}")
    
    times = []
    outputs = []
    
    for i, text in enumerate(test_texts):
        print(f"  Text {i+1}: \"{text}\"")
        
        # 複数回実行してタイミング測定
        run_times = []
        for run in range(3):
            start_time = time.time()
            
            if PROFILER_AVAILABLE and run == 0:
                with profile(
                    activities=[ProfilerActivity.CPU],
                    record_shapes=True,
                    profile_memory=True
                ) as prof:
                    result = generator.generate_fixed_length(text)
                
                # プロファイル結果を保存
                prof.export_chrome_trace(f"phase1_mock_profile_{i}.json")
                print(f"    Profile saved to phase1_mock_profile_{i}.json")
                
            else:
                result = generator.generate_fixed_length(text)
            
            end_time = time.time()
            run_times.append(end_time - start_time)
            
            if run == 0:
                outputs.append(result)
                print(f"    Output shape: {result['sequences'].shape}")
                print(f"    Predicted length: {result['predicted_length'].item():.1f}")
        
        avg_time = sum(run_times) / len(run_times)
        times.append(avg_time)
        print(f"    Average time: {avg_time:.3f}s")
    
    return {
        'name': name,
        'times': times,
        'avg_time': sum(times) / len(times),
        'outputs': outputs,
        'test_texts': test_texts
    }

def test_generator(generator, test_texts, name):
    """実際のParler TTSジェネレーターのテスト"""
    
    print(f"\n🧪 Testing {name}")
    
    times = []
    outputs = []
    
    for i, text in enumerate(test_texts):
        print(f"  Text {i+1}: \"{text}\"")
        
        try:
            run_times = []
            for run in range(3):
                start_time = time.time()
                result = generator.generate_fixed_length(text)
                end_time = time.time()
                run_times.append(end_time - start_time)
                
                if run == 0:
                    outputs.append(result)
                    print(f"    Output shape: {result['sequences'].shape}")
            
            avg_time = sum(run_times) / len(run_times)
            times.append(avg_time)
            print(f"    Average time: {avg_time:.3f}s")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            times.append(float('inf'))
            outputs.append(None)
    
    return {
        'name': name,
        'times': times,
        'avg_time': sum([t for t in times if t != float('inf')]) / len([t for t in times if t != float('inf')]) if any(t != float('inf') for t in times) else float('inf'),
        'outputs': outputs,
        'test_texts': test_texts
    }

def create_visualization(results):
    """結果の可視化"""
    
    print("\n📊 Creating visualization...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # タイミング比較
        for key, result in results.items():
            if result and result['times']:
                valid_times = [t for t in result['times'] if t != float('inf')]
                if valid_times:
                    ax1.plot(range(len(valid_times)), valid_times, 'o-', label=result['name'])
        
        ax1.set_xlabel('Test Text Index')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Generation Time Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # 平均時間比較
        names = []
        avg_times = []
        for key, result in results.items():
            if result and result['avg_time'] != float('inf'):
                names.append(result['name'])
                avg_times.append(result['avg_time'])
        
        if names:
            ax2.bar(names, avg_times)
            ax2.set_ylabel('Average Time (seconds)')
            ax2.set_title('Average Generation Time')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('phase1_performance_comparison.png', dpi=150, bbox_inches='tight')
        print("📈 Chart saved to phase1_performance_comparison.png")
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")

def main():
    print("Phase 1: Fixed Length TTS Decoder Experiment (Safe Version)")
    print("=" * 60)
    
    # システム情報
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # パフォーマンス比較実行
        results = run_performance_comparison()
        
        # 結果保存
        # JSON serializable形式に変換
        serializable_results = {}
        for key, result in results.items():
            if result:
                serializable_results[key] = {
                    'name': result['name'],
                    'times': result['times'],
                    'avg_time': result['avg_time'],
                    'test_texts': result['test_texts']
                    # 'outputs'は除外（serializableではない）
                }
        
        with open('phase1_results_safe.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to phase1_results_safe.json")
        
        # 可視化
        create_visualization(results)
        
        # 結果サマリー
        print(f"\n📋 Results Summary:")
        for key, result in results.items():
            if result:
                print(f"  {result['name']}: {result['avg_time']:.3f}s average")
        
        print(f"\n✅ Phase 1 (Safe Version) completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Check phase1_results_safe.json for detailed results")
        print(f"2. Review phase1_performance_comparison.png for visual comparison")
        print(f"3. Fix environment issues if needed")
        print(f"4. Re-run with full Parler TTS when dependencies are resolved")
        
    except Exception as e:
        print(f"❌ Phase 1 experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
