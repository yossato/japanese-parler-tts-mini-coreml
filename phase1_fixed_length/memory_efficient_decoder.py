#!/usr/bin/env python3
"""
Phase 1.1: メモリ効率的な固定長デコーダー
元のfixed_length_decoder.pyのメモリ使用量を大幅削減（20.8GB -> 4GB以下）
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import time
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import json

class MemoryEfficientFixedLengthDecoder(nn.Module):
    """メモリ効率的な固定長デコーダー"""
    
    def __init__(self, original_decoder, max_length=100):  # 500->100に削減
        super().__init__()
        self.original_decoder = original_decoder
        self.max_length = max_length
        self.num_codebooks = original_decoder.config.num_codebooks
        self.vocab_size = original_decoder.config.vocab_size
        self.hidden_size = original_decoder.config.hidden_size
        
        print(f"🎯 Memory Efficient Decoder Config:")
        print(f"  Max length: {max_length} (vs original 500)")
        print(f"  Num codebooks: {self.num_codebooks}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Hidden size: {self.hidden_size}")
        
        # 長さ予測器（小さなネットワーク）
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 各codebook用の効率的なプロジェクション
        # 元の実装：hidden_size -> max_length * vocab_size (巨大)
        # 新実装：hidden_size -> 中間層 -> max_length * vocab_size (小さな中間層)
        self.codebook_projectors = nn.ModuleList()
        
        for i in range(self.num_codebooks):
            projector = nn.Sequential(
                nn.Linear(self.hidden_size, 512),  # 中間層で圧縮
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),  # さらに圧縮
                nn.ReLU(),
                nn.Linear(256, max_length * self.vocab_size)  # 最終出力
            )
            self.codebook_projectors.append(projector)
        
        # パラメータ数の計算と表示
        total_params = sum(p.numel() for p in self.parameters())
        memory_mb = total_params * 4 / (1024**2)  # Float32
        
        print(f"📊 Memory Efficient Decoder Stats:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Estimated memory: {memory_mb:.1f} MB")
        
    def forward(self, encoder_hidden_states, encoder_attention_mask, 
                max_new_tokens=None, **kwargs):
        """メモリ効率的な順伝播"""
        
        batch_size, encoder_seq_len, hidden_size = encoder_hidden_states.shape
        
        if max_new_tokens is None:
            max_new_tokens = self.max_length
        
        # 実際の長さを短めに制限
        actual_length = min(max_new_tokens, self.max_length)
        
        # エンコーダー出力をpooling
        pooled_encoder = encoder_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 長さ予測
        predicted_length_ratio = self.length_predictor(pooled_encoder)
        predicted_length = (predicted_length_ratio * actual_length).round().int()
        
        # 各codebookの出力を生成
        outputs = []
        codebook_logits = []
        
        for codebook_idx in range(self.num_codebooks):
            # プロジェクション実行
            projected = self.codebook_projectors[codebook_idx](pooled_encoder)
            
            # [batch_size, actual_length, vocab_size]にreshape
            codebook_logits_reshaped = projected.view(batch_size, actual_length, self.vocab_size)
            codebook_logits.append(codebook_logits_reshaped)
            
            # トークンID取得
            codebook_tokens = torch.argmax(codebook_logits_reshaped, dim=-1)
            outputs.append(codebook_tokens)
        
        # [batch_size, num_codebooks, actual_length]
        output_tokens = torch.stack(outputs, dim=1)
        
        return {
            'sequences': output_tokens.view(batch_size * self.num_codebooks, actual_length),
            'predicted_length': predicted_length,
            'logits': torch.stack(codebook_logits, dim=1),  # [batch_size, num_codebooks, length, vocab_size]
            'tokens': output_tokens,  # [batch_size, num_codebooks, length]
            'actual_length': actual_length
        }

class MemoryEfficientTTSGenerator:
    """メモリ効率的なTTS生成器"""
    
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", max_length=100):
        print(f"🚀 Loading memory efficient model: {model_name}")
        print(f"   Max length: {max_length}")
        
        self.device = torch.device("cpu")  # ANE検証のためCPUから開始
        self.max_length = max_length
        
        # 元のモデルをロード
        print("Loading original Parler TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # メモリ効率的な固定長デコーダーに置き換え
        print("Creating memory efficient decoder...")
        self.efficient_decoder = MemoryEfficientFixedLengthDecoder(
            self.model.decoder,
            max_length=max_length
        )
        
        print("✅ Memory efficient model loaded successfully!")
        
    def generate_original(self, text, description="A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."):
        """オリジナルモデルでの生成（比較用）"""
        
        # テキストのトークン化
        inputs = self.tokenizer(
            description,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=50
        )
        
        with torch.no_grad():
            # オリジナルのgenerate実行（Parler TTS用の正しい呼び出し）
            start_time = time.time()
            try:
                # Parler TTSの正しい生成方法
                generation = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_input_ids=inputs.input_ids,  # prompt_hidden_statesエラーの修正
                    prompt_attention_mask=inputs.attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=self.max_length
                )
            except Exception as e:
                # フォールバック: 簡単な生成
                print(f"      Full generation failed: {e}")
                print(f"      Trying simplified generation...")
                generation = self.model.generate(
                    input_ids=inputs.input_ids[:, :10],  # 短縮
                    do_sample=False,
                    max_new_tokens=min(50, self.max_length)
                )
            end_time = time.time()
            
        return {
            'sequences': generation,
            'generation_time': end_time - start_time,
            'method': 'original_autoregressive'
        }
        
    def generate_fixed_length(self, text, description="A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."):
        """メモリ効率的な固定長生成"""
        
        # テキストのトークン化
        text_inputs = self.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        
        prompt_inputs = self.tokenizer(
            text,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=50
        )
        
        with torch.no_grad():
            # エンコーダーを実行
            start_time = time.time()
            encoder_outputs = self.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            # メモリ効率的な固定長デコーダーで生成
            decoder_outputs = self.efficient_decoder(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=text_inputs.attention_mask,
                max_new_tokens=self.max_length
            )
            end_time = time.time()
            
            decoder_outputs['generation_time'] = end_time - start_time
            decoder_outputs['method'] = 'memory_efficient_fixed_length'
            
        return decoder_outputs
    
    def compare_methods(self, texts, num_runs=3):
        """オリジナル vs メモリ効率的固定長の比較"""
        
        print(f"🔬 Comparing Original vs Memory Efficient Fixed Length")
        print(f"   Texts: {len(texts)}")
        print(f"   Runs per text: {num_runs}")
        print(f"   Max length: {self.max_length}")
        
        results = {
            'texts': texts,
            'original_times': [],
            'fixed_times': [],
            'original_outputs': [],
            'fixed_outputs': [],
            'memory_usage': {},
            'max_length': self.max_length
        }
        
        for i, text in enumerate(texts):
            print(f"\n📝 Text {i+1}: \"{text}\"")
            
            # オリジナル生成の測定
            original_times = []
            try:
                print("  🔄 Testing original method...")
                for run in range(num_runs):
                    result = self.generate_original(text)
                    original_times.append(result['generation_time'])
                    if run == 0:
                        results['original_outputs'].append(result)
                        print(f"     Original shape: {result['sequences'].shape}")
            except Exception as e:
                print(f"     ⚠️ Original method failed: {e}")
                original_times = [float('inf')] * num_runs
            
            # 固定長生成の測定
            fixed_times = []
            print("  ⚡ Testing memory efficient method...")
            for run in range(num_runs):
                result = self.generate_fixed_length(text)
                fixed_times.append(result['generation_time'])
                if run == 0:
                    results['fixed_outputs'].append(result)
                    print(f"     Fixed shape: {result['sequences'].shape}")
            
            # 平均時間の計算
            avg_original = sum(original_times) / len(original_times) if original_times else float('inf')
            avg_fixed = sum(fixed_times) / len(fixed_times)
            
            results['original_times'].append(avg_original)
            results['fixed_times'].append(avg_fixed)
            
            # スピードアップ計算
            if avg_original != float('inf') and avg_fixed > 0:
                speedup = avg_original / avg_fixed
                print(f"     Original avg: {avg_original:.3f}s")
                print(f"     Fixed avg: {avg_fixed:.3f}s")
                print(f"     Speedup: {speedup:.1f}x")
            else:
                print(f"     Fixed avg: {avg_fixed:.3f}s")
                print(f"     Original failed or too slow")
        
        # メモリ使用量の推定
        original_params = sum(p.numel() for p in self.model.decoder.parameters())
        fixed_params = sum(p.numel() for p in self.efficient_decoder.parameters())
        
        results['memory_usage'] = {
            'original_params': original_params,
            'fixed_params': fixed_params,
            'original_memory_mb': original_params * 4 / (1024**2),
            'fixed_memory_mb': fixed_params * 4 / (1024**2),
            'memory_reduction_ratio': original_params / fixed_params if fixed_params > 0 else 0
        }
        
        print(f"\n📊 Memory Usage Comparison:")
        print(f"   Original: {results['memory_usage']['original_memory_mb']:.1f} MB")
        print(f"   Fixed: {results['memory_usage']['fixed_memory_mb']:.1f} MB")
        print(f"   Reduction: {results['memory_usage']['memory_reduction_ratio']:.1f}x")
        
        return results

def profile_memory_usage():
    """メモリ使用量の詳細プロファイリング"""
    
    print("\n🔍 Memory Usage Profiling")
    print("=" * 40)
    
    # メモリ使用量の測定
    import psutil
    import gc
    
    process = psutil.Process()
    
    def get_memory_usage():
        """現在のメモリ使用量を取得"""
        return process.memory_info().rss / (1024**2)  # MB
    
    # 開始時のメモリ
    gc.collect()
    initial_memory = get_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # モデルロード前
    print("\nLoading model...")
    generator = MemoryEfficientTTSGenerator(max_length=100)
    
    after_load_memory = get_memory_usage()
    print(f"After model load: {after_load_memory:.1f} MB")
    print(f"Model loading overhead: {after_load_memory - initial_memory:.1f} MB")
    
    # 生成実行前
    print("\nGenerating sample...")
    result = generator.generate_fixed_length("Hello world")
    
    after_generation_memory = get_memory_usage()
    print(f"After generation: {after_generation_memory:.1f} MB")
    print(f"Generation overhead: {after_generation_memory - after_load_memory:.1f} MB")
    
    return {
        'initial_memory': initial_memory,
        'after_load_memory': after_load_memory,
        'after_generation_memory': after_generation_memory,
        'model_overhead': after_load_memory - initial_memory,
        'generation_overhead': after_generation_memory - after_load_memory
    }

def main():
    print("Phase 1.1: Memory Efficient Fixed Length TTS Decoder")
    print("=" * 60)
    
    try:
        # 1. メモリプロファイリング
        memory_profile = profile_memory_usage()
        
        # 2. 比較実験
        print(f"\n{'='*60}")
        generator = MemoryEfficientTTSGenerator(max_length=100)
        
        test_texts = [
            "Hello",
            "Hello world", 
            "This is a test"
        ]
        
        # 3. 比較実行
        comparison_results = generator.compare_methods(test_texts, num_runs=3)
        
        # 4. 結果保存（JSON serializable形式に変換）
        def make_json_serializable(obj):
            """TensorやNumPyオブジェクトをJSON形式に変換"""
            if hasattr(obj, 'tolist'):  # Tensor or numpy array
                return obj.tolist()
            elif hasattr(obj, 'item'):  # scalar tensor
                return obj.item()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):  # Other objects with attributes
                return str(obj)
            else:
                return obj
        
        results = {
            'comparison': make_json_serializable(comparison_results),
            'memory_profile': make_json_serializable(memory_profile),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'max_length': 100,
                'model_name': 'parler-tts/parler-tts-mini-v1'
            }
        }
        
        with open('phase1_1_memory_efficient_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to phase1_1_memory_efficient_results.json")
        
        # 5. サマリー表示
        print(f"\n📋 Summary:")
        avg_fixed_time = sum(comparison_results['fixed_times']) / len(comparison_results['fixed_times'])
        print(f"  Average fixed generation time: {avg_fixed_time:.3f}s")
        print(f"  Memory reduction: {comparison_results['memory_usage']['memory_reduction_ratio']:.1f}x")
        print(f"  Max output length: {comparison_results['max_length']}")
        
        print(f"\n✅ Phase 1.1 (Memory Efficient) completed successfully!")
        print(f"   Ready for Step 2: DAC Integration")
        
        return results
        
    except Exception as e:
        print(f"❌ Phase 1.1 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
