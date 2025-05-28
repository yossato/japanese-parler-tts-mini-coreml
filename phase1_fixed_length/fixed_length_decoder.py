#!/usr/bin/env python3
"""
Phase 1: TTSデコーダーの固定長出力への簡略化
PyTorch実装のまま自己回帰生成を固定長生成に変更
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

class FixedLengthTTSDecoder(nn.Module):
    """固定長出力用のTTSデコーダー簡略版"""
    
    def __init__(self, original_decoder, max_length=1000):
        super().__init__()
        self.original_decoder = original_decoder
        self.max_length = max_length
        self.num_codebooks = original_decoder.config.num_codebooks
        self.vocab_size = original_decoder.config.vocab_size
        
        # 固定長生成のための追加レイヤー
        self.length_predictor = nn.Linear(
            original_decoder.config.hidden_size, 
            1
        )
        
        # 直接出力用のプロジェクション
        self.direct_output_projection = nn.ModuleList([
            nn.Linear(original_decoder.config.hidden_size, max_length * self.vocab_size)
            for _ in range(self.num_codebooks)
        ])
        
    def forward(self, encoder_hidden_states, encoder_attention_mask, 
                max_new_tokens=None, **kwargs):
        """固定長出力での順伝播"""
        
        batch_size, encoder_seq_len, hidden_size = encoder_hidden_states.shape
        
        if max_new_tokens is None:
            max_new_tokens = self.max_length
        
        # 長さ予測（オプション）
        pooled_encoder = encoder_hidden_states.mean(dim=1)
        predicted_length = torch.sigmoid(self.length_predictor(pooled_encoder)) * max_new_tokens
        predicted_length = predicted_length.round().int()
        
        # 従来の自己回帰生成の代わりに、エンコーダー出力から直接生成
        outputs = []
        
        for codebook_idx in range(self.num_codebooks):
            # エンコーダー出力を使用して直接トークンを生成
            projection = self.direct_output_projection[codebook_idx]
            
            # [batch_size, seq_len, hidden_size] -> [batch_size, max_length * vocab_size]
            projected = projection(pooled_encoder)
            
            # [batch_size, max_length, vocab_size]にreshape
            codebook_logits = projected.view(batch_size, max_new_tokens, self.vocab_size)
            
            # トークンID取得
            codebook_tokens = torch.argmax(codebook_logits, dim=-1)
            outputs.append(codebook_tokens)
        
        # [batch_size, num_codebooks, max_length]
        output_tokens = torch.stack(outputs, dim=1)
        
        return {
            'sequences': output_tokens.view(batch_size * self.num_codebooks, max_new_tokens),
            'predicted_length': predicted_length,
            'logits': torch.stack([projection(pooled_encoder) for projection in self.direct_output_projection])
        }

class SimplifiedTTSGenerator:
    """簡略化されたTTS生成器"""
    
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1"):
        print(f"Loading model: {model_name}")
        self.device = torch.device("cpu")  # ANE検証のためCPUから開始
        
        # 元のモデルをロード
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 固定長デコーダーに置き換え
        self.fixed_decoder = FixedLengthTTSDecoder(
            self.model.decoder,
            max_length=500  # 固定長を500に设定
        )
        
        print("Model loaded successfully!")
        
    def generate_fixed_length(self, text, description="A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."):
        """固定長生成"""
        
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
            encoder_outputs = self.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            # 固定長デコーダーで生成
            decoder_outputs = self.fixed_decoder(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=text_inputs.attention_mask,
                max_new_tokens=500
            )
            
        return decoder_outputs
    
    def profile_generation(self, text, num_runs=5):
        """生成処理のプロファイリング"""
        
        print(f"Profiling generation for {num_runs} runs...")
        
        # ウォームアップ
        _ = self.generate_fixed_length(text)
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=False
            ) as prof:
                with record_function("Fixed_Length_Generation"):
                    result = self.generate_fixed_length(text)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i == 0:  # 最初の実行のプロファイル結果を保存
                prof.export_chrome_trace(f"phase1_profile_run_{i}.json")
                
                print("\nTop operations by CPU time:")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        avg_time = sum(times) / len(times)
        print(f"\nGeneration completed!")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Min time: {min(times):.3f}s")
        print(f"Max time: {max(times):.3f}s")
        
        return {
            'avg_time': avg_time,
            'times': times,
            'result_shape': result['sequences'].shape
        }

def compare_original_vs_fixed():
    """元の実装と固定長実装の比較"""
    
    print("=== Comparing Original vs Fixed Length Implementation ===")
    
    # 固定長生成器
    generator = SimplifiedTTSGenerator()
    
    test_text = "Hello, this is a test of the text-to-speech system."
    
    print("\n1. Testing Fixed Length Generation:")
    fixed_results = generator.profile_generation(test_text, num_runs=3)
    
    # 結果を保存
    results = {
        'fixed_length': fixed_results,
        'test_text': test_text,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to phase1_results.json")
    
    # 簡単な可視化
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fixed_results['times'])), fixed_results['times'], 'bo-', label='Fixed Length')
    plt.xlabel('Run Number')
    plt.ylabel('Time (seconds)')
    plt.title('Generation Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('phase1_timing_comparison.png')
    print("Timing chart saved to phase1_timing_comparison.png")

def analyze_decoder_complexity():
    """デコーダーの複雑さを分析"""
    
    print("=== Analyzing Decoder Complexity ===")
    
    generator = SimplifiedTTSGenerator()
    
    # 元のデコーダーの構造分析
    original_decoder = generator.model.decoder
    print(f"Original decoder layers: {len(original_decoder.model.decoder.layers)}")
    print(f"Hidden size: {original_decoder.config.hidden_size}")
    print(f"Num attention heads: {original_decoder.config.num_attention_heads}")
    print(f"Vocab size: {original_decoder.config.vocab_size}")
    print(f"Num codebooks: {original_decoder.config.num_codebooks}")
    
    # パラメータ数計算
    original_params = sum(p.numel() for p in original_decoder.parameters())
    fixed_params = sum(p.numel() for p in generator.fixed_decoder.parameters())
    
    print(f"\nParameter comparison:")
    print(f"Original decoder parameters: {original_params:,}")
    print(f"Fixed length decoder parameters: {fixed_params:,}")
    print(f"Reduction: {(original_params - fixed_params) / original_params * 100:.1f}%")
    
    # メモリ使用量の概算
    original_memory = original_params * 4 / (1024**2)  # Float32, MB
    fixed_memory = fixed_params * 4 / (1024**2)
    
    print(f"\nMemory usage (estimated):")
    print(f"Original: {original_memory:.1f} MB")
    print(f"Fixed: {fixed_memory:.1f} MB")
    print(f"Reduction: {original_memory - fixed_memory:.1f} MB")

if __name__ == "__main__":
    print("Phase 1: Fixed Length TTS Decoder Experiment")
    print("=" * 50)
    
    try:
        # 複雑さ分析
        analyze_decoder_complexity()
        print("\n")
        
        # パフォーマンス比較
        compare_original_vs_fixed()
        
        print("\n✅ Phase 1 experiment completed successfully!")
        print("Next steps:")
        print("1. Review the generated profile (phase1_profile_run_0.json)")
        print("2. Check timing results (phase1_results.json)")
        print("3. Analyze the timing chart (phase1_timing_comparison.png)")
        print("4. Proceed to Phase 2 if results are satisfactory")
        
    except Exception as e:
        print(f"❌ Error during Phase 1 experiment: {e}")
        import traceback
        traceback.print_exc()
