#!/usr/bin/env python3
"""
Phase 2: TTSデコーダーをANE（Apple Neural Engine）で動作させる
固定長デコーダーをCore MLに変換してANEで実行
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')
sys.path.append('/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length')

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from fixed_length_decoder import FixedLengthTTSDecoder, SimplifiedTTSGenerator
import time
import json

class ANEOptimizedTTSDecoder(nn.Module):
    """ANE最適化されたTTSデコーダー"""
    
    def __init__(self, hidden_size=1024, num_codebooks=9, vocab_size=1088, max_length=500):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # ANEに最適化されたアーキテクチャ
        # 1. シンプルな全結合層を使用（ANEが得意）
        self.encoder_projection = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()  # ANEでサポートされている活性化関数
        
        # 2. 各codebookごとに分離されたheads
        self.codebook_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, max_length * vocab_size)
            ) for _ in range(num_codebooks)
        ])
        
        # 3. 長さ予測器（オプション）
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
            codebook_outputs: [batch_size, num_codebooks, max_length]
        """
        # Global pooling (ANEに最適化)
        pooled = encoder_hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Encoder projection
        projected = self.activation(self.encoder_projection(pooled))
        
        # 長さ予測
        predicted_length = self.length_predictor(projected)
        
        # 各codebookの出力生成
        codebook_outputs = []
        for head in self.codebook_heads:
            # [batch_size, max_length * vocab_size]
            head_output = head(projected)
            
            # [batch_size, max_length, vocab_size]
            reshaped = head_output.view(-1, self.max_length, self.vocab_size)
            
            # Argmax to get token IDs
            tokens = torch.argmax(reshaped, dim=-1)
            codebook_outputs.append(tokens)
        
        # [batch_size, num_codebooks, max_length]
        result = torch.stack(codebook_outputs, dim=1)
        
        return result, predicted_length

class TTSDecoderCoreMLConverter:
    """TTS Decoder Core ML変換器"""
    
    def __init__(self):
        self.batch_size = 1
        self.seq_length = 50  # エンコーダー出力の最大長
        self.hidden_size = 1024
        self.max_length = 500
        self.num_codebooks = 9
        self.vocab_size = 1088
    
    def create_ane_optimized_model(self):
        """ANE最適化モデル作成"""
        model = ANEOptimizedTTSDecoder(
            hidden_size=self.hidden_size,
            num_codebooks=self.num_codebooks,
            vocab_size=self.vocab_size,
            max_length=self.max_length
        )
        model.eval()
        return model
    
    def convert_to_coreml(self, pytorch_model):
        """PyTorchモデルをCore MLに変換"""
        
        # サンプル入力作成
        sample_input = torch.randn(
            self.batch_size, 
            self.seq_length, 
            self.hidden_size
        )
        
        print("Converting to Core ML...")
        print(f"Input shape: {sample_input.shape}")
        
        # トレーシング実行
        with torch.no_grad():
            traced_model = torch.jit.trace(pytorch_model, sample_input)
            
            # 出力確認
            sample_output = traced_model(sample_input)
            print(f"Output shapes: {[out.shape for out in sample_output]}")
        
        # Core ML変換
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="encoder_hidden_states",
                    shape=(self.batch_size, self.seq_length, self.hidden_size),
                    dtype=np.float32
                )
            ],
            outputs=[
                ct.TensorType(name="audio_tokens", dtype=np.int32),
                ct.TensorType(name="predicted_length", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.ALL,  # ANE使用
            minimum_deployment_target=ct.target.macOS13,  # macOS 13以上
            convert_to="neuralnetwork"  # Neural Network形式（ANE最適化）
        )
        
        # モデル情報設定
        coreml_model.author = "Parler TTS ANE Experiment"
        coreml_model.short_description = "ANE-optimized TTS Decoder"
        coreml_model.version = "1.0"
        
        return coreml_model
    
    def optimize_for_ane(self, coreml_model):
        """ANE向け最適化"""
        
        # 量子化設定（ANEでは16bit floatが効率的）
        try:
            # 8bit量子化を試行
            quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, 
                nbits=8
            )
            print("8-bit quantization applied")
            return quantized_model
        except Exception as e:
            print(f"Quantization failed: {e}, using original model")
            return coreml_model
    
    def benchmark_coreml_model(self, coreml_model, num_runs=10):
        """Core MLモデルのベンチマーク"""
        
        print(f"Benchmarking Core ML model ({num_runs} runs)...")
        
        # サンプル入力準備
        sample_input = np.random.randn(
            self.batch_size, 
            self.seq_length, 
            self.hidden_size
        ).astype(np.float32)
        
        import coremltools.models as ct_models
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            # Core ML推論実行
            input_dict = {"encoder_hidden_states": sample_input}
            result = coreml_model.predict(input_dict)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if i == 0:
                print(f"Output keys: {result.keys()}")
                for key, value in result.items():
                    if hasattr(value, 'shape'):
                        print(f"{key} shape: {value.shape}")
        
        avg_time = sum(times) / len(times)
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Min time: {min(times)*1000:.2f} ms") 
        print(f"Max time: {max(times)*1000:.2f} ms")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'times_ms': [t * 1000 for t in times],
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }

class HybridTTSPipeline:
    """PyTorch + Core ML ハイブリッドパイプライン"""
    
    def __init__(self, coreml_decoder_path):
        # PyTorchエンコーダー（Phase 3で置き換え予定）
        self.pytorch_generator = SimplifiedTTSGenerator()
        
        # Core MLデコーダー
        import coremltools as ct
        self.coreml_decoder = ct.models.MLModel(coreml_decoder_path)
        
        print("Hybrid pipeline initialized")
    
    def generate_with_ane_decoder(self, text, description="A female speaker delivers her words expressively."):
        """ANEデコーダーを使用した生成"""
        
        # Phase 1: PyTorchでテキストエンコーディング
        text_inputs = self.pytorch_generator.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        
        with torch.no_grad():
            encoder_outputs = self.pytorch_generator.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
        
        # Phase 2: Core ML (ANE) でデコーディング
        encoder_hidden_states = encoder_outputs.last_hidden_state.numpy()
        
        start_time = time.time()
        coreml_result = self.coreml_decoder.predict({
            "encoder_hidden_states": encoder_hidden_states
        })
        ane_time = time.time() - start_time
        
        return {
            'audio_tokens': coreml_result['audio_tokens'],
            'predicted_length': coreml_result['predicted_length'],
            'ane_inference_time': ane_time
        }

def run_phase2_experiment():
    """Phase 2実験の実行"""
    
    print("Phase 2: ANE TTS Decoder Experiment")
    print("=" * 50)
    
    # 1. ANE最適化モデル作成
    converter = TTSDecoderCoreMLConverter()
    pytorch_model = converter.create_ane_optimized_model()
    
    # 2. Core ML変換
    print("\n1. Converting to Core ML...")
    coreml_model = converter.convert_to_coreml(pytorch_model)
    
    # 3. ANE最適化
    print("\n2. Optimizing for ANE...")
    optimized_model = converter.optimize_for_ane(coreml_model)
    
    # 4. モデル保存
    model_path = "ane_tts_decoder.mlmodel"
    optimized_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # 5. ベンチマーク実行
    print("\n3. Benchmarking Core ML model...")
    benchmark_results = converter.benchmark_coreml_model(optimized_model, num_runs=10)
    
    # 6. ハイブリッドパイプラインテスト
    print("\n4. Testing hybrid pipeline...")
    try:
        pipeline = HybridTTSPipeline(model_path)
        
        test_text = "Hello world, this is a test."
        result = pipeline.generate_with_ane_decoder(test_text)
        
        hybrid_results = {
            'audio_tokens_shape': result['audio_tokens'].shape,
            'predicted_length': float(result['predicted_length'][0]),
            'ane_inference_time_ms': result['ane_inference_time'] * 1000
        }
        
        print(f"Hybrid generation successful!")
        print(f"Audio tokens shape: {hybrid_results['audio_tokens_shape']}")
        print(f"ANE inference time: {hybrid_results['ane_inference_time_ms']:.2f} ms")
        
    except Exception as e:
        print(f"Hybrid pipeline test failed: {e}")
        hybrid_results = {'error': str(e)}
    
    # 7. 結果保存
    results = {
        'phase': 2,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'coreml_benchmark': benchmark_results,
        'hybrid_pipeline': hybrid_results,
        'model_path': model_path
    }
    
    with open('phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2 completed! Results saved to phase2_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = run_phase2_experiment()
        
        print("\n📊 Summary:")
        if 'coreml_benchmark' in results:
            print(f"ANE Decoder Average Time: {results['coreml_benchmark']['avg_time_ms']:.2f} ms")
        
        if 'error' not in results.get('hybrid_pipeline', {}):
            print("✅ Hybrid pipeline working successfully")
        else:
            print("❌ Hybrid pipeline needs debugging")
        
        print("\nNext steps:")
        print("1. Compare ANE decoder speed with PyTorch version")
        print("2. Verify output quality")
        print("3. Proceed to Phase 3 for full ANE conversion")
        
    except Exception as e:
        print(f"❌ Phase 2 experiment failed: {e}")
        import traceback
        traceback.print_exc()
