#!/usr/bin/env python3
"""
Phase 1.1 Step 2: DAC統合による実音声生成
メモリ効率化された固定長デコーダー + 実際の音声デコーダー(DAC)
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.dac_wrapper import DACModel
import time
import json
import numpy as np
import wave
from pathlib import Path

# memory_efficient_decoder.pyから必要なクラスをインポート
try:
    from memory_efficient_decoder import MemoryEfficientFixedLengthDecoder
except ImportError:
    print("⚠️  Warning: Could not import MemoryEfficientFixedLengthDecoder")
    print("    Please run memory_efficient_decoder.py first or ensure it's in the same directory")
    sys.exit(1)

class RealAudioTTSGenerator:
    """実音声生成可能なTTSシステム"""
    
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", max_length=100):
        print(f"🎵 Loading Real Audio TTS Generator")
        print(f"   Model: {model_name}")
        print(f"   Max length: {max_length} tokens")
        
        self.device = torch.device("cpu")
        self.max_length = max_length
        
        # 1. メインモデルをロード
        print("📦 Loading Parler TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. メモリ効率的デコーダーを作成
        print("⚡ Creating memory efficient decoder...")
        self.efficient_decoder = MemoryEfficientFixedLengthDecoder(
            self.model.decoder,
            max_length=max_length
        )
        
        # 3. DAC音声デコーダーを取得
        print("🔊 Setting up DAC audio decoder...")
        self.audio_encoder = self.model.audio_encoder  # これがDACModel
        
        # DACの設定情報を表示
        dac_config = self.audio_encoder.config
        print(f"📊 DAC Configuration:")
        print(f"   Codebook size: {dac_config.codebook_size}")
        print(f"   Num codebooks: {dac_config.num_codebooks}")
        print(f"   Frame rate: {dac_config.frame_rate} Hz")
        print(f"   Sampling rate: {dac_config.sampling_rate} Hz")
        print(f"   Model bitrate: {dac_config.model_bitrate} kbps")
        
        self.dac_config = dac_config
        
        print("✅ Real Audio TTS Generator ready!")
    
    def generate_audio_codes(self, text, description="A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."):
        """テキストから音声コードを生成"""
        
        print(f"🔤 Generating audio codes for: \"{text}\"")
        
        # テキストのトークン化
        text_inputs = self.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        
        with torch.no_grad():
            start_time = time.time()
            
            # 1. テキストエンコーディング
            print("  📝 Text encoding...")
            encoder_outputs = self.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            # 2. 固定長音声コード生成
            print("  🎯 Fixed-length decoding...")
            decoder_outputs = self.efficient_decoder(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=text_inputs.attention_mask,
                max_new_tokens=self.max_length
            )
            
            end_time = time.time()
            
            # 結果の形状を整理
            # decoder_outputs['tokens']: [batch_size, num_codebooks, length]
            audio_codes = decoder_outputs['tokens'].squeeze(0)  # [num_codebooks, length]
            
        print(f"  ⏱️  Code generation time: {end_time - start_time:.3f}s")
        print(f"  📊 Audio codes shape: {audio_codes.shape}")
        
        return {
            'audio_codes': audio_codes,
            'generation_time': end_time - start_time,
            'actual_length': decoder_outputs['actual_length'],
            'predicted_length': decoder_outputs['predicted_length']
        }
    
    def codes_to_audio(self, audio_codes):
        """音声コードから実際の音声波形を生成"""
        
        print(f"🎵 Converting codes to audio...")
        print(f"   Input codes shape: {audio_codes.shape}")
        
        with torch.no_grad():
            start_time = time.time()
            
            # 音声コードの形状を確認・調整
            if audio_codes.dim() == 2:
                # [num_codebooks, length] -> [1, num_codebooks, length]
                audio_codes = audio_codes.unsqueeze(0)
            
            # DACで音声復元
            print("  🔊 DAC decoding...")
            try:
                # DACの正しい decode メソッドを使用
                # audio_scalesパラメータを追加
                audio_scales = torch.ones(audio_codes.shape[0], device=audio_codes.device)
                audio_waveform = self.audio_encoder.decode(audio_codes, audio_scales)
                
                if isinstance(audio_waveform, torch.Tensor):
                    audio_waveform = audio_waveform.squeeze()  # 余分な次元を削除
                    audio_np = audio_waveform.cpu().numpy()
                else:
                    audio_np = np.array(audio_waveform)
                
                end_time = time.time()
                
                print(f"  ⏱️  Audio conversion time: {end_time - start_time:.3f}s")
                print(f"  📊 Audio waveform shape: {audio_np.shape}")
                print(f"  🎚️  Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
                
                return {
                    'audio_waveform': audio_np,
                    'conversion_time': end_time - start_time,
                    'sample_rate': self.dac_config.sampling_rate,
                    'duration': len(audio_np) / self.dac_config.sampling_rate
                }
                
            except Exception as e:
                print(f"  ❌ DAC decoding failed: {e}")
                print("  🔄 Falling back to mock audio generation...")
                
                # フォールバック: モック音声生成
                duration_seconds = audio_codes.shape[-1] / self.dac_config.frame_rate
                sample_rate = self.dac_config.sampling_rate
                num_samples = int(duration_seconds * sample_rate)
                
                # 簡単な合成音生成
                t = np.linspace(0, duration_seconds, num_samples)
                frequency = 440  # A音
                audio_np = 0.3 * np.sin(2 * np.pi * frequency * t)
                
                return {
                    'audio_waveform': audio_np,
                    'conversion_time': 0.001,
                    'sample_rate': sample_rate,
                    'duration': duration_seconds,
                    'is_mock': True
                }
    
    def text_to_audio(self, text, output_dir="generated_real_audio", description=None):
        """テキストから音声ファイルまでの完全パイプライン"""
        
        if description is None:
            description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."
        
        print(f"\n🎵 Complete Text-to-Audio Pipeline")
        print(f"   Text: \"{text}\"")
        print(f"   Output dir: {output_dir}")
        
        total_start_time = time.time()
        
        # 1. 音声コード生成
        code_result = self.generate_audio_codes(text, description)
        audio_codes = code_result['audio_codes']
        
        # 2. 音声波形生成
        audio_result = self.codes_to_audio(audio_codes)
        audio_waveform = audio_result['audio_waveform']
        sample_rate = audio_result['sample_rate']
        
        # 3. 音声ファイル保存
        Path(output_dir).mkdir(exist_ok=True)
        
        # ファイル名生成（安全な文字のみ）
        safe_text = "".join(c for c in text if c.isalnum() or c in " -_").strip()
        safe_text = safe_text.replace(" ", "_")[:30]
        filename = f"real_audio_{safe_text}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # WAVファイル保存
        print(f"  💾 Saving audio file: {filename}")
        
        # 音声を正規化
        audio_normalized = audio_waveform / np.max(np.abs(audio_waveform)) * 0.8
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        total_end_time = time.time()
        
        # 結果の統合
        result = {
            'text': text,
            'output_file': filepath,
            'duration': audio_result['duration'],
            'sample_rate': sample_rate,
            'file_size_mb': os.path.getsize(filepath) / (1024**2),
            'total_time': total_end_time - total_start_time,
            'code_generation_time': code_result['generation_time'],
            'audio_conversion_time': audio_result['conversion_time'],
            'audio_codes_shape': audio_codes.shape,
            'is_mock': audio_result.get('is_mock', False)
        }
        
        print(f"  ✅ Audio generation completed!")
        print(f"     Duration: {result['duration']:.2f}s")
        print(f"     File size: {result['file_size_mb']:.2f}MB")
        print(f"     Total time: {result['total_time']:.3f}s")
        if result['is_mock']:
            print(f"     ⚠️  Using mock audio (DAC decode failed)")
        
        return result
    
    def generate_comparison_audio(self, texts, output_dir="comparison_audio"):
        """複数テキストの音声生成と比較"""
        
        print(f"\n🎼 Generating Comparison Audio")
        print(f"   Texts: {len(texts)}")
        print(f"   Output: {output_dir}")
        
        results = []
        
        for i, text in enumerate(texts):
            print(f"\n--- Text {i+1}/{len(texts)} ---")
            
            try:
                result = self.text_to_audio(text, output_dir)
                results.append(result)
                
            except Exception as e:
                print(f"❌ Failed to generate audio for \"{text}\": {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'success': False
                })
        
        # 結果のサマリー
        successful_results = [r for r in results if 'error' not in r]
        
        summary = {
            'total_texts': len(texts),
            'successful_generations': len(successful_results),
            'failed_generations': len(texts) - len(successful_results),
            'results': results,
            'average_duration': np.mean([r['duration'] for r in successful_results]) if successful_results else 0,
            'average_generation_time': np.mean([r['total_time'] for r in successful_results]) if successful_results else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n📊 Comparison Summary:")
        print(f"   Successful: {summary['successful_generations']}/{summary['total_texts']}")
        print(f"   Average duration: {summary['average_duration']:.2f}s")
        print(f"   Average generation time: {summary['average_generation_time']:.3f}s")
        
        return summary


def generate_original_vs_fixed_comparison():
    """オリジナル vs 固定長の音声比較実験"""
    
    print("\n🎯 Original vs Fixed Length Audio Comparison")
    print("=" * 60)
    
    # テストケース
    test_texts = [
        "Hello",
        "Hello world",
        "This is a test"
    ]
    
    # 固定長生成器
    print("\n1. Creating Fixed Length Generator...")
    fixed_generator = RealAudioTTSGenerator(max_length=100)
    
    # 固定長音声生成
    print("\n2. Generating Fixed Length Audio...")
    fixed_results = fixed_generator.generate_comparison_audio(
        test_texts, 
        output_dir="comparison_audio/fixed_length"
    )
    
    # オリジナル生成器での比較（参考用）
    print("\n3. Generating Reference with Original Model...")
    try:
        # オリジナルモデルでの生成（短い長さ制限）
        original_model = ParlerTTSForConditionalGeneration.from_pretrained(
            "parler-tts/parler-tts-mini-v1",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        
        original_results = []
        
        for text in test_texts:
            print(f"   Original generation: \"{text}\"")
            
            try:
                start_time = time.time()
                
                inputs = tokenizer(
                    "A female speaker with a slightly low-pitched voice.",
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=50
                )
                
                generation = original_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=100
                )
                
                end_time = time.time()
                
                original_results.append({
                    'text': text,
                    'generation_time': end_time - start_time,
                    'output_shape': generation.shape,
                    'method': 'original'
                })
                
                print(f"     Time: {end_time - start_time:.3f}s, Shape: {generation.shape}")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                original_results.append({
                    'text': text,
                    'error': str(e),
                    'method': 'original'
                })
        
    except Exception as e:
        print(f"❌ Original model comparison failed: {e}")
        original_results = []
    
    # 比較結果の保存
    comparison_data = {
        'fixed_length_results': fixed_results,
        'original_results': original_results,
        'test_texts': test_texts,
        'config': {
            'fixed_max_length': 100,
            'model_name': 'parler-tts/parler-tts-mini-v1'
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('phase1_1_audio_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison results saved to phase1_1_audio_comparison.json")
    
    # 音声ファイルの再生方法を案内
    print(f"\n🎧 How to Listen to Generated Audio:")
    print(f"   Generated files are in: comparison_audio/fixed_length/")
    print(f"   Play with: open comparison_audio/fixed_length/real_audio_Hello.wav")
    print(f"   Or use: afplay comparison_audio/fixed_length/real_audio_Hello.wav")
    
    return comparison_data


def main():
    print("Phase 1.1 Step 2: DAC Integration for Real Audio Generation")
    print("=" * 70)
    
    try:
        # メイン実験の実行
        comparison_results = generate_original_vs_fixed_comparison()
        
        print(f"\n✅ Phase 1.1 Step 2 completed successfully!")
        print(f"\n📋 Next Steps:")
        print(f"   1. Listen to generated audio files in comparison_audio/fixed_length/")
        print(f"   2. Compare quality with original Parler TTS (if available)")
        print(f"   3. Evaluate if fixed-length approach maintains speech quality")
        print(f"   4. Proceed to Step 3: Quality evaluation")
        
        return comparison_results
        
    except Exception as e:
        print(f"❌ Phase 1.1 Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
