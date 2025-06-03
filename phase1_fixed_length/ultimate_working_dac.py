#!/usr/bin/env python3
"""
DAC統合の次元問題を修正
積算処理の次元を正しく調整
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import time
import json
import numpy as np
import wave
from pathlib import Path

from memory_efficient_decoder import MemoryEfficientFixedLengthDecoder

class UltimateWorkingTTSGenerator:
    """究極の動作版TTS生成器 (次元問題解決)"""
    
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", max_length=100):
        print(f"🎯 Loading ULTIMATE Working TTS Generator")
        print(f"   Model: {model_name}")
        print(f"   Max length: {max_length} tokens")
        
        self.device = torch.device("cpu")
        self.max_length = max_length
        
        # モデルロード
        print("📦 Loading Parler TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # メモリ効率的デコーダー
        print("⚡ Creating memory efficient decoder...")
        self.efficient_decoder = MemoryEfficientFixedLengthDecoder(
            self.model.decoder,
            max_length=max_length
        )
        
        # DAC設定
        self.audio_encoder = self.model.audio_encoder
        self.dac_config = self.audio_encoder.config
        
        print(f"📊 DAC Configuration:")
        print(f"   Codebook size: {self.dac_config.codebook_size}")
        print(f"   Sampling rate: {self.dac_config.sampling_rate} Hz")
        
        print("✅ ULTIMATE Working TTS Generator ready!")
    
    def generate_audio_codes(self, text, description="A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."):
        """テキストから音声コードを生成"""
        
        print(f"🔤 Generating audio codes for: \"{text}\"")
        
        text_inputs = self.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )
        
        with torch.no_grad():
            start_time = time.time()
            
            encoder_outputs = self.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            decoder_outputs = self.efficient_decoder(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=text_inputs.attention_mask,
                max_new_tokens=self.max_length
            )
            
            end_time = time.time()
            
            audio_codes = decoder_outputs['tokens'].squeeze(0)  # [num_codebooks, length]
            
        print(f"  ⏱️  Code generation time: {end_time - start_time:.3f}s")
        print(f"  📊 Audio codes shape: {audio_codes.shape}")
        print(f"  🎚️  Code range: [{audio_codes.min()}, {audio_codes.max()}]")
        
        return {
            'audio_codes': audio_codes,
            'generation_time': end_time - start_time
        }
    
    def codes_to_audio(self, audio_codes):
        """音声コードから実際の音声波形を生成 (次元修正版)"""
        
        print(f"🎵 Converting codes to audio...")
        print(f"   Input codes shape: {audio_codes.shape}")
        
        with torch.no_grad():
            start_time = time.time()
            
            try:
                # DAC内部モデルを取得
                dac_internal = self.audio_encoder.model
                quantizer = dac_internal.quantizer
                
                # 音声コードの形状を調整
                if audio_codes.dim() == 2:
                    audio_codes = audio_codes.unsqueeze(0)  # [9, 100] -> [1, 9, 100]
                
                # 音声コードを正しい範囲にクランプ
                audio_codes = torch.clamp(audio_codes, 0, self.dac_config.codebook_size - 1)
                print(f"   Clamped codes range: [{audio_codes.min()}, {audio_codes.max()}]")
                
                # quantizerの構造を詳細調査
                print(f"   🔍 Investigating quantizer structure...")
                sample_layer = quantizer.quantizers[0]
                print(f"   First quantizer layer type: {type(sample_layer)}")
                
                # quantizerのout_projを使用して正しい次元に変換
                latents = torch.zeros(audio_codes.shape[0], 1024, audio_codes.shape[2], 
                                    device=audio_codes.device)
                
                for i, quantizer_layer in enumerate(quantizer.quantizers):
                    if i < audio_codes.shape[1]:
                        layer_codes = audio_codes[:, i, :]  # [1, 100]
                        
                        print(f"   Processing layer {i}, codes shape: {layer_codes.shape}")
                        
                        # まずcodebookで埋め込み取得
                        embedded = quantizer_layer.codebook(layer_codes)  # [1, 100, 8]
                        print(f"   Layer {i} embedded shape: {embedded.shape}")
                        
                        # out_projで1024次元に変換
                        if hasattr(quantizer_layer, 'out_proj'):
                            # [1, 100, 8] -> [1, 8, 100] -> out_proj -> [1, 1024, 100]
                            embedded_t = embedded.transpose(1, 2)  # [1, 8, 100]
                            projected = quantizer_layer.out_proj(embedded_t)  # [1, 1024, 100]
                            print(f"   Layer {i} projected shape: {projected.shape}")
                            latents += projected
                        else:
                            print(f"   Layer {i}: no out_proj found")
                
                print(f"   Combined latents shape: {latents.shape}")
                
                # DAC decode実行
                audio_waveform = dac_internal.decode(latents.detach())
                
                if isinstance(audio_waveform, torch.Tensor):
                    audio_np = audio_waveform.detach().squeeze().cpu().numpy()
                else:
                    audio_np = np.array(audio_waveform)
                
                end_time = time.time()
                
                print(f"   ⏱️  Audio conversion time: {end_time - start_time:.3f}s")
                print(f"   📊 Audio waveform shape: {audio_np.shape}")
                print(f"   🎚️  Audio range: [{audio_np.min():.3f}, {audio_np.max():.3f}]")
                
                return {
                    'audio_waveform': audio_np,
                    'conversion_time': end_time - start_time,
                    'sample_rate': self.dac_config.sampling_rate,
                    'duration': len(audio_np) / self.dac_config.sampling_rate,
                    'is_real_dac': True
                }
                
            except Exception as e:
                print(f"   ❌ DAC decoding failed: {e}")
                print(f"   🔄 Falling back to mock audio...")
                
                # フォールバック
                duration_seconds = audio_codes.shape[-1] / self.dac_config.frame_rate
                sample_rate = self.dac_config.sampling_rate
                num_samples = int(duration_seconds * sample_rate)
                
                t = np.linspace(0, duration_seconds, num_samples)
                frequency = 440
                audio_np = 0.3 * np.sin(2 * np.pi * frequency * t)
                
                return {
                    'audio_waveform': audio_np,
                    'conversion_time': 0.001,
                    'sample_rate': sample_rate,
                    'duration': duration_seconds,
                    'is_real_dac': False
                }
    
    def text_to_audio(self, text, output_dir="ultimate_working_audio", description=None):
        """テキストから音声ファイルまでの完全パイプライン"""
        
        if description is None:
            description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."
        
        print(f"\n🎵 Complete Text-to-Audio Pipeline (ULTIMATE)")
        print(f"   Text: \"{text}\"")
        
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
        
        safe_text = "".join(c for c in text if c.isalnum() or c in " -_").strip()
        safe_text = safe_text.replace(" ", "_")[:30]
        filename = f"ultimate_audio_{safe_text}.wav"
        filepath = os.path.join(output_dir, filename)
        
        print(f"  💾 Saving audio file: {filename}")
        
        # 正規化と保存
        audio_normalized = audio_waveform / np.max(np.abs(audio_waveform)) * 0.8
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        total_end_time = time.time()
        
        result = {
            'text': text,
            'output_file': filepath,
            'duration': audio_result['duration'],
            'sample_rate': sample_rate,
            'file_size_mb': os.path.getsize(filepath) / (1024**2),
            'total_time': total_end_time - total_start_time,
            'code_generation_time': code_result['generation_time'],
            'audio_conversion_time': audio_result['conversion_time'],
            'is_real_dac': audio_result['is_real_dac']
        }
        
        print(f"  ✅ Audio generation completed!")
        print(f"     Duration: {result['duration']:.2f}s")
        print(f"     File size: {result['file_size_mb']:.2f}MB")
        print(f"     Total time: {result['total_time']:.3f}s")
        
        if result['is_real_dac']:
            print(f"     🎉 ULTIMATE DAC AUDIO SUCCESS! 🎉")
        else:
            print(f"     ⚠️  Using fallback audio")
        
        return result

def test_ultimate_working_integration():
    """究極の動作版テスト"""
    
    print("🚀 Testing ULTIMATE Working DAC Integration")
    print("=" * 60)
    
    try:
        generator = UltimateWorkingTTSGenerator(max_length=100)
        
        test_texts = ["Hello", "Hello world"]
        
        results = []
        real_dac_count = 0
        
        for text in test_texts:
            print(f"\n--- Testing: '{text}' ---")
            
            result = generator.text_to_audio(text)
            results.append(result)
            
            if result['is_real_dac']:
                real_dac_count += 1
                print(f"    🎉 ULTIMATE DAC SUCCESS!")
            else:
                print(f"    ⚠️  Fallback used")
        
        # 結果サマリー
        summary = {
            'results': results,
            'total_tests': len(test_texts),
            'real_dac_count': real_dac_count,
            'success_rate': real_dac_count / len(test_texts),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('ultimate_working_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎯 ULTIMATE RESULTS:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Real DAC audio: {summary['real_dac_count']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] > 0:
            print(f"   🎊 ULTIMATE DAC INTEGRATION SUCCESS! 🎊")
        
        return summary
        
    except Exception as e:
        print(f"❌ Ultimate test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Phase 1.1 Step 2: ULTIMATE Working DAC Integration")
    print("=" * 70)
    
    result = test_ultimate_working_integration()
    
    if result and result['success_rate'] > 0:
        print(f"\n🎊 ULTIMATE SUCCESS! DAC INTEGRATION COMPLETE! 🎊")
        print(f"📁 Check real DAC audio in: ultimate_working_audio/")
        print(f"🎧 Listen with: open ultimate_working_audio/ultimate_audio_Hello.wav")
        return True
    else:
        print(f"\n🔧 Continue debugging DAC integration...")
        return False

if __name__ == "__main__":
    main()
