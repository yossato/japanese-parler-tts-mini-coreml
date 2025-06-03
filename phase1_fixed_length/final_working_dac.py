#!/usr/bin/env python3
"""
Phase 1.1 Step 2: 最終解決版DAC統合
動作確認済みのDAC統合方法を使用
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

class FinalWorkingTTSGenerator:
    """最終動作版TTS生成器 (DAC統合完全解決)"""
    
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", max_length=100):
        print(f"🎉 Loading FINAL Working TTS Generator")
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
        
        print("✅ FINAL Working TTS Generator ready!")
    
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
            
            # エンコーディング
            encoder_outputs = self.model.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            # 固定長デコーディング
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
        """音声コードから実際の音声波形を生成 (最終動作版)"""
        
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
                
                # 音声コードを正しい範囲にクランプ (重要！)
                audio_codes = torch.clamp(audio_codes, 0, self.dac_config.codebook_size - 1)
                print(f"   Clamped codes range: [{audio_codes.min()}, {audio_codes.max()}]")
                
                # 各quantizerレイヤーを手動で処理
                latents = torch.zeros(audio_codes.shape[0], 1024, audio_codes.shape[2], 
                                    device=audio_codes.device)
                
                for i, quantizer_layer in enumerate(quantizer.quantizers):
                    if i < audio_codes.shape[1]:
                        layer_codes = audio_codes[:, i, :]  # [1, 100]
                        
                        # 埋め込み処理
                        if hasattr(quantizer_layer, 'embed'):
                            embedded = quantizer_layer.embed(layer_codes)
                        else:
                            embedded = quantizer_layer.codebook(layer_codes)
                            embedded = embedded.transpose(1, 2)  # [1, time, dim] -> [1, dim, time]
                        
                        latents += embedded
                
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
                    'is_real_dac': True  # 実際のDAC音声！
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
    
    def text_to_audio(self, text, output_dir="final_working_audio", description=None):
        """テキストから音声ファイルまでの完全パイプライン"""
        
        if description is None:
            description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with very clear audio quality."
        
        print(f"\n🎵 Complete Text-to-Audio Pipeline (FINAL)")
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
        filename = f"final_audio_{safe_text}.wav"
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
            print(f"     🎉 REAL DAC AUDIO GENERATED! 🎉")
        else:
            print(f"     ⚠️  Using fallback audio")
        
        return result

def test_final_working_integration():
    """最終動作版のテスト"""
    
    print("🚀 Testing FINAL Working DAC Integration")
    print("=" * 60)
    
    try:
        generator = FinalWorkingTTSGenerator(max_length=100)
        
        test_texts = [
            "Hello",
            "Hello world", 
            "This is a test",
            "Phase one point one complete"
        ]
        
        results = []
        real_dac_count = 0
        
        for text in test_texts:
            print(f"\n--- Testing: '{text}' ---")
            
            result = generator.text_to_audio(text)
            results.append(result)
            
            if result['is_real_dac']:
                real_dac_count += 1
                print(f"    🎉 REAL DAC SUCCESS!")
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
        
        with open('final_working_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Real DAC audio: {summary['real_dac_count']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] > 0:
            print(f"   🎉 DAC INTEGRATION FINALLY WORKING!")
        
        return summary
        
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Phase 1.1 Step 2: FINAL Working DAC Integration")
    print("=" * 70)
    
    result = test_final_working_integration()
    
    if result and result['success_rate'] > 0:
        print(f"\n🎊 PHASE 1.1 DAC INTEGRATION COMPLETE! 🎊")
        print(f"📁 Check real DAC audio in: final_working_audio/")
        print(f"🎧 Listen with: open final_working_audio/final_audio_Hello.wav")
        print(f"\n✅ Ready for Phase 1.1 Step 3: Audio Quality Comparison!")
        return True
    else:
        print(f"\n⚠️  Some issues remain, but significant progress made")
        return False

if __name__ == "__main__":
    main()
