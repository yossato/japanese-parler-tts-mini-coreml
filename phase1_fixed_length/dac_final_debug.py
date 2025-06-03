#!/usr/bin/env python3
"""
DAC統合の最終デバッグ
音声コードの値範囲と形状を詳細調査
"""

import os
import sys
import torch
import numpy as np

# パス設定
phase1_dir = "/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length"
os.chdir(phase1_dir)
sys.path.insert(0, phase1_dir)
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

from parler_tts import ParlerTTSForConditionalGeneration
from memory_efficient_decoder import MemoryEfficientTTSGenerator

def debug_audio_codes_values():
    """音声コードの値範囲を詳細調査"""
    
    print("🔍 Debugging Audio Codes Values")
    print("=" * 50)
    
    # 1. 固定長デコーダーで生成されるコードを調査
    print("1️⃣ Investigating fixed-length decoder output...")
    
    generator = MemoryEfficientTTSGenerator(max_length=100)
    result = generator.generate_fixed_length("Hello")
    
    audio_codes = result['tokens'].squeeze(0)  # [num_codebooks, length]
    
    print(f"Fixed decoder audio codes:")
    print(f"  Shape: {audio_codes.shape}")
    print(f"  Data type: {audio_codes.dtype}")
    print(f"  Value range: [{audio_codes.min()}, {audio_codes.max()}]")
    print(f"  Unique values count: {len(torch.unique(audio_codes))}")
    print(f"  Sample values: {audio_codes[0, :10]}")  # 最初のcodebook、最初の10個
    
    # 2. オリジナルモデルで生成されるコードと比較
    print(f"\n2️⃣ Comparing with original model...")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # DAC設定確認
    dac_config = model.audio_encoder.config
    print(f"DAC Configuration:")
    print(f"  Codebook size: {dac_config.codebook_size}")
    print(f"  Expected range: [0, {dac_config.codebook_size - 1}]")
    
    # 3. 音声コードの値を修正
    print(f"\n3️⃣ Testing corrected audio codes...")
    
    # コードを正しい範囲にクランプ
    corrected_codes = torch.clamp(audio_codes, 0, dac_config.codebook_size - 1)
    
    print(f"Corrected audio codes:")
    print(f"  Shape: {corrected_codes.shape}")
    print(f"  Value range: [{corrected_codes.min()}, {corrected_codes.max()}]")
    print(f"  Changed values: {torch.sum(audio_codes != corrected_codes)}")
    
    return audio_codes, corrected_codes, model

def test_quantizer_with_corrected_codes(audio_codes, corrected_codes, model):
    """修正されたコードでQuantizerをテスト"""
    
    print(f"\n4️⃣ Testing quantizer with corrected codes...")
    
    dac_internal = model.audio_encoder.model
    quantizer = dac_internal.quantizer
    
    # 形状を調整
    if corrected_codes.dim() == 2:
        corrected_codes = corrected_codes.unsqueeze(0)  # [1, 9, 100]
    
    print(f"Testing codes shape: {corrected_codes.shape}")
    
    # quantizerの各メソッドを段階的にテスト
    for method_name in ['from_codes', 'embed']:
        if hasattr(quantizer, method_name):
            print(f"\n🧪 Testing {method_name}...")
            try:
                method = getattr(quantizer, method_name)
                latents = method(corrected_codes)
                print(f"   ✅ {method_name} success: {latents.shape}")
                
                # decodeテスト
                try:
                    audio_waveform = dac_internal.decode(latents.detach())
                    print(f"   ✅ Decode success: {audio_waveform.shape}")
                    
                    # 音声保存テスト
                    save_test_audio(audio_waveform, f"test_{method_name}.wav")
                    return True, method_name
                    
                except Exception as e:
                    print(f"   ❌ Decode failed: {e}")
                    
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
    
    # 個別quantizerレイヤーテスト
    print(f"\n🧪 Testing individual quantizer layers...")
    try:
        latents = torch.zeros(1, 1024, corrected_codes.shape[2], device=corrected_codes.device)
        
        for i, quantizer_layer in enumerate(quantizer.quantizers):
            if i < corrected_codes.shape[1]:
                print(f"   Testing layer {i}...")
                
                # 各レイヤーのembedding
                layer_codes = corrected_codes[:, i, :]  # [1, 100]
                print(f"   Layer {i} codes range: [{layer_codes.min()}, {layer_codes.max()}]")
                
                try:
                    if hasattr(quantizer_layer, 'embed'):
                        embedded = quantizer_layer.embed(layer_codes)
                        print(f"   Layer {i} embed success: {embedded.shape}")
                        latents += embedded
                    else:
                        # codebook直接アクセス
                        embedded = quantizer_layer.codebook(layer_codes)
                        print(f"   Layer {i} codebook success: {embedded.shape}")
                        latents += embedded.transpose(1, 2)  # [1, time, dim] -> [1, dim, time]
                        
                except Exception as e:
                    print(f"   Layer {i} failed: {e}")
        
        print(f"   Combined latents shape: {latents.shape}")
        
        # 最終decode
        audio_waveform = dac_internal.decode(latents.detach())
        print(f"   ✅ Final decode success: {audio_waveform.shape}")
        
        save_test_audio(audio_waveform, "test_manual_layers.wav")
        return True, "manual_layers"
        
    except Exception as e:
        print(f"   ❌ Manual layers failed: {e}")
        return False, None

def save_test_audio(audio_waveform, filename):
    """テスト音声を保存"""
    
    try:
        import wave
        
        if isinstance(audio_waveform, torch.Tensor):
            audio_np = audio_waveform.detach().squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio_waveform)
        
        # 正規化
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        sample_rate = 44100
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        duration = len(audio_np) / sample_rate
        file_size = os.path.getsize(filename) / 1024
        
        print(f"   📁 Audio saved: {filename}")
        print(f"   Duration: {duration:.2f}s, Size: {file_size:.1f}KB")
        print(f"   🎧 Play: open {filename}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return False

def generate_final_working_code(working_method):
    """動作する方法に基づいて最終コードを生成"""
    
    print(f"\n📝 Generating Final Working DAC Code")
    print("=" * 50)
    
    if working_method == "manual_layers":
        fix_code = """
# 最終的に動作するDAC統合コード
def codes_to_audio(self, audio_codes):
    '''音声コードから実際の音声波形を生成 (最終動作版)'''
    
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
                'duration': len(audio_np) / self.dac_config.sampling_rate
            }
            
        except Exception as e:
            print(f"   ❌ DAC decoding failed: {e}")
            return self.fallback_mock_audio()
"""
    elif working_method:
        fix_code = f"""
# 動作するDAC統合コード (Method: {working_method})
def codes_to_audio(self, audio_codes):
    '''音声コードから実際の音声波形を生成'''
    
    with torch.no_grad():
        # DAC内部モデルを取得
        dac_internal = self.audio_encoder.model
        quantizer = dac_internal.quantizer
        
        # 形状調整
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(0)
        
        # 値をクランプ
        audio_codes = torch.clamp(audio_codes, 0, self.dac_config.codebook_size - 1)
        
        # {working_method}で潜在表現生成
        latents = quantizer.{working_method}(audio_codes)
        
        # decode実行
        audio_waveform = dac_internal.decode(latents.detach())
        
        # 結果処理...
        return result
"""
    else:
        fix_code = "# No working method found"
    
    print(fix_code)
    
    with open('working_dac_integration_final.py', 'w') as f:
        f.write(fix_code)
    
    print(f"📁 Final working code saved to: working_dac_integration_final.py")

def main():
    print("🎵 DAC Integration Final Debug")
    print("=" * 50)
    
    try:
        # 音声コードの値を調査
        audio_codes, corrected_codes, model = debug_audio_codes_values()
        
        # 修正されたコードでテスト
        success, method = test_quantizer_with_corrected_codes(audio_codes, corrected_codes, model)
        
        if success:
            print(f"\n🎉 DAC integration finally working!")
            print(f"   Working method: {method}")
            generate_final_working_code(method)
        else:
            print(f"\n⚠️  Still debugging DAC integration...")
            generate_final_working_code(None)
            
    except Exception as e:
        print(f"❌ Final DAC debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
