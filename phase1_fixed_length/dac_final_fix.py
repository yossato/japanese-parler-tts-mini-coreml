#!/usr/bin/env python3
"""
DAC統合の最終修正版
正しい音声コード形状とaudio_scalesパラメータでテスト
"""

import os
import sys
import torch
import numpy as np
import wave

# パス設定
phase1_dir = "/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length"
os.chdir(phase1_dir)
sys.path.insert(0, phase1_dir)
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

from parler_tts import ParlerTTSForConditionalGeneration

def test_correct_dac_usage():
    """正しいDACの使用方法をテスト"""
    
    print("🔧 Testing Correct DAC Usage")
    print("=" * 40)
    
    # Parler TTSモデルをロード
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    dac_model = model.audio_encoder
    
    # 正しい音声コード形状でテスト
    batch_size = 1
    num_codebooks = 9
    seq_length = 100
    
    print(f"📊 Testing different audio_codes shapes:")
    
    # パターン1: [batch_size, num_codebooks, seq_length]
    print(f"\n1️⃣ Shape: [batch_size, num_codebooks, seq_length] = [1, 9, 100]")
    audio_codes_1 = torch.randint(0, 1024, (batch_size, num_codebooks, seq_length))
    try:
        audio_scales_1 = torch.ones(batch_size)
        result = dac_model.decode(audio_codes_1, audio_scales_1)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, audio_codes_1, audio_scales_1, "pattern_1"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン2: [num_codebooks, seq_length] (バッチなし)
    print(f"\n2️⃣ Shape: [num_codebooks, seq_length] = [9, 100]")
    audio_codes_2 = torch.randint(0, 1024, (num_codebooks, seq_length))
    try:
        audio_scales_2 = torch.ones(1)  # バッチサイズ1
        result = dac_model.decode(audio_codes_2, audio_scales_2)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, audio_codes_2, audio_scales_2, "pattern_2"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン3: unsqueezeでバッチ次元追加
    print(f"\n3️⃣ Shape: unsqueezed [9, 100] -> [1, 9, 100]")
    audio_codes_3 = torch.randint(0, 1024, (num_codebooks, seq_length)).unsqueeze(0)
    try:
        audio_scales_3 = torch.ones(1)
        result = dac_model.decode(audio_codes_3, audio_scales_3)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, audio_codes_3, audio_scales_3, "pattern_3"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン4: 転置してみる [seq_length, num_codebooks] -> [num_codebooks, seq_length]
    print(f"\n4️⃣ Shape: transposed and batched")
    audio_codes_4 = torch.randint(0, 1024, (seq_length, num_codebooks)).transpose(0, 1).unsqueeze(0)
    try:
        audio_scales_4 = torch.ones(1)
        result = dac_model.decode(audio_codes_4, audio_scales_4)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, audio_codes_4, audio_scales_4, "pattern_4"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n❌ All patterns failed!")
    return None, None, None, None

def save_test_audio(audio_waveform, sample_rate, filename):
    """テスト音声を保存"""
    
    if audio_waveform is None:
        return False
    
    try:
        # NumPy配列に変換
        if isinstance(audio_waveform, torch.Tensor):
            audio_np = audio_waveform.squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio_waveform)
        
        # 正規化
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        # WAVファイル保存
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        file_size = os.path.getsize(filename) / 1024  # KB
        duration = len(audio_np) / sample_rate
        
        print(f"   📁 Audio saved: {filename}")
        print(f"   Duration: {duration:.2f}s, Size: {file_size:.1f}KB")
        print(f"   🎧 Play: open {filename}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return False

def main():
    print("🎵 DAC Integration Final Fix")
    print("=" * 50)
    
    try:
        # 正しいDAC使用方法をテスト
        result, audio_codes, audio_scales, pattern = test_correct_dac_usage()
        
        if result is not None:
            print(f"\n✅ Found working pattern: {pattern}")
            print(f"   Audio codes shape: {audio_codes.shape}")
            print(f"   Audio scales shape: {audio_scales.shape}")
            print(f"   Output audio shape: {result.shape}")
            
            # テスト音声を保存
            success = save_test_audio(result, 44100, f"dac_test_{pattern}.wav")
            
            if success:
                print(f"\n🎯 DAC Integration Fixed!")
                print(f"   Working method: {pattern}")
                print(f"   Audio codes: {audio_codes.shape}")
                print(f"   Audio scales: {audio_scales.shape}")
                
                # dac_integration.pyの修正コードを生成
                generate_fixed_dac_code(pattern, audio_codes.shape, audio_scales.shape)
                
                return True
        else:
            print(f"\n❌ DAC integration still not working")
            return False
            
    except Exception as e:
        print(f"❌ Final DAC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_fixed_dac_code(pattern, codes_shape, scales_shape):
    """動作するパターンに基づいて修正コードを生成"""
    
    print(f"\n📝 Generating Fixed DAC Integration Code")
    print("=" * 50)
    
    if pattern == "pattern_1":
        fix_code = f"""
# 修正版のDAC decode呼び出し (Pattern 1)
# audio_codes shape: {codes_shape}
# audio_scales shape: {scales_shape}

# 音声コードの形状を確認・調整
if audio_codes.dim() == 2:
    # [num_codebooks, length] -> [1, num_codebooks, length]
    audio_codes = audio_codes.unsqueeze(0)

# audio_scalesを正しく作成
batch_size = audio_codes.shape[0]
audio_scales = torch.ones(batch_size, device=audio_codes.device)

# DAC decode実行
audio_waveform = self.audio_encoder.decode(audio_codes, audio_scales)
"""
    elif pattern == "pattern_3":
        fix_code = f"""
# 修正版のDAC decode呼び出し (Pattern 3)
# audio_codes shape: {codes_shape}
# audio_scales shape: {scales_shape}

# 音声コードの形状を確認・調整
if audio_codes.dim() == 2:
    # [num_codebooks, length] -> [1, num_codebooks, length]
    audio_codes = audio_codes.unsqueeze(0)

# audio_scalesを作成
audio_scales = torch.ones(1, device=audio_codes.device)

# DAC decode実行
audio_waveform = self.audio_encoder.decode(audio_codes, audio_scales)
"""
    else:
        fix_code = f"# Pattern {pattern} - Custom implementation needed"
    
    print(fix_code)
    
    # ファイルに保存
    with open('dac_fix_code.txt', 'w') as f:
        f.write(fix_code)
    
    print(f"📁 Fixed code saved to: dac_fix_code.txt")

if __name__ == "__main__":
    main()
