#!/usr/bin/env python3
"""
DAC統合のデバッグと修正
Parler TTSのDAC（音声デコーダー）の正しい使用方法を調査・実装
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

def investigate_dac_model():
    """DACモデルの構造と使用方法を調査"""
    
    print("🔍 DAC Model Investigation")
    print("=" * 40)
    
    # Parler TTSモデルをロード
    print("Loading Parler TTS model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # DACモデルを取得
    dac_model = model.audio_encoder
    print(f"✅ DAC model loaded: {type(dac_model)}")
    
    # DACモデルの詳細調査
    print(f"\n📊 DAC Model Details:")
    print(f"   Model type: {dac_model.config.model_type}")
    print(f"   Num codebooks: {dac_model.config.num_codebooks}")
    print(f"   Codebook size: {dac_model.config.codebook_size}")
    print(f"   Frame rate: {dac_model.config.frame_rate}")
    print(f"   Sampling rate: {dac_model.config.sampling_rate}")
    
    # DACのメソッドを調査
    print(f"\n🔧 DAC Methods:")
    dac_methods = [method for method in dir(dac_model) if not method.startswith('_')]
    for method in dac_methods[:10]:  # 最初の10個を表示
        print(f"   - {method}")
    
    # decode メソッドの詳細調査
    if hasattr(dac_model, 'decode'):
        decode_method = getattr(dac_model, 'decode')
        print(f"\n🎯 decode method: {decode_method}")
        
        # メソッドのシグネチャを取得
        import inspect
        try:
            sig = inspect.signature(decode_method)
            print(f"   Signature: decode{sig}")
            
            for param_name, param in sig.parameters.items():
                print(f"   - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
                if param.default != inspect.Parameter.empty:
                    print(f"     Default: {param.default}")
        except Exception as e:
            print(f"   Could not get signature: {e}")
    
    return dac_model

def test_dac_decode_methods(dac_model):
    """DACのdecodeメソッドの異なる呼び出しパターンをテスト"""
    
    print(f"\n🧪 Testing DAC Decode Methods")
    print("=" * 40)
    
    # テスト用の音声コード作成
    batch_size = 1
    num_codebooks = 9
    seq_length = 100
    
    # ランダムな音声コード（実際の範囲内）
    audio_codes = torch.randint(0, 1024, (batch_size, num_codebooks, seq_length))
    print(f"Test audio codes shape: {audio_codes.shape}")
    
    # パターン1: audio_scales付き
    print(f"\n1️⃣ Testing with audio_scales parameter:")
    try:
        audio_scales = torch.ones(batch_size, device=audio_codes.device)
        result = dac_model.decode(audio_codes, audio_scales)
        print(f"   ✅ Success! Output shape: {result.shape}")
        print(f"   Output type: {type(result)}")
        print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")
        return result, "audio_scales"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン2: 音声コードのみ
    print(f"\n2️⃣ Testing with audio codes only:")
    try:
        result = dac_model.decode(audio_codes)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, "codes_only"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン3: 異なる形状で試行
    print(f"\n3️⃣ Testing with different audio codes shape:")
    try:
        # [num_codebooks, seq_length] -> [1, num_codebooks, seq_length]
        audio_codes_alt = audio_codes.squeeze(0) if audio_codes.shape[0] == 1 else audio_codes
        if audio_codes_alt.dim() == 2:
            audio_codes_alt = audio_codes_alt.unsqueeze(0)
        
        result = dac_model.decode(audio_codes_alt)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, "alt_shape"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # パターン4: forward メソッドでの復号化
    print(f"\n4️⃣ Testing with forward method:")
    try:
        # forwardメソッドを試行
        with torch.no_grad():
            result = dac_model(audio_codes, decode=True)
        print(f"   ✅ Success! Output shape: {result.shape}")
        return result, "forward"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n❌ All decode methods failed!")
    return None, None

def create_fixed_dac_integration():
    """修正されたDAC統合コードを作成"""
    
    print(f"\n🔧 Creating Fixed DAC Integration")
    print("=" * 40)
    
    # DACモデルを調査
    dac_model = investigate_dac_model()
    
    # decode メソッドをテスト
    working_audio, working_method = test_dac_decode_methods(dac_model)
    
    if working_audio is not None:
        print(f"\n✅ Found working method: {working_method}")
        
        # 音声ファイルに保存してテスト
        audio_np = working_audio.squeeze().cpu().numpy()
        
        # 正規化
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        # WAVファイル保存
        import wave
        sample_rate = dac_model.config.sampling_rate
        output_path = "dac_test_output.wav"
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"   📁 Test audio saved: {output_path}")
        print(f"   Duration: {len(audio_np)/sample_rate:.2f}s")
        print(f"   🎧 Play with: open {output_path}")
        
        return working_method
    else:
        print(f"\n❌ No working DAC decode method found")
        return None

def main():
    print("🎵 DAC Integration Debug and Fix")
    print("=" * 50)
    
    try:
        working_method = create_fixed_dac_integration()
        
        if working_method:
            print(f"\n🎯 Recommended DAC Integration Method: {working_method}")
            
            # dac_integration.pyの修正コードを生成
            print(f"\n📝 Generating fixed dac_integration.py code...")
            generate_fixed_code(working_method)
        else:
            print(f"\n⚠️  DAC integration requires further investigation")
            print(f"   Current fallback (mock audio) remains active")
            
    except Exception as e:
        print(f"❌ DAC debug failed: {e}")
        import traceback
        traceback.print_exc()

def generate_fixed_code(working_method):
    """動作する方法に基づいて修正コードを生成"""
    
    if working_method == "audio_scales":
        fix_code = """
# 修正版のDAC decode呼び出し
audio_scales = torch.ones(audio_codes.shape[0], device=audio_codes.device)
audio_waveform = self.audio_encoder.decode(audio_codes, audio_scales)
"""
    elif working_method == "codes_only":
        fix_code = """
# 修正版のDAC decode呼び出し（audio_codesのみ）
audio_waveform = self.audio_encoder.decode(audio_codes)
"""
    elif working_method == "forward":
        fix_code = """
# 修正版のDAC decode呼び出し（forwardメソッド）
audio_waveform = self.audio_encoder(audio_codes, decode=True)
"""
    else:
        fix_code = "# Unknown method"
    
    print(f"📝 Fixed code for dac_integration.py:")
    print(fix_code)

if __name__ == "__main__":
    main()
