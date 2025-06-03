#!/usr/bin/env python3
"""
DAC Quantizer経由での音声生成テスト
ResidualVectorQuantizeのdecodeメソッドを使用
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

def test_quantizer_decode():
    """DAC Quantizer経由での音声生成テスト"""
    
    print("🔬 Testing DAC Quantizer Decode Method")
    print("=" * 50)
    
    # モデルロード
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    dac_model = model.audio_encoder
    quantizer = dac_model.model.quantizer
    decoder = dac_model.model.decoder
    
    print(f"✅ DAC components loaded:")
    print(f"   Quantizer: {type(quantizer)}")
    print(f"   Decoder: {type(decoder)}")
    print(f"   Num quantizers: {len(quantizer.quantizers)}")
    
    # テスト用音声コード作成
    batch_size = 1
    num_codebooks = 9
    seq_length = 100
    
    # 様々な形状でテスト
    test_patterns = [
        ("Pattern A: [1, 9, 100]", torch.randint(0, 1024, (batch_size, num_codebooks, seq_length))),
        ("Pattern B: [9, 100]", torch.randint(0, 1024, (num_codebooks, seq_length))),
        ("Pattern C: [100, 9]", torch.randint(0, 1024, (seq_length, num_codebooks))),
    ]
    
    for pattern_name, audio_codes in test_patterns:
        print(f"\n🧪 {pattern_name}")
        print(f"   Shape: {audio_codes.shape}")
        
        try:
            # Quantizer decode実行
            print("   Testing quantizer.decode()...")
            decoded_latents = quantizer.decode(audio_codes)
            print(f"   ✅ Quantizer decode success: {decoded_latents.shape}")
            
            # Decoder実行
            print("   Testing decoder...")
            audio_waveform = decoder(decoded_latents)
            print(f"   ✅ Decoder success: {audio_waveform.shape}")
            
            # 音声保存
            save_path = f"quantizer_test_{pattern_name.split()[1].replace(':', '').lower()}.wav"
            if save_audio_file(audio_waveform, save_path):
                print(f"   🎵 Audio saved: {save_path}")
                return audio_codes, decoded_latents, audio_waveform, pattern_name
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n❌ All quantizer patterns failed")
    return None, None, None, None

def test_direct_dac_methods():
    """DAC内部メソッドの直接テスト"""
    
    print(f"\n🔧 Testing Direct DAC Internal Methods")
    print("=" * 50)
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    dac_model = model.audio_encoder
    
    # DAC内部のメソッドをリスト化
    dac_methods = [method for method in dir(dac_model.model) if not method.startswith('_')]
    print(f"📋 DAC model methods: {dac_methods[:10]}...")  # 最初の10個
    
    # 特に関心のあるメソッドをテスト
    interesting_methods = ['decode', 'encode', 'forward', 'quantize']
    
    for method_name in interesting_methods:
        if hasattr(dac_model.model, method_name):
            method = getattr(dac_model.model, method_name)
            print(f"\n🎯 Found method: {method_name}")
            
            # メソッドのシグネチャを取得
            import inspect
            try:
                sig = inspect.signature(method)
                print(f"   Signature: {method_name}{sig}")
            except:
                print(f"   Signature: Could not determine")
            
            # テスト実行
            test_dac_method(method, method_name)

def test_dac_method(method, method_name):
    """個別のDACメソッドをテスト"""
    
    # テスト用データ
    test_codes = torch.randint(0, 1024, (1, 9, 100))
    test_audio = torch.randn(1, 1, 44100)  # 1秒の音声
    
    print(f"   Testing {method_name}...")
    
    try:
        if method_name == 'encode':
            result = method(test_audio)
            print(f"   ✅ {method_name} success: {type(result)}")
            if hasattr(result, 'shape'):
                print(f"      Shape: {result.shape}")
            elif isinstance(result, tuple):
                print(f"      Tuple length: {len(result)}")
                for i, r in enumerate(result):
                    if hasattr(r, 'shape'):
                        print(f"      Element {i}: {r.shape}")
                        
        elif method_name == 'decode':
            # encodeの結果が必要
            encoded = method.__self__.encode(test_audio)
            if isinstance(encoded, tuple):
                result = method(encoded[0])  # 通常、最初の要素がcodes
            else:
                result = method(encoded)
            print(f"   ✅ {method_name} success: {result.shape}")
            
        elif method_name == 'quantize':
            # 潜在表現が必要
            test_latents = torch.randn(1, 1024, 100)  # [batch, latent_dim, time]
            result = method(test_latents)
            print(f"   ✅ {method_name} success: {type(result)}")
            
        else:
            # その他のメソッド
            result = method(test_codes)
            print(f"   ✅ {method_name} success: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ {method_name} failed: {e}")

def save_audio_file(audio_waveform, filename):
    """音声ファイル保存"""
    
    try:
        # テンソルをnumpy配列に変換
        if isinstance(audio_waveform, torch.Tensor):
            audio_np = audio_waveform.squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio_waveform)
        
        # 正規化
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        # WAVファイル保存
        sample_rate = 44100
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        duration = len(audio_np) / sample_rate
        file_size = os.path.getsize(filename) / 1024
        
        print(f"      Duration: {duration:.2f}s, Size: {file_size:.1f}KB")
        print(f"      🎧 Play: open {filename}")
        
        return True
        
    except Exception as e:
        print(f"      Save failed: {e}")
        return False

def create_working_dac_integration():
    """動作するDAC統合方法を作成"""
    
    print(f"\n🎯 Creating Working DAC Integration")
    print("=" * 50)
    
    # Quantizer decodeテスト
    codes, latents, audio, pattern = test_quantizer_decode()
    
    if codes is not None:
        print(f"\n✅ Working DAC integration found!")
        print(f"   Pattern: {pattern}")
        print(f"   Input codes: {codes.shape}")
        print(f"   Latents: {latents.shape}")
        print(f"   Output audio: {audio.shape}")
        
        # 修正コード生成
        generate_working_dac_code(pattern, codes.shape)
        return True
    else:
        print(f"\n🔍 Testing direct DAC methods...")
        test_direct_dac_methods()
        return False

def generate_working_dac_code(pattern, codes_shape):
    """動作するDAC統合コードを生成"""
    
    print(f"\n📝 Generating Working DAC Integration Code")
    print("=" * 50)
    
    fix_code = f"""
# 動作するDAC統合コード (パターン: {pattern})
def codes_to_audio(self, audio_codes):
    '''音声コードから実際の音声波形を生成 (修正版)'''
    
    print(f"Converting codes to audio...")
    print(f"Input codes shape: {{audio_codes.shape}}")
    
    with torch.no_grad():
        start_time = time.time()
        
        # 音声コードの形状を調整
        if audio_codes.dim() == 2:
            # [num_codebooks, length] -> [1, num_codebooks, length] (必要に応じて)
            if audio_codes.shape == {codes_shape[1:]}:  # [9, 100]の場合
                audio_codes = audio_codes.unsqueeze(0)
        
        try:
            # DAC内部のquantizerとdecoderを直接使用
            quantizer = self.audio_encoder.model.quantizer
            decoder = self.audio_encoder.model.decoder
            
            # Step 1: Quantizer decode (codes -> latents)
            decoded_latents = quantizer.decode(audio_codes)
            print(f"Quantizer decode: {{decoded_latents.shape}}")
            
            # Step 2: Decoder (latents -> audio)
            audio_waveform = decoder(decoded_latents)
            print(f"Decoder output: {{audio_waveform.shape}}")
            
            if isinstance(audio_waveform, torch.Tensor):
                audio_np = audio_waveform.squeeze().cpu().numpy()
            else:
                audio_np = np.array(audio_waveform)
            
            end_time = time.time()
            
            print(f"Audio conversion time: {{end_time - start_time:.3f}}s")
            print(f"Audio waveform shape: {{audio_np.shape}}")
            print(f"Audio range: [{{audio_np.min():.3f}}, {{audio_np.max():.3f}}]")
            
            return {{
                'audio_waveform': audio_np,
                'conversion_time': end_time - start_time,
                'sample_rate': self.dac_config.sampling_rate,
                'duration': len(audio_np) / self.dac_config.sampling_rate
            }}
            
        except Exception as e:
            print(f"DAC integration failed: {{e}}")
            # フォールバック処理...
            return self.fallback_mock_audio(audio_codes)
"""
    
    print(fix_code)
    
    # ファイルに保存
    with open('working_dac_integration.py', 'w') as f:
        f.write(fix_code)
    
    print(f"\n📁 Working DAC integration saved to: working_dac_integration.py")

def main():
    print("🎵 DAC Quantizer Integration Test")
    print("=" * 50)
    
    try:
        success = create_working_dac_integration()
        
        if success:
            print(f"\n🎉 DAC integration successfully resolved!")
        else:
            print(f"\n⚠️  Further investigation needed")
            
    except Exception as e:
        print(f"❌ DAC quantizer test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
