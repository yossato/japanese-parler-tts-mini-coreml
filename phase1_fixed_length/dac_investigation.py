#!/usr/bin/env python3
"""
Parler TTSの実際のDAC使用方法を調査
公式の生成プロセスからDAC統合の正しい方法を学習
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
from transformers import AutoTokenizer

def investigate_official_generation():
    """Parler TTSの公式生成プロセスを調査"""
    
    print("🔍 Investigating Official Parler TTS Generation Process")
    print("=" * 60)
    
    # モデルとトークナイザーロード
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    
    # テキスト入力準備
    description = "A female speaker with a clear voice."
    text = "Hello world"
    
    inputs = tokenizer(description, text, return_tensors="pt", truncation=True, max_length=50)
    
    print(f"📝 Input text: '{text}'")
    print(f"📝 Description: '{description}'")
    print(f"📊 Input shape: {inputs.input_ids.shape}")
    
    # 公式生成プロセス実行（短時間で）
    print(f"\n🔄 Running official generation process...")
    
    try:
        with torch.no_grad():
            # 短い生成でテスト
            generation = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=50  # 短く設定
            )
        
        print(f"✅ Official generation successful!")
        print(f"📊 Generation shape: {generation.shape}")
        
        # 生成されたコードを分析
        analyze_generated_codes(generation, model)
        
        return generation, model
        
    except Exception as e:
        print(f"❌ Official generation failed: {e}")
        return None, model

def analyze_generated_codes(generation, model):
    """生成されたコードを分析してDAC統合のヒントを得る"""
    
    print(f"\n🔬 Analyzing Generated Audio Codes")
    print("=" * 40)
    
    print(f"Raw generation shape: {generation.shape}")
    print(f"Raw generation dtype: {generation.dtype}")
    print(f"Raw generation range: [{generation.min()}, {generation.max()}]")
    
    # Parler TTSのデコード方法を調査
    dac_model = model.audio_encoder
    decoder_config = model.decoder.config
    
    print(f"\n📊 Model Configuration:")
    print(f"   Num codebooks: {decoder_config.num_codebooks}")
    print(f"   Vocab size: {decoder_config.vocab_size}")
    print(f"   DAC num codebooks: {dac_model.config.num_codebooks}")
    print(f"   DAC codebook size: {dac_model.config.codebook_size}")
    
    # 生成されたコードをDAC用に変換
    try:
        # Parler TTSの生成結果をcodebook形式に変換
        print(f"\n🔄 Converting generated codes to DAC format...")
        
        # 生成されたトークンをreshape
        # generation: [batch_size * num_codebooks, length] -> [batch_size, num_codebooks, length]
        batch_size = 1
        num_codebooks = decoder_config.num_codebooks
        seq_length = generation.shape[-1] // num_codebooks
        
        print(f"Calculated dimensions:")
        print(f"   Batch size: {batch_size}")
        print(f"   Num codebooks: {num_codebooks}")
        print(f"   Sequence length: {seq_length}")
        print(f"   Total tokens: {generation.shape[-1]}")
        
        if generation.shape[-1] % num_codebooks == 0:
            # 正確に分割可能
            audio_codes = generation.view(batch_size, num_codebooks, seq_length)
            print(f"✅ Reshaped to: {audio_codes.shape}")
            
            # DAC decodeを試行
            test_dac_with_real_codes(audio_codes, dac_model)
        else:
            print(f"⚠️  Cannot evenly divide generation into codebooks")
            print(f"   Total tokens: {generation.shape[-1]}")
            print(f"   Num codebooks: {num_codebooks}")
            print(f"   Remainder: {generation.shape[-1] % num_codebooks}")
            
    except Exception as e:
        print(f"❌ Code analysis failed: {e}")
        import traceback
        traceback.print_exc()

def test_dac_with_real_codes(audio_codes, dac_model):
    """実際の生成コードでDACテスト"""
    
    print(f"\n🧪 Testing DAC with Real Generated Codes")
    print("=" * 50)
    
    print(f"Audio codes shape: {audio_codes.shape}")
    print(f"Audio codes dtype: {audio_codes.dtype}")
    print(f"Audio codes range: [{audio_codes.min()}, {audio_codes.max()}]")
    
    # 様々なパターンでテスト
    patterns = [
        ("Direct", audio_codes),
        ("Squeezed", audio_codes.squeeze()),
        ("Float", audio_codes.float()),
        ("Clipped", torch.clamp(audio_codes, 0, 1023)),
    ]
    
    for pattern_name, codes in patterns:
        print(f"\n{pattern_name} pattern - Shape: {codes.shape}")
        try:
            batch_size = codes.shape[0] if codes.dim() >= 3 else 1
            audio_scales = torch.ones(batch_size, device=codes.device)
            
            result = dac_model.decode(codes, audio_scales)
            print(f"   ✅ Success! Output shape: {result.shape}")
            
            # 音声保存テスト
            save_real_dac_audio(result, pattern_name)
            return codes, audio_scales, pattern_name
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n❌ All real code patterns failed")
    return None, None, None

def save_real_dac_audio(audio_waveform, pattern_name):
    """実際のDAC音声を保存"""
    
    try:
        import wave
        
        if isinstance(audio_waveform, torch.Tensor):
            audio_np = audio_waveform.squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio_waveform)
        
        # 正規化
        if np.max(np.abs(audio_np)) > 0:
            audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
        
        filename = f"real_dac_audio_{pattern_name.lower()}.wav"
        sample_rate = 44100
        
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        duration = len(audio_np) / sample_rate
        file_size = os.path.getsize(filename) / 1024
        
        print(f"   📁 Saved: {filename}")
        print(f"   Duration: {duration:.2f}s, Size: {file_size:.1f}KB")
        print(f"   🎧 Play: open {filename}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return False

def investigate_dac_internals():
    """DACの内部動作を詳しく調査"""
    
    print(f"\n🔬 Deep DAC Investigation")
    print("=" * 40)
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    dac_model = model.audio_encoder
    
    # DAC内部のquantizerを調査
    if hasattr(dac_model.model, 'quantizer'):
        quantizer = dac_model.model.quantizer
        print(f"📊 Quantizer info:")
        print(f"   Type: {type(quantizer)}")
        print(f"   Num quantizers: {len(quantizer.quantizers) if hasattr(quantizer, 'quantizers') else 'unknown'}")
        
        # quantizer.decodeメソッドを試す
        if hasattr(quantizer, 'decode'):
            print(f"   Has decode method: ✅")
            
            # テスト用のquantized codes作成
            test_codes = torch.randint(0, 1024, (1, 9, 100))  # [batch, num_q, seq_len]
            
            try:
                # quantizer経由でdecode
                decoded = quantizer.decode(test_codes)
                print(f"   Quantizer decode success: {decoded.shape}")
                
                # decoderに通す
                if hasattr(dac_model.model, 'decoder'):
                    final_audio = dac_model.model.decoder(decoded)
                    print(f"   Final audio shape: {final_audio.shape}")
                    
                    save_real_dac_audio(final_audio, "quantizer_path")
                    return True
                    
            except Exception as e:
                print(f"   Quantizer decode failed: {e}")
    
    return False

def main():
    print("🎵 Parler TTS DAC Investigation")
    print("=" * 50)
    
    try:
        # 1. 公式生成プロセス調査
        generation, model = investigate_official_generation()
        
        if generation is not None:
            print(f"\n✅ Official generation analysis completed")
        else:
            print(f"\n⚠️  Official generation failed, trying alternative approach")
            
        # 2. DAC内部構造調査
        print(f"\n" + "="*50)
        success = investigate_dac_internals()
        
        if success:
            print(f"\n🎯 DAC integration method found!")
        else:
            print(f"\n⚠️  Still investigating DAC integration...")
            
    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
