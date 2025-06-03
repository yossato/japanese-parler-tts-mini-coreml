#!/usr/bin/env python3
"""
Phase 1の結果から実際の音声を生成するデモ
音声コード → WAVファイル変換
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path

# 音声ファイル書き込み用
import wave
import struct

class SimpleAudioDecoder(nn.Module):
    """シンプルな音声デコーダー（DACの簡略版）"""
    
    def __init__(self, num_codebooks=9, codebook_size=1088, sample_rate=16000):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.sample_rate = sample_rate
        
        # 各codebookのembedding
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, 64) for _ in range(num_codebooks)
        ])
        
        # 音声復元用のデコーダー
        self.audio_decoder = nn.Sequential(
            nn.Linear(num_codebooks * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # -1 to 1の音声信号
        )
        
        print(f"✅ Simple Audio Decoder initialized: {num_codebooks} codebooks -> {sample_rate}Hz audio")
    
    def forward(self, audio_codes):
        """
        Args:
            audio_codes: [num_codebooks, sequence_length] 音声コード
        Returns:
            audio_waveform: [sequence_length] 音声波形
        """
        num_codebooks, seq_len = audio_codes.shape
        
        # 各codebookのembedding取得
        embeddings = []
        for i in range(num_codebooks):
            emb = self.codebook_embeddings[i](audio_codes[i])  # [seq_len, 64]
            embeddings.append(emb)
        
        # 結合
        combined = torch.cat(embeddings, dim=-1)  # [seq_len, num_codebooks * 64]
        
        # 音声波形生成
        audio_waveform = self.audio_decoder(combined)  # [seq_len, 1]
        audio_waveform = audio_waveform.squeeze(-1)  # [seq_len]
        
        return audio_waveform

def load_phase1_results():
    """Phase 1の結果をロード"""
    
    results_file = 'phase1_minimal_results.json'
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found. Please run Phase 1 first.")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Phase 1 results loaded from {results_file}")
    return results

def create_mock_audio_codes(texts, seq_length=500, num_codebooks=9, codebook_size=1088):
    """Phase 1の結果をシミュレートして音声コードを作成"""
    
    print(f"Creating mock audio codes for {len(texts)} texts...")
    
    audio_codes_list = []
    
    for i, text in enumerate(texts):
        # テキストの長さに基づいてvariation追加
        seed = sum(ord(c) for c in text) + i
        torch.manual_seed(seed)
        
        # ランダムな音声コード生成（実際のPhase 1では学習されたモデルが生成）
        audio_codes = torch.randint(0, codebook_size, (num_codebooks, seq_length))
        
        # テキストに応じた特徴を少し追加
        if "Hello" in text:
            # "Hello"系の音は高周波成分を強調
            audio_codes[0:3] = torch.clamp(audio_codes[0:3] + 100, 0, codebook_size-1)
        elif "test" in text:
            # "test"系の音は中周波成分を強調
            audio_codes[3:6] = torch.clamp(audio_codes[3:6] + 50, 0, codebook_size-1)
        
        audio_codes_list.append(audio_codes)
        print(f"  Text {i+1}: \"{text}\" -> codes shape {audio_codes.shape}")
    
    return audio_codes_list

def convert_codes_to_audio(audio_codes, text, output_dir='generated_audio'):
    """音声コードを実際の音声ファイルに変換"""
    
    # 出力ディレクトリ作成
    Path(output_dir).mkdir(exist_ok=True)
    
    # 音声デコーダー
    audio_decoder = SimpleAudioDecoder()
    
    # 音声生成（学習済みではないので、構造のデモのみ）
    with torch.no_grad():
        audio_waveform = audio_decoder(audio_codes)
    
    # NumPy配列に変換
    audio_np = audio_waveform.numpy()
    
    # 正規化
    audio_np = audio_np / np.max(np.abs(audio_np)) * 0.8
    
    # サンプリングレート設定
    sample_rate = 16000
    duration = len(audio_np) / sample_rate
    
    # WAVファイル保存
    filename = f"generated_{text.replace(' ', '_').replace(',', '')[:20]}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # 16bit PCMで保存
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # モノラル
        wav_file.setsampwidth(2)  # 16bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"  📁 Audio saved: {filepath}")
    print(f"     Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
    
    return filepath

def generate_audio_from_phase1():
    """Phase 1の結果から音声生成"""
    
    print("🎵 Generating Audio from Phase 1 Results")
    print("=" * 50)
    
    # Phase 1結果をロード（なければモック作成）
    results = load_phase1_results()
    
    if results and 'benchmark' in results:
        texts = results['benchmark']['texts']
        print(f"Using texts from Phase 1: {texts}")
    else:
        # デフォルトテキスト
        texts = [
            "Hello",
            "Hello world", 
            "This is a test of the fixed length TTS system"
        ]
        print(f"Using default texts: {texts}")
    
    # 音声コード作成
    audio_codes_list = create_mock_audio_codes(texts)
    
    # 各テキストの音声生成
    generated_files = []
    print(f"\n🔊 Converting {len(texts)} texts to audio...")
    
    for i, (text, audio_codes) in enumerate(zip(texts, audio_codes_list)):
        print(f"\nText {i+1}: \"{text}\"")
        filepath = convert_codes_to_audio(audio_codes, text)
        generated_files.append(filepath)
    
    # 結果保存
    audio_results = {
        'generated_files': generated_files,
        'texts': texts,
        'timestamp': np.datetime64('now').astype(str),
        'note': 'These are demonstration audio files generated from mock audio codes. Real Parler TTS would produce higher quality speech.'
    }
    
    with open('generated_audio_results.json', 'w') as f:
        json.dump(audio_results, f, indent=2)
    
    print(f"\n✅ Audio generation completed!")
    print(f"📋 Summary:")
    print(f"  Generated files: {len(generated_files)}")
    print(f"  Output directory: generated_audio/")
    print(f"  Results saved to: generated_audio_results.json")
    
    print(f"\n📢 Important Note:")
    print(f"  These are DEMO audio files created with random codes.")
    print(f"  Real Parler TTS would require:")
    print(f"  1. Trained audio decoder (DAC)")
    print(f"  2. Proper text-to-code mapping")
    print(f"  3. High-quality pretrained models")
    
    return generated_files

def play_audio_info():
    """生成された音声の再生方法を説明"""
    
    print(f"\n🎧 How to play generated audio:")
    print(f"  macOS: open generated_audio/generated_Hello.wav")
    print(f"  Or use: afplay generated_audio/generated_Hello.wav")
    print(f"  Python: ")
    print(f"    import sounddevice as sd")
    print(f"    import soundfile as sf")
    print(f"    data, fs = sf.read('generated_audio/generated_Hello.wav')")
    print(f"    sd.play(data, fs)")

def main():
    try:
        # 音声生成実行
        generated_files = generate_audio_from_phase1()
        
        # 再生方法の説明
        play_audio_info()
        
        print(f"\n🎯 What's Next:")
        print(f"  1. Listen to the generated demo audio files")
        print(f"  2. Compare with real Parler TTS output (when available)")
        print(f"  3. Proceed to Phase 2 for ANE optimization")
        
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
