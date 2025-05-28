#!/usr/bin/env python3
"""
Phase 3: 全体をANE（Apple Neural Engine）で動作させる
Text Encoder + TTS Decoder + Audio Decoder の完全なANEパイプライン
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')
sys.path.append('/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length')
sys.path.append('/Users/yoshiaki/Projects/parler_tts_experiment/phase2_ane_decoder')

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time
import json
from fixed_length_decoder import SimplifiedTTSGenerator
from ane_decoder import ANEOptimizedTTSDecoder

class ANEOptimizedTextEncoder(nn.Module):
    """ANE最適化されたテキストエンコーダー"""
    
    def __init__(self, vocab_size=32128, hidden_size=1024, max_length=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # ANE最適化アーキテクチャ
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformerの代わりにCNNベースエンコーダー（ANEが得意）
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Position encoding (学習可能)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # 最終プロジェクション
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        token_embeds = self.embedding(input_ids)  # [batch, seq_len, hidden]
        
        # Position embedding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # 結合
        x = token_embeds + pos_embeds
        
        # CNN encoder (transpose for conv1d)
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.conv_encoder(x)
        x = x.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # Output projection
        x = self.output_projection(x)
        
        return x

class ANEOptimizedAudioDecoder(nn.Module):
    """ANE最適化されたオーディオデコーダー"""
    
    def __init__(self, num_codebooks=9, codebook_size=1024, audio_length=16000):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_length = audio_length
        
        # Codebook embeddings
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, 64) for _ in range(num_codebooks)
        ])
        
        # CNNベースの音声デコーダー（ANE最適化）
        self.audio_decoder = nn.Sequential(
            nn.Conv1d(num_codebooks * 64, 512, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(256, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, audio_codes):
        """
        Args:
            audio_codes: [batch_size, num_codebooks, seq_len]
        Returns:
            audio_waveform: [batch_size, audio_length]
        """
        batch_size, num_codebooks, seq_len = audio_codes.shape
        
        # Codebook embeddings
        embeddings = []
        for i, embedding_layer in enumerate(self.codebook_embeddings):
            codebook_embed = embedding_layer(audio_codes[:, i, :])  # [batch, seq_len, 64]
            embeddings.append(codebook_embed)
        
        # Concatenate embeddings
        x = torch.cat(embeddings, dim=-1)  # [batch, seq_len, num_codebooks * 64]
        x = x.transpose(1, 2)  # [batch, num_codebooks * 64, seq_len]
        
        # CNN decoder
        audio = self.audio_decoder(x)  # [batch, 1, seq_len]
        audio = audio.squeeze(1)  # [batch, seq_len]
        
        # Interpolate to target audio length
        if audio.shape[1] != self.audio_length:
            audio = torch.nn.functional.interpolate(
                audio.unsqueeze(1),
                size=self.audio_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        return audio

class FullANEPipeline:
    """完全なANEパイプライン"""
    
    def __init__(self):
        self.batch_size = 1
        self.text_max_length = 50
        self.hidden_size = 1024
        self.audio_max_length = 500
        self.audio_output_length = 16000
        
        # ANE最適化モデル作成
        self.text_encoder = ANEOptimizedTextEncoder(
            vocab_size=32128,
            hidden_size=self.hidden_size,
            max_length=self.text_max_length
        )
        
        self.tts_decoder = ANEOptimizedTTSDecoder(
            hidden_size=self.hidden_size,
            num_codebooks=9,
            vocab_size=1088,
            max_length=self.audio_max_length
        )
        
        self.audio_decoder = ANEOptimizedAudioDecoder(
            num_codebooks=9,
            codebook_size=1088,
            audio_length=self.audio_output_length
        )
    
    def convert_text_encoder_to_coreml(self):
        """テキストエンコーダーをCore MLに変換"""
        
        print("Converting Text Encoder to Core ML...")
        
        self.text_encoder.eval()
        
        # サンプル入力
        sample_input = torch.randint(
            0, 32128, 
            (self.batch_size, self.text_max_length)
        )
        
        # トレーシング
        with torch.no_grad():
            traced_model = torch.jit.trace(self.text_encoder, sample_input)
        
        # Core ML変換
        text_encoder_coreml = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=(self.batch_size, self.text_max_length),
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(name="hidden_states", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="neuralnetwork"
        )
        
        # 最適化
        text_encoder_coreml = self._optimize_model(text_encoder_coreml, "Text Encoder")
        
        model_path = "ane_text_encoder.mlmodel"
        text_encoder_coreml.save(model_path)
        print(f"Text encoder saved to: {model_path}")
        
        return model_path
    
    def convert_tts_decoder_to_coreml(self):
        """TTSデコーダーをCore MLに変換"""
        
        print("Converting TTS Decoder to Core ML...")
        
        self.tts_decoder.eval()
        
        # サンプル入力
        sample_input = torch.randn(
            self.batch_size,
            self.text_max_length,
            self.hidden_size
        )
        
        # トレーシング
        with torch.no_grad():
            traced_model = torch.jit.trace(self.tts_decoder, sample_input)
        
        # Core ML変換
        tts_decoder_coreml = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="encoder_hidden_states",
                    shape=(self.batch_size, self.text_max_length, self.hidden_size),
                    dtype=np.float32
                )
            ],
            outputs=[
                ct.TensorType(name="audio_codes", dtype=np.int32),
                ct.TensorType(name="predicted_length", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="neuralnetwork"
        )
        
        # 最適化
        tts_decoder_coreml = self._optimize_model(tts_decoder_coreml, "TTS Decoder")
        
        model_path = "ane_tts_decoder_full.mlmodel"
        tts_decoder_coreml.save(model_path)
        print(f"TTS decoder saved to: {model_path}")
        
        return model_path
    
    def convert_audio_decoder_to_coreml(self):
        """オーディオデコーダーをCore MLに変換"""
        
        print("Converting Audio Decoder to Core ML...")
        
        self.audio_decoder.eval()
        
        # サンプル入力
        sample_input = torch.randint(
            0, 1088,
            (self.batch_size, 9, self.audio_max_length)
        )
        
        # トレーシング
        with torch.no_grad():
            traced_model = torch.jit.trace(self.audio_decoder, sample_input)
        
        # Core ML変換
        audio_decoder_coreml = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="audio_codes",
                    shape=(self.batch_size, 9, self.audio_max_length),
                    dtype=np.int32
                )
            ],
            outputs=[
                ct.TensorType(name="audio_waveform", dtype=np.float32)
            ],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS13,
            convert_to="neuralnetwork"
        )
        
        # 最適化
        audio_decoder_coreml = self._optimize_model(audio_decoder_coreml, "Audio Decoder")
        
        model_path = "ane_audio_decoder.mlmodel"
        audio_decoder_coreml.save(model_path)
        print(f"Audio decoder saved to: {model_path}")
        
        return model_path
    
    def _optimize_model(self, model, model_name):
        """モデル最適化"""
        try:
            # 8bit量子化
            quantized = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8
            )
            print(f"{model_name}: 8-bit quantization applied")
            return quantized
        except Exception as e:
            print(f"{model_name}: Quantization failed ({e}), using original")
            return model
    
    def benchmark_full_pipeline(self, text_encoder_path, tts_decoder_path, audio_decoder_path):
        """完全パイプラインのベンチマーク"""
        
        print("Benchmarking Full ANE Pipeline...")
        
        # Core MLモデルロード
        text_encoder = ct.models.MLModel(text_encoder_path)
        tts_decoder = ct.models.MLModel(tts_decoder_path)
        audio_decoder = ct.models.MLModel(audio_decoder_path)
        
        # サンプル入力準備
        sample_text_input = np.random.randint(
            1, 32128, 
            (self.batch_size, self.text_max_length)
        ).astype(np.int32)
        
        num_runs = 5
        pipeline_times = []
        component_times = {'text_encoder': [], 'tts_decoder': [], 'audio_decoder': []}
        
        for i in range(num_runs):
            total_start = time.time()
            
            # 1. Text Encoding
            te_start = time.time()
            text_result = text_encoder.predict({"input_ids": sample_text_input})
            te_time = time.time() - te_start
            component_times['text_encoder'].append(te_time)
            
            # 2. TTS Decoding
            td_start = time.time()
            tts_result = tts_decoder.predict({
                "encoder_hidden_states": text_result["hidden_states"]
            })
            td_time = time.time() - td_start
            component_times['tts_decoder'].append(td_time)
            
            # 3. Audio Decoding
            ad_start = time.time()
            audio_result = audio_decoder.predict({
                "audio_codes": tts_result["audio_codes"]
            })
            ad_time = time.time() - ad_start
            component_times['audio_decoder'].append(ad_time)
            
            total_time = time.time() - total_start
            pipeline_times.append(total_time)
            
            if i == 0:
                print(f"Output shapes verification:")
                print(f"  Text encoder: {text_result['hidden_states'].shape}")
                print(f"  TTS decoder: {tts_result['audio_codes'].shape}")
                print(f"  Audio decoder: {audio_result['audio_waveform'].shape}")
        
        # 結果集計
        results = {
            'pipeline_avg_ms': np.mean(pipeline_times) * 1000,
            'pipeline_std_ms': np.std(pipeline_times) * 1000,
            'component_times_ms': {
                name: {
                    'avg': np.mean(times) * 1000,
                    'std': np.std(times) * 1000
                }
                for name, times in component_times.items()
            }
        }
        
        print(f"\n📊 Full Pipeline Benchmark Results:")
        print(f"Total pipeline time: {results['pipeline_avg_ms']:.2f} ± {results['pipeline_std_ms']:.2f} ms")
        print(f"Component breakdown:")
        for name, stats in results['component_times_ms'].items():
            print(f"  {name}: {stats['avg']:.2f} ± {stats['std']:.2f} ms")
        
        return results

def create_ios_integration_code():
    """iOS統合用のSwiftコード生成"""
    
    swift_code = '''
//
//  ParlerTTSANEPipeline.swift
//  ANE最適化されたParler TTSの完全なiOS実装
//

import Foundation
import CoreML
import AVFoundation

class ParlerTTSANEPipeline {
    
    private let textEncoder: MLModel
    private let ttsDecoder: MLModel
    private let audioDecoder: MLModel
    
    init() throws {
        // Core MLモデルをロード
        guard let textEncoderURL = Bundle.main.url(forResource: "ane_text_encoder", withExtension: "mlmodel"),
              let ttsDecoderURL = Bundle.main.url(forResource: "ane_tts_decoder_full", withExtension: "mlmodel"),
              let audioDecoderURL = Bundle.main.url(forResource: "ane_audio_decoder", withExtension: "mlmodel") else {
            throw TTSError.modelNotFound
        }
        
        textEncoder = try MLModel(contentsOf: textEncoderURL)
        ttsDecoder = try MLModel(contentsOf: ttsDecoderURL)
        audioDecoder = try MLModel(contentsOf: audioDecoderURL)
        
        print("ANE Pipeline initialized successfully")
    }
    
    func synthesizeSpeech(from text: String) async throws -> Data {
        // 1. テキストをトークン化
        let tokens = tokenizeText(text)
        
        // 2. テキストエンコーディング (ANE)
        let textInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: tokens)
        ])
        
        let textOutput = try await textEncoder.prediction(from: textInput)
        guard let hiddenStates = textOutput.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw TTSError.encodingFailed
        }
        
        // 3. TTS デコーディング (ANE)
        let ttsInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_hidden_states": MLFeatureValue(multiArray: hiddenStates)
        ])
        
        let ttsOutput = try await ttsDecoder.prediction(from: ttsInput)
        guard let audioCodes = ttsOutput.featureValue(for: "audio_codes")?.multiArrayValue else {
            throw TTSError.decodingFailed
        }
        
        // 4. オーディオデコーディング (ANE)
        let audioInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_codes": MLFeatureValue(multiArray: audioCodes)
        ])
        
        let audioOutput = try await audioDecoder.prediction(from: audioInput)
        guard let audioWaveform = audioOutput.featureValue(for: "audio_waveform")?.multiArrayValue else {
            throw TTSError.audioGenerationFailed
        }
        
        // 5. WAVファイル形式に変換
        return convertToWAV(audioWaveform)
    }
    
    private func tokenizeText(_ text: String) -> MLMultiArray {
        // 簡単な文字レベルトークン化（実際の実装では適切なトークナイザーを使用）
        let maxLength = 50
        let tokens = text.compactMap { $0.asciiValue }.map { Int32($0) }
        
        let inputArray = try! MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        
        for i in 0..<min(tokens.count, maxLength) {
            inputArray[i] = NSNumber(value: tokens[i])
        }
        
        return inputArray
    }
    
    private func convertToWAV(_ audioArray: MLMultiArray) -> Data {
        let sampleRate: Float = 16000
        let length = audioArray.count
        
        var wavData = Data()
        
        // WAVヘッダー
        let header = createWAVHeader(sampleRate: Int(sampleRate), samples: length)
        wavData.append(header)
        
        // 音声データ
        for i in 0..<length {
            let sample = audioArray[i].floatValue
            let int16Sample = Int16(sample * 32767)
            withUnsafeBytes(of: int16Sample) { bytes in
                wavData.append(contentsOf: bytes)
            }
        }
        
        return wavData
    }
    
    private func createWAVHeader(sampleRate: Int, samples: Int) -> Data {
        var header = Data()
        let dataSize = samples * 2  // 16-bit samples
        let fileSize = dataSize + 36
        
        header.append("RIFF".data(using: .ascii)!)
        header.append(Data(bytes: &fileSize, count: 4))
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        
        var fmtSize: Int32 = 16
        header.append(Data(bytes: &fmtSize, count: 4))
        
        var audioFormat: Int16 = 1
        header.append(Data(bytes: &audioFormat, count: 2))
        
        var numChannels: Int16 = 1
        header.append(Data(bytes: &numChannels, count: 2))
        
        var sampleRateData = Int32(sampleRate)
        header.append(Data(bytes: &sampleRateData, count: 4))
        
        var byteRate = Int32(sampleRate * 2)
        header.append(Data(bytes: &byteRate, count: 4))
        
        var blockAlign: Int16 = 2
        header.append(Data(bytes: &blockAlign, count: 2))
        
        var bitsPerSample: Int16 = 16
        header.append(Data(bytes: &bitsPerSample, count: 2))
        
        header.append("data".data(using: .ascii)!)
        var dataSizeData = Int32(dataSize)
        header.append(Data(bytes: &dataSizeData, count: 4))
        
        return header
    }
}

enum TTSError: Error {
    case modelNotFound
    case encodingFailed
    case decodingFailed
    case audioGenerationFailed
}

// 使用例
class ViewController: UIViewController {
    private let ttsPipeline = try! ParlerTTSANEPipeline()
    
    func generateSpeech() {
        Task {
            do {
                let audioData = try await ttsPipeline.synthesizeSpeech(from: "Hello, this is a test of the ANE-optimized TTS system.")
                
                // 音声再生
                try playAudio(data: audioData)
                
            } catch {
                print("TTS generation failed: \\(error)")
            }
        }
    }
    
    private func playAudio(data: Data) throws {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("tts_output.wav")
        try data.write(to: tempURL)
        
        let audioPlayer = try AVAudioPlayer(contentsOf: tempURL)
        audioPlayer.play()
    }
}
'''
    
    return swift_code

def run_phase3_experiment():
    """Phase 3実験の実行"""
    
    print("Phase 3: Full ANE Pipeline Experiment")
    print("=" * 50)
    
    try:
        # 1. パイプライン作成
        pipeline = FullANEPipeline()
        
        # 2. 各コンポーネントをCore MLに変換
        print("\n1. Converting components to Core ML...")
        text_encoder_path = pipeline.convert_text_encoder_to_coreml()
        tts_decoder_path = pipeline.convert_tts_decoder_to_coreml()
        audio_decoder_path = pipeline.convert_audio_decoder_to_coreml()
        
        # 3. パイプライン全体のベンチマーク
        print("\n2. Benchmarking full pipeline...")
        benchmark_results = pipeline.benchmark_full_pipeline(
            text_encoder_path, tts_decoder_path, audio_decoder_path
        )
        
        # 4. iOS統合コード生成
        print("\n3. Generating iOS integration code...")
        ios_code = create_ios_integration_code()
        with open("ParlerTTSANEPipeline.swift", "w") as f:
            f.write(ios_code)
        print("iOS integration code saved to ParlerTTSANEPipeline.swift")
        
        # 5. 結果保存
        results = {
            'phase': 3,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_paths': {
                'text_encoder': text_encoder_path,
                'tts_decoder': tts_decoder_path,
                'audio_decoder': audio_decoder_path
            },
            'benchmark_results': benchmark_results,
            'ios_integration': "ParlerTTSANEPipeline.swift"
        }
        
        with open('phase3_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Phase 3 completed successfully!")
        print(f"📊 Full pipeline time: {benchmark_results['pipeline_avg_ms']:.2f} ms")
        print(f"📱 iOS integration code ready: ParlerTTSANEPipeline.swift")
        
        return results
        
    except Exception as e:
        print(f"❌ Phase 3 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_phase3_experiment()
    
    if results:
        print("\n🎉 All phases completed successfully!")
        print("\n📋 Final Summary:")
        print(f"- Text Encoder: {results['model_paths']['text_encoder']}")
        print(f"- TTS Decoder: {results['model_paths']['tts_decoder']}")  
        print(f"- Audio Decoder: {results['model_paths']['audio_decoder']}")
        print(f"- Full pipeline time: {results['benchmark_results']['pipeline_avg_ms']:.2f} ms")
        
        print("\n🚀 Next Steps:")
        print("1. Copy generated .mlmodel files to your iOS project")
        print("2. Add ParlerTTSANEPipeline.swift to your Xcode project")  
        print("3. Test the full ANE-optimized TTS pipeline")
        print("4. Compare performance with original PyTorch implementation")
        
    else:
        print("\n❌ Phase 3 failed. Check error messages above.")
