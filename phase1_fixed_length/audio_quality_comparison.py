#!/usr/bin/env python3
"""
Phase 1.1 Step 3: 音声品質比較実験
オリジナル vs 固定長の音声品質を人間が評価するための音声生成
"""

import sys
import os
sys.path.append('/Users/yoshiaki/Projects/parler-tts')

import torch
import json
import time
import numpy as np
import wave
from pathlib import Path
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# 前のステップで作成したクラスをインポート
try:
    from memory_efficient_decoder import MemoryEfficientTTSGenerator
    from dac_integration import RealAudioTTSGenerator
except ImportError as e:
    print(f"⚠️  Warning: Could not import required modules: {e}")
    print("    Please ensure previous steps completed successfully")
    sys.exit(1)

class AudioQualityComparison:
    """音声品質比較実験のためのクラス"""
    
    def __init__(self, max_length=100):
        print(f"🎵 Audio Quality Comparison Experiment")
        print(f"   Max length: {max_length}")
        
        self.max_length = max_length
        self.model_name = "parler-tts/parler-tts-mini-v1"
        
        # 実音声生成器を初期化
        print("📦 Loading Real Audio TTS Generator...")
        self.real_audio_generator = RealAudioTTSGenerator(
            model_name=self.model_name,
            max_length=max_length
        )
        
        # オリジナルモデルも準備（参考用）
        print("📦 Loading Original Model for Reference...")
        try:
            self.original_model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.original_available = True
            print("✅ Original model loaded successfully")
        except Exception as e:
            print(f"⚠️  Original model loading failed: {e}")
            self.original_available = False
        
        print("✅ Audio Quality Comparison ready!")
    
    def generate_original_audio(self, text, description, output_path):
        """オリジナルモデルでの音声生成（可能な場合）"""
        
        if not self.original_available:
            print("  ❌ Original model not available")
            return None
        
        try:
            print(f"  🔄 Generating with original model...")
            
            start_time = time.time()
            
            # テキスト入力の準備
            inputs = self.tokenizer(
                description,
                text,
                return_tensors="pt",
                truncation=True,
                max_length=50
            )
            
            # オリジナル生成
            with torch.no_grad():
                generation = self.original_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=self.max_length,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            end_time = time.time()
            
            # 音声コードからDAC使用で音声生成
            # 注意: これは簡略化された処理です
            audio_codes = generation.view(1, 9, -1)  # [batch_size, num_codebooks, length]
            
            try:
                # DACで音声復元を試行
                audio_waveform = self.real_audio_generator.audio_encoder.decode(audio_codes)
                if isinstance(audio_waveform, torch.Tensor):
                    audio_np = audio_waveform.squeeze().cpu().numpy()
                else:
                    audio_np = np.array(audio_waveform)
                
                # 音声ファイル保存
                self._save_audio_file(audio_np, output_path, self.real_audio_generator.dac_config.sampling_rate)
                
                return {
                    'success': True,
                    'generation_time': end_time - start_time,
                    'output_path': output_path,
                    'audio_shape': audio_np.shape,
                    'method': 'original'
                }
                
            except Exception as e:
                print(f"    ⚠️  DAC decoding failed for original: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'generation_time': end_time - start_time,
                    'method': 'original'
                }
            
        except Exception as e:
            print(f"  ❌ Original generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'original'
            }
    
    def generate_fixed_length_audio(self, text, description, output_path):
        """固定長モデルでの音声生成"""
        
        try:
            print(f"  ⚡ Generating with fixed length model...")
            
            result = self.real_audio_generator.text_to_audio(
                text=text,
                output_dir=os.path.dirname(output_path),
                description=description
            )
            
            # ファイルを指定パスに移動
            if os.path.exists(result['output_file']):
                if result['output_file'] != output_path:
                    os.rename(result['output_file'], output_path)
                
                return {
                    'success': True,
                    'generation_time': result['total_time'],
                    'output_path': output_path,
                    'duration': result['duration'],
                    'is_mock': result.get('is_mock', False),
                    'method': 'fixed_length'
                }
            else:
                return {
                    'success': False,
                    'error': 'Output file not found',
                    'method': 'fixed_length'
                }
                
        except Exception as e:
            print(f"  ❌ Fixed length generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'fixed_length'
            }
    
    def _save_audio_file(self, audio_np, output_path, sample_rate):
        """音声ファイルを保存"""
        
        # 正規化
        audio_normalized = audio_np / np.max(np.abs(audio_np)) * 0.8
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # ディレクトリ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # WAVファイル保存
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # モノラル
            wav_file.setsampwidth(2)  # 16bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def run_comparison_experiment(self, test_cases, output_dir="audio_quality_comparison"):
        """比較実験を実行"""
        
        print(f"\n🎼 Running Audio Quality Comparison Experiment")
        print(f"   Test cases: {len(test_cases)}")
        print(f"   Output directory: {output_dir}")
        
        # 出力ディレクトリ準備
        Path(output_dir).mkdir(exist_ok=True)
        Path(f"{output_dir}/original").mkdir(exist_ok=True)
        Path(f"{output_dir}/fixed_length").mkdir(exist_ok=True)
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
            print(f"Text: \"{test_case['text']}\"")
            print(f"Description: {test_case['description'][:50]}...")
            
            # ファイル名生成
            safe_text = "".join(c for c in test_case['text'] if c.isalnum() or c in " -_").strip()
            safe_text = safe_text.replace(" ", "_")[:20]
            
            original_path = f"{output_dir}/original/{safe_text}_original.wav"
            fixed_path = f"{output_dir}/fixed_length/{safe_text}_fixed.wav"
            
            test_result = {
                'test_case': test_case,
                'original_result': None,
                'fixed_result': None
            }
            
            # 1. オリジナル生成
            if self.original_available:
                print("1. Original generation:")
                test_result['original_result'] = self.generate_original_audio(
                    test_case['text'], 
                    test_case['description'], 
                    original_path
                )
            else:
                print("1. Original generation: SKIPPED (not available)")
                test_result['original_result'] = {
                    'success': False,
                    'error': 'Original model not available',
                    'method': 'original'
                }
            
            # 2. 固定長生成
            print("2. Fixed length generation:")
            test_result['fixed_result'] = self.generate_fixed_length_audio(
                test_case['text'], 
                test_case['description'], 
                fixed_path
            )
            
            results.append(test_result)
            
            # 結果サマリー
            print("   Results:")
            if test_result['original_result'] and test_result['original_result']['success']:
                print(f"     ✅ Original: {test_result['original_result']['generation_time']:.3f}s")
            else:
                error_msg = test_result['original_result'].get('error', 'Failed') if test_result['original_result'] else 'Not available'
                print(f"     ❌ Original: {error_msg}")
            
            if test_result['fixed_result']['success']:
                print(f"     ✅ Fixed: {test_result['fixed_result']['generation_time']:.3f}s")
                if test_result['fixed_result'].get('is_mock'):
                    print(f"        ⚠️  (Using mock audio)")
            else:
                print(f"     ❌ Fixed: {test_result['fixed_result'].get('error', 'Failed')}")
        
        # 実験結果の保存
        experiment_summary = {
            'test_cases': test_cases,
            'results': results,
            'config': {
                'max_length': self.max_length,
                'model_name': self.model_name,
                'original_available': self.original_available
            },
            'statistics': self._calculate_statistics(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(f'{output_dir}/comparison_results.json', 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        
        # 聴取テスト用の案内を生成
        self._generate_listening_guide(output_dir, results)
        
        print(f"\n✅ Comparison experiment completed!")
        print(f"   Results saved to: {output_dir}/comparison_results.json")
        print(f"   Listening guide: {output_dir}/listening_guide.txt")
        
        return experiment_summary
    
    def _calculate_statistics(self, results):
        """実験結果の統計を計算"""
        
        original_successes = sum(1 for r in results if r['original_result'] and r['original_result']['success'])
        fixed_successes = sum(1 for r in results if r['fixed_result']['success'])
        
        original_times = [r['original_result']['generation_time'] 
                         for r in results if r['original_result'] and r['original_result']['success']]
        fixed_times = [r['fixed_result']['generation_time'] 
                      for r in results if r['fixed_result']['success']]
        
        speedup = None
        if original_times and fixed_times:
            avg_original_time = np.mean(original_times)
            avg_fixed_time = np.mean(fixed_times)
            speedup = avg_original_time / avg_fixed_time if avg_fixed_time > 0 else None
        
        return {
            'total_tests': len(results),
            'original_success_rate': original_successes / len(results),
            'fixed_success_rate': fixed_successes / len(results),
            'original_avg_time': np.mean(original_times) if original_times else None,
            'fixed_avg_time': np.mean(fixed_times) if fixed_times else None,
            'speedup': speedup
        }
    
    def _generate_listening_guide(self, output_dir, results):
        """聴取テスト用のガイドを生成"""
        
        guide_content = []
        guide_content.append("Audio Quality Comparison - Listening Guide")
        guide_content.append("=" * 50)
        guide_content.append("")
        guide_content.append("Instructions:")
        guide_content.append("1. Listen to each pair of audio files")
        guide_content.append("2. Compare the quality between Original and Fixed Length versions")
        guide_content.append("3. Rate each aspect on a scale of 1-5:")
        guide_content.append("   - Naturalness (sounds like human speech)")
        guide_content.append("   - Clarity (words are clear and understandable)")
        guide_content.append("   - Fluency (speech flows naturally)")
        guide_content.append("   - Overall Quality")
        guide_content.append("")
        guide_content.append("Test Cases:")
        guide_content.append("-" * 30)
        
        for i, result in enumerate(results):
            test_case = result['test_case']
            guide_content.append(f"\n{i+1}. Text: \"{test_case['text']}\"")
            guide_content.append(f"   Description: {test_case['description']}")
            
            # ファイルパス
            safe_text = "".join(c for c in test_case['text'] if c.isalnum() or c in " -_").strip()
            safe_text = safe_text.replace(" ", "_")[:20]
            
            if result['original_result'] and result['original_result']['success']:
                guide_content.append(f"   Original: original/{safe_text}_original.wav")
            else:
                guide_content.append(f"   Original: [Generation failed]")
            
            if result['fixed_result']['success']:
                guide_content.append(f"   Fixed Length: fixed_length/{safe_text}_fixed.wav")
                if result['fixed_result'].get('is_mock'):
                    guide_content.append(f"   Note: Fixed Length uses mock audio (DAC failed)")
            else:
                guide_content.append(f"   Fixed Length: [Generation failed]")
            
            guide_content.append("")
            guide_content.append("   Rating (1-5):")
            guide_content.append("   Original    - Naturalness: ___ Clarity: ___ Fluency: ___ Overall: ___")
            guide_content.append("   Fixed Length - Naturalness: ___ Clarity: ___ Fluency: ___ Overall: ___")
            guide_content.append("   Which is better? [Original/Fixed Length/Similar]")
            guide_content.append("   Comments: ________________________")
        
        guide_content.append("\n")
        guide_content.append("How to Play Audio Files:")
        guide_content.append("- macOS: open [filename].wav")
        guide_content.append("- macOS Terminal: afplay [filename].wav")
        guide_content.append("- Or use any audio player")
        
        # ガイドファイル保存
        with open(f'{output_dir}/listening_guide.txt', 'w') as f:
            f.write('\n'.join(guide_content))


def create_test_cases():
    """テストケースを作成"""
    
    return [
        {
            'text': 'Hello',
            'description': 'A female speaker with a clear voice.'
        },
        {
            'text': 'Hello world',
            'description': 'A male speaker with a deep voice.'
        },
        {
            'text': 'This is a test',
            'description': 'A young female speaker with an expressive voice.'
        },
        {
            'text': 'Good morning',
            'description': 'A cheerful speaker with a bright tone.'
        },
        {
            'text': 'How are you today',
            'description': 'A calm speaker with a steady pace.'
        }
    ]


def main():
    print("Phase 1.1 Step 3: Audio Quality Comparison Experiment")
    print("=" * 60)
    
    try:
        # テストケース準備
        test_cases = create_test_cases()
        
        print(f"📋 Test Cases:")
        for i, case in enumerate(test_cases):
            print(f"   {i+1}. \"{case['text']}\" - {case['description']}")
        
        # 比較実験の実行
        comparison = AudioQualityComparison(max_length=100)
        experiment_results = comparison.run_comparison_experiment(test_cases)
        
        # 結果サマリー
        stats = experiment_results['statistics']
        print(f"\n📊 Experiment Statistics:")
        print(f"   Total tests: {stats['total_tests']}")
        print(f"   Original success rate: {stats['original_success_rate']:.1%}")
        print(f"   Fixed length success rate: {stats['fixed_success_rate']:.1%}")
        
        if stats['original_avg_time'] and stats['fixed_avg_time']:
            print(f"   Average generation time:")
            print(f"     Original: {stats['original_avg_time']:.3f}s")
            print(f"     Fixed Length: {stats['fixed_avg_time']:.3f}s")
            if stats['speedup']:
                print(f"     Speedup: {stats['speedup']:.1f}x")
        
        print(f"\n🎧 Next Steps:")
        print(f"   1. Check the audio files in: audio_quality_comparison/")
        print(f"   2. Follow the listening guide: audio_quality_comparison/listening_guide.txt")
        print(f"   3. Listen to both original and fixed length versions")
        print(f"   4. Evaluate the quality differences")
        print(f"   5. Decide if fixed length approach is viable for Phase 2")
        
        return experiment_results
        
    except Exception as e:
        print(f"❌ Audio quality comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
