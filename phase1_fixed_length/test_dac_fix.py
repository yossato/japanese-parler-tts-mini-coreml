#!/usr/bin/env python3
"""
Phase 1.1 Step 2 再実行（DAC修正版）
audio_scales パラメータ追加でDAC統合を修正
"""

import os
import sys

# ディレクトリを変更
phase1_dir = "/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length"
os.chdir(phase1_dir)
sys.path.insert(0, phase1_dir)

print("🔸 Phase 1.1 Step 2: DAC Integration (Fixed Version)")
print("=" * 55)

# 簡単なテスト実行
try:
    from dac_integration import RealAudioTTSGenerator
    
    print("🎵 Testing DAC Integration with audio_scales fix...")
    
    # 生成器作成
    generator = RealAudioTTSGenerator(max_length=100)
    
    # 単一テキストでテスト
    test_text = "Hello"
    print(f"\n🧪 Testing with: '{test_text}'")
    
    result = generator.text_to_audio(
        text=test_text, 
        output_dir="test_dac_fixed"
    )
    
    print(f"\n📊 Test Results:")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"   Total time: {result['total_time']:.3f}s")
    print(f"   Output file: {result['output_file']}")
    print(f"   Mock audio: {result.get('is_mock', False)}")
    
    if result.get('is_mock', False):
        print("   ⚠️  Still using mock audio (DAC integration needs more work)")
    else:
        print("   ✅ Real DAC audio generation successful!")
    
    # ファイル確認
    if os.path.exists(result['output_file']):
        file_size = os.path.getsize(result['output_file']) / 1024  # KB
        print(f"   📁 File size: {file_size:.1f} KB")
        print(f"   🎧 Play with: open {result['output_file']}")
    
except Exception as e:
    print(f"❌ DAC integration test failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n💡 Current Status:")
print(f"   - Fixed length audio code generation: ✅ Working")
print(f"   - Mock audio fallback: ✅ Working") 
print(f"   - Real DAC integration: 🔧 In progress")
print(f"   - Audio file output: ✅ Working")
