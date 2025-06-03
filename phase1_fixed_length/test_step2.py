#!/usr/bin/env python3
"""
Phase 1.1 Step 2 実行
DAC統合による実音声生成テスト
"""

import os
import sys

# ディレクトリを変更
phase1_dir = "/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length"
os.chdir(phase1_dir)
sys.path.insert(0, phase1_dir)

print("🔸 Phase 1.1 Step 2: DAC Integration for Real Audio")
print("=" * 55)

try:
    from dac_integration import main
    result = main()
    
    if result:
        print("\n✅ Step 2 completed successfully!")
        print("🎵 Audio files should be generated in comparison_audio/")
        
        # 生成されたファイルを確認
        audio_dirs = ['comparison_audio', 'generated_real_audio']
        for dir_name in audio_dirs:
            if os.path.exists(dir_name):
                wav_files = [f for f in os.listdir(dir_name) if f.endswith('.wav')]
                if wav_files:
                    print(f"📁 Found {len(wav_files)} audio files in {dir_name}/")
                    for f in wav_files[:3]:  # 最初の3個を表示
                        print(f"   - {f}")
                    if len(wav_files) > 3:
                        print(f"   ... and {len(wav_files) - 3} more files")
            
    else:
        print("\n❌ Step 2 failed!")
        
except Exception as e:
    print(f"\n❌ Step 2 execution failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎧 Next: Listen to the generated audio files!")
print(f"   Example: open comparison_audio/fixed_length/*.wav")
