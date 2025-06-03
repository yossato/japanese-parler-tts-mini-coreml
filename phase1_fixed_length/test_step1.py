#!/usr/bin/env python3
"""
Phase 1.1 Step 1 単体実行
修正版のメモリ効率デコーダーテスト
"""

import os
import sys

# ディレクトリを変更
phase1_dir = "/Users/yoshiaki/Projects/parler_tts_experiment/phase1_fixed_length"
os.chdir(phase1_dir)
sys.path.insert(0, phase1_dir)

print("🔸 Phase 1.1 Step 1: Memory Efficient Decoder (Fixed)")
print("=" * 55)

try:
    from memory_efficient_decoder import main
    result = main()
    
    if result:
        print("\n✅ Step 1 completed successfully!")
        print("📁 Results saved to: phase1_1_memory_efficient_results.json")
        
        # ファイル確認
        if os.path.exists('phase1_1_memory_efficient_results.json'):
            print("✅ JSON file created successfully")
        else:
            print("⚠️  JSON file not found, but no error occurred")
            
    else:
        print("\n❌ Step 1 failed!")
        
except Exception as e:
    print(f"\n❌ Step 1 execution failed: {e}")
    import traceback
    traceback.print_exc()
