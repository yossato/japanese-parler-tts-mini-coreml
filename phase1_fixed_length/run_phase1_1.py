#!/usr/bin/env python3
"""
Phase 1.1 統合実行スクリプト
メモリ効率化 → DAC統合 → 音声品質比較の全ステップを実行
"""

import os
import sys
import json
import time
from pathlib import Path

def print_header(title):
    """ヘッダーを印刷"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """ステップヘッダーを印刷"""
    print(f"\n🔸 Step {step_num}: {description}")
    print(f"{'-'*40}")

def run_step_1():
    """Step 1: メモリ効率的固定長デコーダー"""
    print_step(1, "Memory Efficient Fixed Length Decoder")
    
    try:
        from memory_efficient_decoder import main as memory_efficient_main
        result = memory_efficient_main()
        
        if result:
            print("✅ Step 1 completed successfully!")
            return True
        else:
            print("❌ Step 1 failed!")
            return False
            
    except Exception as e:
        print(f"❌ Step 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_step_2():
    """Step 2: DAC統合による実音声生成"""
    print_step(2, "DAC Integration for Real Audio Generation")
    
    try:
        from dac_integration import main as dac_integration_main
        result = dac_integration_main()
        
        if result:
            print("✅ Step 2 completed successfully!")
            return True
        else:
            print("❌ Step 2 failed!")
            return False
            
    except Exception as e:
        print(f"❌ Step 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_step_3():
    """Step 3: 音声品質比較実験"""
    print_step(3, "Audio Quality Comparison Experiment")
    
    try:
        from audio_quality_comparison import main as audio_comparison_main
        result = audio_comparison_main()
        
        if result:
            print("✅ Step 3 completed successfully!")
            return True
        else:
            print("❌ Step 3 failed!")
            return False
            
    except Exception as e:
        print(f"❌ Step 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_results():
    """全ステップの結果を収集・統合"""
    print_step("Final", "Collecting and Integrating All Results")
    
    results = {
        'phase1_1_summary': {},
        'step_results': {},
        'files_generated': [],
        'recommendations': []
    }
    
    # Step 1の結果
    try:
        if os.path.exists('phase1_1_memory_efficient_results.json'):
            with open('phase1_1_memory_efficient_results.json', 'r') as f:
                results['step_results']['step1'] = json.load(f)
            results['files_generated'].append('phase1_1_memory_efficient_results.json')
            print("📊 Step 1 results collected")
    except Exception as e:
        print(f"⚠️  Could not collect Step 1 results: {e}")
    
    # Step 2の結果
    try:
        if os.path.exists('phase1_1_audio_comparison.json'):
            with open('phase1_1_audio_comparison.json', 'r') as f:
                results['step_results']['step2'] = json.load(f)
            results['files_generated'].append('phase1_1_audio_comparison.json')
            print("📊 Step 2 results collected")
    except Exception as e:
        print(f"⚠️  Could not collect Step 2 results: {e}")
    
    # Step 3の結果
    try:
        if os.path.exists('audio_quality_comparison/comparison_results.json'):
            with open('audio_quality_comparison/comparison_results.json', 'r') as f:
                results['step_results']['step3'] = json.load(f)
            results['files_generated'].append('audio_quality_comparison/comparison_results.json')
            print("📊 Step 3 results collected")
    except Exception as e:
        print(f"⚠️  Could not collect Step 3 results: {e}")
    
    # 生成されたファイルの確認
    audio_dirs = ['comparison_audio', 'generated_real_audio', 'audio_quality_comparison']
    for dir_name in audio_dirs:
        if os.path.exists(dir_name):
            audio_files = [f for f in os.listdir(dir_name) if f.endswith('.wav')]
            if audio_files:
                results['files_generated'].extend([f"{dir_name}/{f}" for f in audio_files])
    
    # 推奨事項の生成
    results['recommendations'] = generate_recommendations(results['step_results'])
    
    # 統合結果の保存
    results['phase1_1_summary'] = {
        'total_steps': 3,
        'completed_steps': len([k for k in results['step_results'].keys()]),
        'success_rate': len([k for k in results['step_results'].keys()]) / 3,
        'audio_files_generated': len([f for f in results['files_generated'] if f.endswith('.wav')]),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('phase1_1_integrated_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Integrated results saved to: phase1_1_integrated_results.json")
    
    return results

def generate_recommendations(step_results):
    """実験結果に基づく推奨事項を生成"""
    
    recommendations = []
    
    # Step 1の結果分析
    if 'step1' in step_results:
        step1 = step_results['step1']
        if 'memory_profile' in step1:
            memory_overhead = step1['memory_profile'].get('model_overhead', 0)
            if memory_overhead > 1000:  # 1GB以上
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'description': f'Model memory overhead is {memory_overhead:.1f}MB. Consider further optimization.',
                    'action': 'Implement quantization or model pruning'
                })
            else:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'low',
                    'description': f'Memory usage is reasonable at {memory_overhead:.1f}MB.',
                    'action': 'Proceed to Phase 2'
                })
    
    # Step 3の結果分析
    if 'step3' in step_results:
        step3 = step_results['step3']
        if 'statistics' in step3:
            stats = step3['statistics']
            fixed_success_rate = stats.get('fixed_success_rate', 0)
            
            if fixed_success_rate >= 0.8:  # 80%以上成功
                recommendations.append({
                    'type': 'quality_assessment',
                    'priority': 'medium',
                    'description': f'Fixed length approach shows {fixed_success_rate:.1%} success rate.',
                    'action': 'Proceed with human listening test to evaluate quality'
                })
            else:
                recommendations.append({
                    'type': 'quality_assessment',
                    'priority': 'high',
                    'description': f'Fixed length approach has low success rate ({fixed_success_rate:.1%}).',
                    'action': 'Debug and improve fixed length implementation'
                })
            
            # スピードアップの評価
            speedup = stats.get('speedup')
            if speedup and speedup > 2:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'positive',
                    'description': f'Achieved {speedup:.1f}x speedup with fixed length approach.',
                    'action': 'This supports the approach for ANE optimization'
                })
    
    # 全体的な推奨事項
    if len(step_results) == 3:  # 全ステップ完了
        recommendations.append({
            'type': 'next_phase',
            'priority': 'medium',
            'description': 'Phase 1.1 completed successfully.',
            'action': 'Review audio quality and proceed to Phase 2 (ANE optimization) if satisfactory'
        })
    
    return recommendations

def print_final_summary(results):
    """最終サマリーを表示"""
    print_header("Phase 1.1 Final Summary")
    
    summary = results['phase1_1_summary']
    
    print(f"📊 Execution Summary:")
    print(f"   Completed steps: {summary['completed_steps']}/3")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Audio files generated: {summary['audio_files_generated']}")
    
    print(f"\n📁 Generated Files:")
    for file_path in results['files_generated'][:10]:  # 最初の10個を表示
        print(f"   {file_path}")
    if len(results['files_generated']) > 10:
        print(f"   ... and {len(results['files_generated']) - 10} more files")
    
    print(f"\n💡 Key Recommendations:")
    for i, rec in enumerate(results['recommendations'][:5], 1):  # 最初の5個
        priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'positive': '✅'}
        emoji = priority_emoji.get(rec['priority'], '📌')
        print(f"   {i}. {emoji} {rec['description']}")
        print(f"      Action: {rec['action']}")
    
    print(f"\n🎧 Manual Evaluation Required:")
    print(f"   1. Listen to audio files in: audio_quality_comparison/")
    print(f"   2. Follow listening guide: audio_quality_comparison/listening_guide.txt")
    print(f"   3. Compare original vs fixed length audio quality")
    print(f"   4. Decide whether to proceed to Phase 2 (ANE optimization)")
    
    print(f"\n🚀 Next Steps:")
    print(f"   If audio quality is acceptable:")
    print(f"   → python run_experiment.py --phase 2")
    print(f"   ")
    print(f"   If audio quality needs improvement:")
    print(f"   → Adjust max_length parameter and repeat Phase 1.1")
    print(f"   → Consider alternative approaches (quantization, etc.)")

def main():
    print_header("Phase 1.1: Memory Efficient Fixed Length TTS with Audio Generation")
    
    print(f"🎯 Objectives:")
    print(f"   1. Create memory-efficient fixed length decoder (vs 20.8GB original)")
    print(f"   2. Integrate with DAC for real audio generation")
    print(f"   3. Compare audio quality: Original vs Fixed Length")
    print(f"   4. Enable human evaluation of approach viability")
    
    start_time = time.time()
    success_count = 0
    
    # Step 1: メモリ効率化
    if run_step_1():
        success_count += 1
    
    # Step 2: DAC統合
    if run_step_2():
        success_count += 1
    
    # Step 3: 音声品質比較
    if run_step_3():
        success_count += 1
    
    # 結果統合
    print_step("Integration", "Collecting All Results")
    results = collect_results()
    
    # 最終サマリー
    total_time = time.time() - start_time
    print_final_summary(results)
    
    print_header("Execution Complete")
    print(f"⏱️  Total execution time: {total_time:.1f} seconds")
    print(f"✅ Successful steps: {success_count}/3")
    
    if success_count == 3:
        print(f"🎉 Phase 1.1 completed successfully!")
        print(f"🎧 Please evaluate the generated audio files and proceed accordingly.")
        return True
    else:
        print(f"⚠️  Phase 1.1 completed with {3-success_count} failed steps.")
        print(f"🔧 Check the error messages above and retry failed components.")
        return False

if __name__ == "__main__":
    # 現在のディレクトリをphase1_fixed_lengthに設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # パスを追加
    sys.path.insert(0, script_dir)
    
    success = main()
    sys.exit(0 if success else 1)
