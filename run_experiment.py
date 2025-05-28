#!/usr/bin/env python3
"""
Parler TTS ANE Experiment Runner
段階的にANE最適化を実行するメインスクリプト
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """必要な依存関係をチェック"""
    required_packages = [
        'torch',
        'coremltools', 
        'transformers',
        'matplotlib',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease run: chmod +x setup_environment.sh && ./setup_environment.sh")
        return False
    
    return True

def run_phase(phase_num, force=False):
    """指定されたPhaseを実行"""
    
    phase_configs = {
        1: {
            'name': 'Fixed Length Decoder (Minimal)',
            'script': 'phase1_fixed_length/minimal_decoder.py',
            'description': '最小限の依存関係でPyTorch固定長変換を実証'
        },
        2: {
            'name': 'ANE Decoder',
            'script': 'phase2_ane_decoder/ane_decoder.py', 
            'description': 'TTSデコーダーをANEで動作させる'
        },
        3: {
            'name': 'Full ANE Pipeline',
            'script': 'phase3_full_ane/full_ane_pipeline.py',
            'description': '全体をANEで動作させる'
        }
    }
    
    if phase_num not in phase_configs:
        print(f"❌ Invalid phase number: {phase_num}")
        return False
    
    config = phase_configs[phase_num]
    script_path = Path(config['script'])
    results_file = f"phase{phase_num}_results.json"
    
    print(f"\n🚀 Running Phase {phase_num}: {config['name']}")
    print(f"📝 {config['description']}")
    print("=" * 60)
    
    # 既存の結果をチェック
    if Path(results_file).exists() and not force:
        print(f"⚠️  Results file {results_file} already exists.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Skipped.")
            return True
    
    # スクリプト実行
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        print("📤 Script Output:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ Phase {phase_num} completed successfully!")
            
            # 結果ファイルの確認
            if Path(results_file).exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"📊 Results saved to {results_file}")
                return True
            else:
                print(f"⚠️  No results file generated")
                return True
        else:
            print(f"❌ Phase {phase_num} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to run Phase {phase_num}: {e}")
        return False

def analyze_results():
    """全Phaseの結果を分析"""
    
    print("\n📊 Analyzing Results from All Phases")
    print("=" * 50)
    
    results_summary = {}
    
    for phase_num in [1, 2, 3]:
        results_file = f"phase{phase_num}_results.json"
        
        if Path(results_file).exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                results_summary[f"Phase {phase_num}"] = results
                print(f"✅ Phase {phase_num}: Results loaded")
            except Exception as e:
                print(f"❌ Phase {phase_num}: Failed to load results - {e}")
        else:
            print(f"⚠️  Phase {phase_num}: No results file found")
    
    if not results_summary:
        print("No results to analyze.")
        return
    
    # パフォーマンス比較
    print("\n🏃 Performance Comparison:")
    
    if "Phase 1" in results_summary:
        phase1 = results_summary["Phase 1"]
        if 'fixed_length' in phase1:
            avg_time = phase1['fixed_length']['avg_time']
            print(f"  Phase 1 (Fixed Length): {avg_time:.3f}s")
    
    if "Phase 2" in results_summary:
        phase2 = results_summary["Phase 2"]
        if 'coreml_benchmark' in phase2:
            avg_time_ms = phase2['coreml_benchmark']['avg_time_ms']
            print(f"  Phase 2 (ANE Decoder): {avg_time_ms:.2f}ms")
    
    if "Phase 3" in results_summary:
        phase3 = results_summary["Phase 3"]
        if 'benchmark_results' in phase3:
            pipeline_time = phase3['benchmark_results']['pipeline_avg_ms']
            print(f"  Phase 3 (Full ANE): {pipeline_time:.2f}ms")
            
            # コンポーネント別時間
            if 'component_times_ms' in phase3['benchmark_results']:
                print("\n    Component breakdown:")
                for component, stats in phase3['benchmark_results']['component_times_ms'].items():
                    print(f"      {component}: {stats['avg']:.2f}ms")
    
    # ファイル生成確認
    print("\n📁 Generated Files:")
    generated_files = [
        "ane_text_encoder.mlmodel",
        "ane_tts_decoder.mlmodel", 
        "ane_tts_decoder_full.mlmodel",
        "ane_audio_decoder.mlmodel",
        "ParlerTTSANEPipeline.swift",
        "phase1_timing_comparison.png"
    ]
    
    for file_path in generated_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({file_size/1024/1024:.1f} MB)")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    # 総合結果をファイルに保存
    summary_file = "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n💾 Complete analysis saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Parler TTS ANE Experiment Runner")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], 
                       help='Run specific phase (1-3)')
    parser.add_argument('--all', action='store_true',
                       help='Run all phases sequentially')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze results from all phases')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run even if results exist')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("🧪 Parler TTS ANE Experiment Runner")
    print("=" * 50)
    
    # 依存関係チェック
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are satisfied")
        return
    
    if not check_dependencies():
        return
    
    # 結果分析のみ
    if args.analyze:
        analyze_results()
        return
    
    # 特定のPhase実行
    if args.phase:
        success = run_phase(args.phase, force=args.force)
        if success:
            print(f"\n🎉 Phase {args.phase} completed successfully!")
        return
    
    # 全Phase実行
    if args.all:
        print("🚀 Running all phases sequentially...")
        
        all_success = True
        for phase_num in [1, 2, 3]:
            success = run_phase(phase_num, force=args.force)
            if not success:
                print(f"❌ Phase {phase_num} failed. Stopping execution.")
                all_success = False
                break
        
        if all_success:
            print("\n🎉 All phases completed successfully!")
            analyze_results()
        
        return
    
    # デフォルト: ヘルプ表示
    parser.print_help()
    print("\n💡 Usage examples:")
    print("  python run_experiment.py --all          # Run all phases")
    print("  python run_experiment.py --phase 1      # Run Phase 1 only")
    print("  python run_experiment.py --analyze      # Analyze results")
    print("  python run_experiment.py --check-deps   # Check dependencies")

if __name__ == "__main__":
    main()
