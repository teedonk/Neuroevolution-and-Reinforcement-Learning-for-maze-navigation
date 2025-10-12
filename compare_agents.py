"""
Generate comparison visualizations between NEAT and DQN.
Run: python compare_agents.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)
    print()
    
    # Check if training logs exist
    neat_exists = os.path.exists('logs/neat')
    dqn_exists = os.path.exists('logs/dqn')
    
    if not neat_exists and not dqn_exists:
        print("❌ No training logs found!")
        print("   Please run: python train_agents.py")
        return 1
    
    if not neat_exists:
        print("⚠️  NEAT logs not found. Only DQN results will be shown.")
    if not dqn_exists:
        print("⚠️  DQN logs not found. Only NEAT results will be shown.")
    
    print()
    
    try:
        from analysis.visualize_training import TrainingVisualizer
        
        print("📊 Creating comparison dashboard...")
        visualizer = TrainingVisualizer(
            neat_log_dir='logs/neat',
            dqn_log_dir='logs/dqn'
        )
        
        # Generate all visualizations
        print("   ✓ Comparison dashboard")
        visualizer.create_comparison_dashboard()
        
        print("   ✓ Decision boundaries")
        visualizer.visualize_decision_boundaries()
        
        print("   ✓ Live comparison")
        visualizer.create_live_comparison()
        
        print("   ✓ Adaptation animation")
        visualizer.create_adaptation_animation(save_path='analysis/adaptation.gif')
        
        print("\n✅ All visualizations generated!")
        print("\n📁 Outputs saved to: analysis/")
        print("   - comparison_dashboard.png")
        print("   - decision_boundaries.png")
        print("   - live_comparison.png")
        print("   - adaptation.gif")
        print("\n💡 Tip: Open analysis/interactive_dashboard.html for interactive view!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
