"""
Test robustness of trained agents.
Run: python test_robustness.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_model_files():
    """Find trained model files."""
    # Find NEAT model
    neat_models = glob.glob('logs/neat/best_genome*.pkl') or \
                  glob.glob('logs/neat/winner*.pkl')
    neat_model = neat_models[0] if neat_models else None
    
    # Find DQN model
    dqn_models = glob.glob('logs/dqn/best_model.pth') or \
                 glob.glob('logs/dqn/final_model*.pth')
    dqn_model = dqn_models[0] if dqn_models else None
    
    # Find NEAT config
    neat_config = 'neuroevolution/config-neat.txt'
    if not os.path.exists(neat_config):
        neat_config = None
    
    return neat_model, dqn_model, neat_config

def main():
    print("="*70)
    print("ROBUSTNESS TESTING SUITE")
    print("="*70)
    print()
    
    # Find models
    neat_model, dqn_model, neat_config = find_model_files()
    
    if not neat_model and not dqn_model:
        print("❌ No trained models found!")
        print("   Please run: python train_agents.py")
        return 1
    
    print("📂 Found models:")
    if neat_model:
        print(f"   ✓ NEAT: {neat_model}")
    else:
        print("   ✗ NEAT: Not found")
    
    if dqn_model:
        print(f"   ✓ DQN: {dqn_model}")
    else:
        print("   ✗ DQN: Not found")
    
    if not neat_config:
        print("   ⚠️  NEAT config not found, will skip NEAT tests")
    
    print()
    
    try:
        from analysis.robustness_tests import RobustnessTestSuite
        
        print("🔬 Initializing test suite...")
        suite = RobustnessTestSuite(
            neat_model_path=neat_model,
            dqn_model_path=dqn_model,
            neat_config_path=neat_config
        )
        
        print("\n" + "-"*70)
        print("Running Tests (this may take 10-15 minutes)")
        print("-"*70)
        
        # Run all tests
        suite.run_all_tests()
        
        print("\n✅ All robustness tests completed!")
        print("\n📁 Results saved to:")
        print("   - analysis/noise_sensitivity.png")
        print("   - analysis/generalization.png")
        print("   - analysis/failure_modes.png")
        print("   - analysis/robustness_scores.png")
        print("   - analysis/robustness_test_results.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
