"""
Simple training script that works cross-platform.
Run: python train_agents.py [--quick] [--neat-only] [--dqn-only]
"""

import argparse
import os
import sys

# Ensure correct imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.maze_env import MazeEnv
from neuroevolution.neat_solver import NEATMazeSolver, create_neat_config
from reinforcement_learning.dqn_solver import DQNMazeSolver

def main():
    parser = argparse.ArgumentParser(description='Train NEAT and/or DQN agents')
    parser.add_argument('--quick', action='store_true', help='Quick training mode (fewer epochs)')
    parser.add_argument('--neat-only', action='store_true', help='Train only NEAT')
    parser.add_argument('--dqn-only', action='store_true', help='Train only DQN')
    parser.add_argument('--neat-generations', type=int, default=50, help='NEAT generations')
    parser.add_argument('--dqn-episodes', type=int, default=500, help='DQN episodes')
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.neat_generations = 10
        args.dqn_episodes = 100
        print("🚀 Quick mode enabled: NEAT=10 generations, DQN=100 episodes\n")
    
    # Create directories
    os.makedirs('logs/neat', exist_ok=True)
    os.makedirs('logs/dqn', exist_ok=True)
    
    train_neat = not args.dqn_only
    train_dqn = not args.neat_only
    
    # Train NEAT
    if train_neat:
        print("="*70)
        print("TRAINING NEAT AGENT")
        print("="*70)
        print(f"Generations: {args.neat_generations}")
        print()
        
        try:
            # Create config in neuroevolution directory
            config_path = os.path.join('neuroevolution', 'config-neat.txt')
            config_path = create_neat_config(config_path)
            
            # Initialize and train
            neat_solver = NEATMazeSolver(config_path, log_dir='logs/neat')
            winner = neat_solver.train(generations=args.neat_generations)
            
            # Evaluate
            print("\n📊 Evaluating NEAT agent...")
            results = neat_solver.evaluate_best(render=False, num_episodes=10)
            
            # Statistics
            success_rate = sum(r['reached_goal'] for r in results) / len(results)
            avg_steps = sum(r['steps'] for r in results) / len(results)
            print(f"\n✅ NEAT Training Complete!")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Steps: {avg_steps:.1f}")
            
            # Visualize
            print("\n📈 Generating visualizations...")
            neat_solver.visualize_training()
            
        except Exception as e:
            print(f"\n❌ NEAT training failed: {e}")
            import traceback
            traceback.print_exc()
            if not train_dqn:
                return 1
    
    # Train DQN
    if train_dqn:
        print("\n" + "="*70)
        print("TRAINING DQN AGENT")
        print("="*70)
        print(f"Episodes: {args.dqn_episodes}")
        print()
        
        try:
            # Create environment and solver
            env = MazeEnv()
            dqn_solver = DQNMazeSolver(env, log_dir='logs/dqn')
            
            # Train
            dqn_solver.train(num_episodes=args.dqn_episodes, verbose=True)
            
            # Evaluate
            print("\n📊 Evaluating DQN agent...")
            results = dqn_solver.evaluate(num_episodes=10, render=False)
            
            # Statistics
            success_rate = sum(r['reached_goal'] for r in results) / len(results)
            avg_steps = sum(r['steps'] for r in results) / len(results)
            print(f"\n✅ DQN Training Complete!")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Avg Steps: {avg_steps:.1f}")
            
            # Visualize
            print("\n📈 Generating visualizations...")
            dqn_solver.visualize_training()
            
        except Exception as e:
            print(f"\n❌ DQN training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to:")
    if train_neat:
        print("   - NEAT: logs/neat/")
    if train_dqn:
        print("   - DQN: logs/dqn/")
    
    print("\n📊 Next steps:")
    print("   1. View training plots (check the figures that opened)")
    print("   2. Run comparison: python compare_agents.py")
    print("   3. Test robustness: python test_robustness.py")
    print("   4. Open dashboard: analysis/interactive_dashboard.html")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
