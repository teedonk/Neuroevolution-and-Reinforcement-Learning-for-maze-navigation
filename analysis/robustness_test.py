import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import neat
import pickle
import json
import os
from typing import Dict, List, Tuple
import sys
sys.path.append('..')
from env.maze_env import MazeEnv

class RobustnessTestSuite:
    """
    Comprehensive robustness testing for both NEAT and DQN agents.
    Tests stability, generalization, and failure modes.
    """
    
    def __init__(self, neat_model_path: str = None, dqn_model_path: str = None,
                 neat_config_path: str = 'neuroevolution/config-neat.txt'):
        self.env = MazeEnv()
        self.neat_model_path = neat_model_path
        self.dqn_model_path = dqn_model_path
        self.neat_config_path = neat_config_path
        
        # Load models
        self.neat_genome = None
        self.neat_net = None
        self.dqn_model = None
        
        if neat_model_path and os.path.exists(neat_model_path):
            self.load_neat_model()
        
        if dqn_model_path and os.path.exists(dqn_model_path):
            self.load_dqn_model()
        
        # Results storage
        self.results = {
            'noise_sensitivity': {},
            'generalization': {},
            'failure_modes': {},
            'robustness_scores': {}
        }
    
    def load_neat_model(self):
        """Load trained NEAT model."""
        with open(self.neat_model_path, 'rb') as f:
            self.neat_genome = pickle.load(f)
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.neat_config_path)
        
        self.neat_net = neat.nn.FeedForwardNetwork.create(self.neat_genome, config)
        print(f"NEAT model loaded from {self.neat_model_path}")
    
    def load_dqn_model(self):
        """Load trained DQN model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import DQN class
        from reinforcement_learning.dqn_solver import DQN
        
        checkpoint = torch.load(self.dqn_model_path, map_location=device)
        
        input_dim = 12
        output_dim = 4
        self.dqn_model = DQN(input_dim, output_dim).to(device)
        self.dqn_model.load_state_dict(checkpoint['policy_net'])
        self.dqn_model.eval()
        
        print(f"DQN model loaded from {self.dqn_model_path}")
    
    def test_noise_sensitivity(self, noise_levels: List[float] = None):
        """
        Test how agents perform with noisy observations.
        Critical for understanding robustness.
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
        
        print("\n" + "="*60)
        print("NOISE SENSITIVITY TEST")
        print("="*60)
        
        neat_results = []
        dqn_results = []
        
        for noise in noise_levels:
            print(f"\nTesting with noise level: {noise}")
            
            # Test NEAT
            if self.neat_net:
                neat_perf = self._evaluate_with_noise(self.neat_net, noise, agent_type='neat')
                neat_results.append(neat_perf)
                print(f"  NEAT - Success: {neat_perf['success_rate']:.2%}, "
                      f"Avg Steps: {neat_perf['avg_steps']:.1f}")
            
            # Test DQN
            if self.dqn_model:
                dqn_perf = self._evaluate_with_noise(self.dqn_model, noise, agent_type='dqn')
                dqn_results.append(dqn_perf)
                print(f"  DQN  - Success: {dqn_perf['success_rate']:.2%}, "
                      f"Avg Steps: {dqn_perf['avg_steps']:.1f}")
        
        self.results['noise_sensitivity'] = {
            'noise_levels': noise_levels,
            'neat': neat_results,
            'dqn': dqn_results
        }
        
        # Visualize
        self._plot_noise_sensitivity()
        
        return self.results['noise_sensitivity']
    
    def _evaluate_with_noise(self, model, noise_level: float, agent_type: str, 
                            num_episodes: int = 20) -> Dict:
        """Evaluate model with noisy observations."""
        successes = 0
        steps_list = []
        rewards_list = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            while not done and steps < 500:
                # Add noise to observation
                noisy_obs = obs + np.random.normal(0, noise_level, obs.shape)
                noisy_obs = np.clip(noisy_obs, 0, 1)
                
                # Get action
                if agent_type == 'neat':
                    output = model.activate(noisy_obs)
                    action = np.argmax(output)
                else:  # DQN
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(noisy_obs).unsqueeze(0)
                        q_values = model(obs_tensor)
                        action = q_values.argmax(dim=1).item()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            # Check success
            reached_goal = (terminated and 
                          self.env.maze[int(self.env.agent_pos[0]), 
                          int(self.env.agent_pos[1])] == self.env.GOAL)
            
            if reached_goal:
                successes += 1
            
            steps_list.append(steps)
            rewards_list.append(total_reward)
        
        return {
            'success_rate': successes / num_episodes,
            'avg_steps': np.mean(steps_list),
            'std_steps': np.std(steps_list),
            'avg_reward': np.mean(rewards_list),
            'std_reward': np.std(rewards_list)
        }
    
    def _plot_noise_sensitivity(self):
        """Plot noise sensitivity results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        noise_levels = self.results['noise_sensitivity']['noise_levels']
        neat_results = self.results['noise_sensitivity']['neat']
        dqn_results = self.results['noise_sensitivity']['dqn']
        
        # Success rate vs noise
        ax = axes[0]
        if neat_results:
            neat_success = [r['success_rate'] for r in neat_results]
            ax.plot(noise_levels, neat_success, 'b-o', linewidth=2, 
                   markersize=8, label='NEAT')
        
        if dqn_results:
            dqn_success = [r['success_rate'] for r in dqn_results]
            ax.plot(noise_levels, dqn_success, 'r-s', linewidth=2, 
                   markersize=8, label='DQN')
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Robustness to Observation Noise', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Steps vs noise
        ax = axes[1]
        if neat_results:
            neat_steps = [r['avg_steps'] for r in neat_results]
            neat_std = [r['std_steps'] for r in neat_results]
            ax.errorbar(noise_levels, neat_steps, yerr=neat_std, fmt='b-o', 
                       linewidth=2, markersize=8, capsize=5, label='NEAT')
        
        if dqn_results:
            dqn_steps = [r['avg_steps'] for r in dqn_results]
            dqn_std = [r['std_steps'] for r in dqn_results]
            ax.errorbar(noise_levels, dqn_steps, yerr=dqn_std, fmt='r-s', 
                       linewidth=2, markersize=8, capsize=5, label='DQN')
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Average Steps', fontsize=12)
        ax.set_title('Efficiency Under Noise', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/noise_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_generalization(self, num_test_mazes: int = 10):
        """
        Test generalization to new maze configurations.
        """
        print("\n" + "="*60)
        print("GENERALIZATION TEST")
        print("="*60)
        
        neat_results = []
        dqn_results = []
        
        for i in range(num_test_mazes):
            print(f"\nTesting on maze {i+1}/{num_test_mazes}")
            
            # Create random maze
            test_maze = self._generate_random_maze()
            test_env = MazeEnv(maze_layout=test_maze)
            
            # Test NEAT
            if self.neat_net:
                neat_perf = self._evaluate_on_env(self.neat_net, test_env, agent_type='neat')
                neat_results.append(neat_perf)
                print(f"  NEAT - Success: {neat_perf['success']}, Steps: {neat_perf['steps']}")
            
            # Test DQN
            if self.dqn_model:
                dqn_perf = self._evaluate_on_env(self.dqn_model, test_env, agent_type='dqn')
                dqn_results.append(dqn_perf)
                print(f"  DQN  - Success: {dqn_perf['success']}, Steps: {dqn_perf['steps']}")
        
        self.results['generalization'] = {
            'neat': neat_results,
            'dqn': dqn_results
        }
        
        # Visualize
        self._plot_generalization()
        
        return self.results['generalization']
    
    def _generate_random_maze(self, size: int = 10) -> np.ndarray:
        """Generate a random maze configuration."""
        maze = np.zeros((size, size), dtype=int)
        
        # Add random walls
        num_walls = size * 2
        for _ in range(num_walls):
            x, y = np.random.randint(0, size, 2)
            maze[x, y] = 1
        
        # Add goal
        maze[size-2, size-2] = 2
        
        # Ensure start is clear
        maze[0, 0] = 0
        
        # Add some traps
        num_traps = 2
        for _ in range(num_traps):
            x, y = np.random.randint(1, size-1, 2)
            if maze[x, y] == 0:
                maze[x, y] = 3
        
        return maze
    
    def _evaluate_on_env(self, model, env: MazeEnv, agent_type: str) -> Dict:
        """Evaluate model on a specific environment."""
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < 500:
            # Get action
            if agent_type == 'neat':
                output = model.activate(obs)
                action = np.argmax(output)
            else:  # DQN
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = model(obs_tensor)
                    action = q_values.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        reached_goal = (terminated and 
                      env.maze[int(env.agent_pos[0]), 
                      int(env.agent_pos[1])] == env.GOAL)
        
        return {
            'success': reached_goal,
            'steps': steps,
            'reward': total_reward,
            'distance': info['distance_to_goal']
        }
    
    def _plot_generalization(self):
        """Plot generalization results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        neat_results = self.results['generalization']['neat']
        dqn_results = self.results['generalization']['dqn']
        
        # Success rate comparison
        ax = axes[0]
        methods = ['NEAT', 'DQN']
        success_rates = []
        
        if neat_results:
            neat_success = sum(r['success'] for r in neat_results) / len(neat_results)
            success_rates.append(neat_success)
        else:
            success_rates.append(0)
        
        if dqn_results:
            dqn_success = sum(r['success'] for r in dqn_results) / len(dqn_results)
            success_rates.append(dqn_success)
        else:
            success_rates.append(0)
        
        bars = ax.bar(methods, success_rates, color=['#3498db', '#e74c3c'], 
                     alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Success Rate on New Mazes', fontsize=12)
        ax.set_title('Generalization Performance', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Distribution of steps
        ax = axes[1]
        if neat_results:
            neat_steps = [r['steps'] for r in neat_results]
            ax.hist(neat_steps, bins=15, alpha=0.6, color='blue', 
                   label='NEAT', edgecolor='black')
        
        if dqn_results:
            dqn_steps = [r['steps'] for r in dqn_results]
            ax.hist(dqn_steps, bins=15, alpha=0.6, color='red', 
                   label='DQN', edgecolor='black')
        
        ax.set_xlabel('Steps to Goal', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Solving Times', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('analysis/generalization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_failure_modes(self):
        """
        Identify and visualize failure modes for both agents.
        """
        print("\n" + "="*60)
        print("FAILURE MODE ANALYSIS")
        print("="*60)
        
        failure_types = {
            'stuck_in_loop': 0,
            'trapped': 0,
            'timeout': 0,
            'wrong_direction': 0
        }
        
        neat_failures = failure_types.copy()
        dqn_failures = failure_types.copy()
        
        num_episodes = 50
        
        for ep in range(num_episodes):
            # Test NEAT
            if self.neat_net:
                failure_type = self._identify_failure_mode(self.neat_net, agent_type='neat')
                if failure_type:
                    neat_failures[failure_type] += 1
            
            # Test DQN
            if self.dqn_model:
                failure_type = self._identify_failure_mode(self.dqn_model, agent_type='dqn')
                if failure_type:
                    dqn_failures[failure_type] += 1
        
        self.results['failure_modes'] = {
            'neat': neat_failures,
            'dqn': dqn_failures
        }
        
        print("\nNEAT Failure Modes:")
        for mode, count in neat_failures.items():
            print(f"  {mode}: {count}/{num_episodes} ({count/num_episodes:.1%})")
        
        print("\nDQN Failure Modes:")
        for mode, count in dqn_failures.items():
            print(f"  {mode}: {count}/{num_episodes} ({count/num_episodes:.1%})")
        
        # Visualize
        self._plot_failure_modes()
        
        return self.results['failure_modes']
    
    def _identify_failure_mode(self, model, agent_type: str) -> str:
        """Identify the type of failure."""
        obs, _ = self.env.reset()
        done = False
        steps = 0
        trajectory = []
        
        while not done and steps < 500:
            # Get action
            if agent_type == 'neat':
                output = model.activate(obs)
                action = np.argmax(output)
            else:  # DQN
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = model(obs_tensor)
                    action = q_values.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            trajectory.append(self.env.agent_pos.copy())
            steps += 1
            done = terminated or truncated
        
        # Analyze failure
        reached_goal = (terminated and 
                      self.env.maze[int(self.env.agent_pos[0]), 
                      int(self.env.agent_pos[1])] == self.env.GOAL)
        
        if reached_goal:
            return None  # No failure
        
        # Check for loop
        if len(trajectory) > 20:
            recent_positions = [tuple(p) for p in trajectory[-20:]]
            if len(set(recent_positions)) < 5:
                return 'stuck_in_loop'
        
        # Check if trapped
        x, y = int(self.env.agent_pos[0]), int(self.env.agent_pos[1])
        if self.env.maze[x, y] == self.env.TRAP:
            return 'trapped'
        
        # Check if timeout
        if steps >= 500:
            return 'timeout'
        
        # Default: wrong direction
        return 'wrong_direction'
    
    def _plot_failure_modes(self):
        """Plot failure mode comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        neat_failures = self.results['failure_modes']['neat']
        dqn_failures = self.results['failure_modes']['dqn']
        
        failure_types = list(neat_failures.keys())
        x = np.arange(len(failure_types))
        width = 0.35
        
        neat_counts = [neat_failures[ft] for ft in failure_types]
        dqn_counts = [dqn_failures[ft] for ft in failure_types]
        
        bars1 = ax.bar(x - width/2, neat_counts, width, label='NEAT', 
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, dqn_counts, width, label='DQN', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Failure Type', fontsize=12)
        ax.set_ylabel('Number of Occurrences', fontsize=12)
        ax.set_title('Failure Mode Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ft.replace('_', ' ').title() for ft in failure_types])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('analysis/failure_modes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_robustness_score(self) -> Dict:
        """
        Compute overall robustness score for both agents.
        """
        print("\n" + "="*60)
        print("COMPUTING ROBUSTNESS SCORES")
        print("="*60)
        
        scores = {'neat': {}, 'dqn': {}}
        
        # Noise robustness (0-100)
        if 'noise_sensitivity' in self.results:
            for agent in ['neat', 'dqn']:
                if self.results['noise_sensitivity'][agent]:
                    # Average success rate across noise levels
                    success_rates = [r['success_rate'] for r in 
                                   self.results['noise_sensitivity'][agent]]
                    scores[agent]['noise_robustness'] = np.mean(success_rates) * 100
        
        # Generalization score (0-100)
        if 'generalization' in self.results:
            for agent in ['neat', 'dqn']:
                if self.results['generalization'][agent]:
                    successes = sum(r['success'] for r in self.results['generalization'][agent])
                    total = len(self.results['generalization'][agent])
                    scores[agent]['generalization_score'] = (successes / total) * 100
        
        # Failure resilience (0-100)
        if 'failure_modes' in self.results:
            for agent in ['neat', 'dqn']:
                failures = self.results['failure_modes'][agent]
                total_failures = sum(failures.values())
                scores[agent]['failure_resilience'] = max(0, 100 - total_failures)
        
        # Overall score
        for agent in ['neat', 'dqn']:
            agent_scores = [v for v in scores[agent].values() if v is not None]
            if agent_scores:
                scores[agent]['overall'] = np.mean(agent_scores)
        
        self.results['robustness_scores'] = scores
        
        # Print results
        print("\nROBUSTNESS SCORES:")
        print("\nNEAT:")
        for metric, score in scores['neat'].items():
            print(f"  {metric}: {score:.2f}/100")
        
        print("\nDQN:")
        for metric, score in scores['dqn'].items():
            print(f"  {metric}: {score:.2f}/100")
        
        # Visualize
        self._plot_robustness_scores()
        
        return scores
    
    def _plot_robustness_scores(self):
        """Plot robustness scores comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = self.results['robustness_scores']
        
        # Extract metrics (excluding 'overall')
        metrics = [m for m in scores['neat'].keys() if m != 'overall']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        neat_scores = [scores['neat'].get(m, 0) for m in metrics]
        dqn_scores = [scores['dqn'].get(m, 0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, neat_scores, width, label='NEAT', 
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, dqn_scores, width, label='DQN', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score (0-100)', fontsize=12)
        ax.set_title('Robustness Scores Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=15)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add overall scores as text
        overall_text = f"Overall Scores:\nNEAT: {scores['neat']['overall']:.1f}/100\n"
        overall_text += f"DQN: {scores['dqn']['overall']:.1f}/100"
        ax.text(0.98, 0.98, overall_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('analysis/robustness_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_tests(self):
        """Run complete robustness test suite."""
        print("\n" + "="*70)
        print(" " * 15 + "COMPREHENSIVE ROBUSTNESS TEST SUITE")
        print("="*70)
        
        # Test 1: Noise Sensitivity
        self.test_noise_sensitivity()
        
        # Test 2: Generalization
        self.test_generalization()
        
        # Test 3: Failure Modes
        self.test_failure_modes()
        
        # Compute final scores
        self.compute_robustness_score()
        
        # Save results
        self._save_results()
        
        print("\n" + "="*70)
        print("ALL ROBUSTNESS TESTS COMPLETE")
        print("Results saved to analysis/ directory")
        print("="*70)
    
    def _save_results(self):
        """Save all test results to JSON."""
        # Convert to serializable format
        results_copy = {}
        for key, value in self.results.items():
            results_copy[key] = value
        
        filepath = 'analysis/robustness_test_results.json'
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\nResults saved to {filepath}")

if __name__ == '__main__':
    # Initialize test suite
    suite = RobustnessTestSuite(
        neat_model_path='logs/neat/best_genome_gen_50.pkl',
        dqn_model_path='logs/dqn/best_model.pth'
    )
    
    # Run all tests
    suite.run_all_tests()
