import neat
import numpy as np
import pickle
import json
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append('..')
from env.maze_env import MazeEnv

class NEATMazeSolver:
    """
    NEAT-based solver for maze navigation with comprehensive logging.
    """
    
    def __init__(self, config_path: str, log_dir: str = 'logs/neat'):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
        self.config_path = config_path  # Store config path for later use
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Logging structures
        self.generation_stats = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'min_fitness': [],
            'species_count': [],
            'success_rate': [],
            'avg_steps': [],
            'best_trajectory': []
        }
        
        self.current_generation = 0
        self.best_genome = None
        self.best_fitness = -float('inf')
        
        # Environment for evaluation
        self.env = MazeEnv()
        
    def eval_genome(self, genome, config) -> Tuple[float, Dict]:
        """
        Evaluate a single genome in the maze environment.
        Returns fitness and additional metrics.
        """
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        reached_goal = False
        min_distance = float('inf')
        
        while not done and steps < 500:
            # Get action from neural network
            output = net.activate(obs)
            action = np.argmax(output)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Track minimum distance reached
            current_distance = np.linalg.norm(self.env.agent_pos - self.env.goal_pos)
            min_distance = min(min_distance, current_distance)
            
            if terminated and self.env.maze[int(self.env.agent_pos[0]), 
                                           int(self.env.agent_pos[1])] == self.env.GOAL:
                reached_goal = True
        
        # Improved fitness function
        if reached_goal:
            # Big reward for success, bonus for efficiency
            fitness = 2000 + (500 - steps) * 5  # Up to 4500 for fast solution
        else:
            # Reward for getting closer to goal
            max_distance = np.linalg.norm(np.array([0, 0]) - self.env.goal_pos)
            distance_fitness = (1 - min_distance / max_distance) * 800
            
            # Reward exploration heavily
            exploration_bonus = len(self.env.visited_cells) * 5
            
            # Penalize timeout
            timeout_penalty = -200 if steps >= 500 else 0
            
            fitness = distance_fitness + exploration_bonus + total_reward + timeout_penalty
        
        metrics = {
            'fitness': fitness,
            'reached_goal': reached_goal,
            'steps': steps,
            'final_distance': np.linalg.norm(self.env.agent_pos - self.env.goal_pos),
            'min_distance': min_distance,
            'cells_explored': len(self.env.visited_cells),
            'trajectory': self.env.get_trajectory().copy()
        }
        
        return fitness, metrics
    
    def eval_genomes(self, genomes, config):
        """Evaluate all genomes in a generation."""
        generation_metrics = []
        successful_runs = 0
        
        for genome_id, genome in genomes:
            fitness, metrics = self.eval_genome(genome, config)
            genome.fitness = fitness
            generation_metrics.append(metrics)
            
            if metrics['reached_goal']:
                successful_runs += 1
            
            # Track best genome
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genome
                self.generation_stats['best_trajectory'] = metrics['trajectory']
        
        # Log generation statistics
        fitnesses = [m['fitness'] for m in generation_metrics]
        steps = [m['steps'] for m in generation_metrics]
        
        self.generation_stats['generation'].append(self.current_generation)
        self.generation_stats['best_fitness'].append(max(fitnesses))
        self.generation_stats['avg_fitness'].append(np.mean(fitnesses))
        self.generation_stats['min_fitness'].append(min(fitnesses))
        self.generation_stats['success_rate'].append(successful_runs / len(genomes))
        self.generation_stats['avg_steps'].append(np.mean(steps))
        
        print(f"Generation {self.current_generation}:")
        print(f"  Best Fitness: {max(fitnesses):.2f}")
        print(f"  Avg Fitness: {np.mean(fitnesses):.2f}")
        print(f"  Success Rate: {successful_runs}/{len(genomes)}")
        print(f"  Avg Steps: {np.mean(steps):.1f}")
        
        self.current_generation += 1
        
        # Save periodic checkpoints
        if self.current_generation % 10 == 0:
            self.save_checkpoint()
    
    def train(self, generations: int = 100):
        """Train NEAT for specified generations."""
        print("Starting NEAT Training...")
        print(f"Configuration: {self.config}")
        
        # Create population
        population = neat.Population(self.config)
        
        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(10, 
                                filename_prefix=f'{self.log_dir}/neat-checkpoint-'))
        
        # Track species
        def track_species(species_set, generation):
            self.generation_stats['species_count'].append(len(species_set.species))
        
        # Run evolution
        winner = population.run(self.eval_genomes, generations)
        
        # Save final results
        self.best_genome = winner
        self.save_results(winner, stats)
        
        return winner
    
    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_data = {
            'generation': self.current_generation,
            'best_fitness': self.best_fitness,
            'generation_stats': self.generation_stats
        }
        
        filepath = os.path.join(self.log_dir, f'training_stats_gen_{self.current_generation}.json')
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            stats_copy = {k: v if not isinstance(v, list) or len(v) == 0 or 
                         not isinstance(v[0], np.ndarray) else [t.tolist() if isinstance(t, np.ndarray) 
                         else t for t in v]
                         for k, v in self.generation_stats.items()}
            checkpoint_data['generation_stats'] = stats_copy
            json.dump(checkpoint_data, f, indent=2)
        
        # Save best genome
        if self.best_genome:
            with open(os.path.join(self.log_dir, f'best_genome_gen_{self.current_generation}.pkl'), 'wb') as f:
                pickle.dump(self.best_genome, f)
    
    def save_results(self, winner, stats):
        """Save final training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save winner genome
        with open(os.path.join(self.log_dir, f'winner_{timestamp}.pkl'), 'wb') as f:
            pickle.dump(winner, f)
        
        # Save complete statistics
        with open(os.path.join(self.log_dir, f'final_stats_{timestamp}.json'), 'w') as f:
            stats_copy = {k: v if not isinstance(v, list) or len(v) == 0 or 
                         not isinstance(v[0], np.ndarray) else [t.tolist() if isinstance(t, np.ndarray) 
                         else t for t in v]
                         for k, v in self.generation_stats.items()}
            json.dump(stats_copy, f, indent=2)
        
        print(f"\nTraining complete! Results saved to {self.log_dir}")
        print(f"Best fitness achieved: {self.best_fitness:.2f}")
    
    def visualize_training(self):
        """Create training visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fitness over generations
        ax = axes[0, 0]
        ax.plot(self.generation_stats['generation'], 
               self.generation_stats['best_fitness'], 
               label='Best', linewidth=2)
        ax.plot(self.generation_stats['generation'], 
               self.generation_stats['avg_fitness'], 
               label='Average', linewidth=2)
        ax.fill_between(self.generation_stats['generation'],
                        self.generation_stats['min_fitness'],
                        self.generation_stats['best_fitness'],
                        alpha=0.2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Success rate
        ax = axes[0, 1]
        ax.plot(self.generation_stats['generation'], 
               self.generation_stats['success_rate'], 
               linewidth=2, color='green')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Success Rate')
        ax.set_title('Goal Reaching Success Rate')
        ax.grid(True, alpha=0.3)
        
        # Average steps
        ax = axes[1, 0]
        ax.plot(self.generation_stats['generation'], 
               self.generation_stats['avg_steps'], 
               linewidth=2, color='purple')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Steps')
        ax.set_title('Efficiency (Steps to Goal)')
        ax.grid(True, alpha=0.3)
        
        # Species count
        ax = axes[1, 1]
        if len(self.generation_stats['species_count']) > 0:
            ax.plot(self.generation_stats['generation'][:len(self.generation_stats['species_count'])], 
                   self.generation_stats['species_count'], 
                   linewidth=2, color='orange')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Species')
        ax.set_title('Species Diversity')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def evaluate_best(self, render: bool = True, num_episodes: int = 5):
        """Evaluate the best genome."""
        if self.best_genome is None:
            print("No best genome found. Train first!")
            return
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config.filename)
        
        net = neat.nn.FeedForwardNetwork.create(self.best_genome, config)
        
        results = []
        for episode in range(num_episodes):
            obs, _ = self.env.reset(seed=episode)  # Different seed each episode
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 500:
                output = net.activate(obs)
                action = np.argmax(output)
                
                # Add small randomness for robustness
                if np.random.random() < 0.05:
                    action = self.env.action_space.sample()
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
                if render and episode == 0:
                    self.env.render()
            
            reached_goal = (terminated and self.env.maze[int(self.env.agent_pos[0]), 
                           int(self.env.agent_pos[1])] == self.env.GOAL)
            
            results.append({
                'episode': episode,
                'reward': total_reward,
                'steps': steps,
                'reached_goal': reached_goal,
                'trajectory': self.env.get_trajectory()
            })
            
            print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}, Goal={'YES' if reached_goal else 'NO'}")
        
        return results

def create_neat_config(filename: str = 'config-neat.txt'):
    """Create NEAT configuration file."""
    # Create directory only if path contains directory
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    config_content = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 1500
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = relu
activation_mutate_rate  = 0.1
activation_options      = tanh relu sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.8
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.8
conn_delete_prob        = 0.2

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.05

feed_forward            = True
initial_connection      = full_direct

# node add/remove rates
node_add_prob           = 0.5
node_delete_prob        = 0.2

# network parameters
num_hidden              = 3
num_inputs              = 12
num_outputs             = 4

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.9
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 15
species_elitism      = 3

[DefaultReproduction]
elitism            = 5
survival_threshold = 0.2
min_species_size   = 2
"""
    
    with open(filename, 'w') as f:
        f.write(config_content)
    
    print(f"NEAT configuration saved to {filename}")
    return filename

if __name__ == '__main__':
    # Create config file
    config_path = create_neat_config()
    
    # Initialize solver
    solver = NEATMazeSolver(config_path)
    
    # Train
    winner = solver.train(generations=50)
    
    # Visualize training
    solver.visualize_training()
    
    # Evaluate best genome
    results = solver.evaluate_best(render=True, num_episodes=5)
