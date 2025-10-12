import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
sys.path.append('..')
from env.maze_env import MazeEnv

class DQN(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNMazeSolver:
    """
    DQN-based solver for maze navigation with comprehensive logging.
    """
    
    def __init__(self, env: MazeEnv = None, log_dir: str = 'logs/dqn'):
        self.env = env if env else MazeEnv()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network parameters
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        
        # Initialize networks
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 10
        
        # Logging structures
        self.training_stats = {
            'episode': [],
            'reward': [],
            'steps': [],
            'epsilon': [],
            'loss': [],
            'q_values': [],
            'success': [],
            'distance_to_goal': [],
            'exploration_count': []
        }
        
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.best_trajectory = None
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute TD loss."""
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        return loss
    
    def update_network(self):
        """Update policy network using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int = 500, verbose: bool = True):
        """Train DQN agent."""
        print("Starting DQN Training...")
        print(f"Episodes: {num_episodes}")
        print(f"Device: {self.device}")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = []
            episode_q_values = []
            steps = 0
            done = False
            
            while not done and steps < 500:
                # Select action
                action = self.select_action(state, training=True)
                
                # Store Q-values for analysis
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_vals = self.policy_net(state_tensor)
                    episode_q_values.append(q_vals.max().item())
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update network
                loss = self.update_network()
                if loss is not None:
                    episode_loss.append(loss)
                
                episode_reward += reward
                steps += 1
                state = next_state
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Check success
            reached_goal = (terminated and 
                          self.env.maze[int(self.env.agent_pos[0]), 
                          int(self.env.agent_pos[1])] == self.env.GOAL)
            
            # Log statistics
            self.training_stats['episode'].append(episode)
            self.training_stats['reward'].append(episode_reward)
            self.training_stats['steps'].append(steps)
            self.training_stats['epsilon'].append(self.epsilon)
            self.training_stats['loss'].append(np.mean(episode_loss) if episode_loss else 0)
            self.training_stats['q_values'].append(np.mean(episode_q_values) if episode_q_values else 0)
            self.training_stats['success'].append(1 if reached_goal else 0)
            self.training_stats['distance_to_goal'].append(info['distance_to_goal'])
            self.training_stats['exploration_count'].append(info['unique_cells_visited'])
            
            # Track best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_trajectory = self.env.get_trajectory()
                self.save_model('best_model.pth')
            
            # Verbose output
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['reward'][-10:])
                avg_steps = np.mean(self.training_stats['steps'][-10:])
                success_rate = np.mean(self.training_stats['success'][-10:])
                avg_loss = np.mean(self.training_stats['loss'][-10:])
                
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Steps: {avg_steps:.1f}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
            
            # Save periodic checkpoints
            if (episode + 1) % 50 == 0:
                self.save_checkpoint(episode)
        
        # Save final results
        self.save_results()
        print("\nTraining complete!")
        print(f"Best reward: {self.best_reward:.2f}")
        
    def save_model(self, filename: str):
        """Save model weights."""
        filepath = os.path.join(self.log_dir, filename)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_reward': self.best_reward
        }, filepath)
    
    def load_model(self, filename: str):
        """Load model weights."""
        filepath = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.best_reward = checkpoint['best_reward']
        print(f"Model loaded from {filepath}")
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_data = {
            'episode': episode,
            'training_stats': self.training_stats,
            'best_reward': self.best_reward,
            'epsilon': self.epsilon
        }
        
        filepath = os.path.join(self.log_dir, f'checkpoint_ep_{episode}.json')
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.save_model(f'model_ep_{episode}.pth')
    
    def save_results(self):
        """Save final training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training statistics
        filepath = os.path.join(self.log_dir, f'training_stats_{timestamp}.json')
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Save final model
        self.save_model(f'final_model_{timestamp}.pth')
        
        print(f"Results saved to {self.log_dir}")
    
    def visualize_training(self):
        """Create comprehensive training visualizations."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Reward over episodes
        ax = axes[0, 0]
        ax.plot(self.training_stats['episode'], self.training_stats['reward'], 
               alpha=0.3, color='blue')
        # Moving average
        window = 50
        if len(self.training_stats['reward']) >= window:
            moving_avg = np.convolve(self.training_stats['reward'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.training_stats['reward'])), 
                   moving_avg, linewidth=2, color='red', label=f'{window}-ep MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Success rate
        ax = axes[0, 1]
        window = 50
        if len(self.training_stats['success']) >= window:
            success_rate = np.convolve(self.training_stats['success'], 
                                      np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.training_stats['success'])), 
                   success_rate, linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Goal Reaching Success Rate (50-ep MA)')
        ax.grid(True, alpha=0.3)
        
        # Steps per episode
        ax = axes[1, 0]
        ax.plot(self.training_stats['episode'], self.training_stats['steps'], 
               alpha=0.3, color='purple')
        if len(self.training_stats['steps']) >= window:
            moving_avg = np.convolve(self.training_stats['steps'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.training_stats['steps'])), 
                   moving_avg, linewidth=2, color='orange', label=f'{window}-ep MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[1, 1]
        valid_losses = [l for l in self.training_stats['loss'] if l > 0]
        valid_episodes = [e for e, l in zip(self.training_stats['episode'], 
                         self.training_stats['loss']) if l > 0]
        ax.plot(valid_episodes, valid_losses, alpha=0.3, color='red')
        if len(valid_losses) >= window:
            moving_avg = np.convolve(valid_losses, np.ones(window)/window, mode='valid')
            ax.plot(valid_episodes[window-1:], moving_avg, linewidth=2, 
                   color='darkred', label=f'{window}-ep MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax = axes[2, 0]
        ax.plot(self.training_stats['episode'], self.training_stats['epsilon'], 
               linewidth=2, color='brown')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (ε)')
        ax.grid(True, alpha=0.3)
        
        # Q-values
        ax = axes[2, 1]
        ax.plot(self.training_stats['episode'], self.training_stats['q_values'], 
               alpha=0.3, color='cyan')
        if len(self.training_stats['q_values']) >= window:
            moving_avg = np.convolve(self.training_stats['q_values'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(self.training_stats['q_values'])), 
                   moving_avg, linewidth=2, color='blue', label=f'{window}-ep MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Max Q-Value')
        ax.set_title('Q-Value Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def evaluate(self, num_episodes: int = 10, render: bool = False):
        """Evaluate trained agent."""
        self.policy_net.eval()
        results = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=episode)  # Different seed each episode
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 500:
                # Use trained policy with small exploration for robustness
                if np.random.random() < 0.05:  # 5% exploration during eval
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(state, training=False)
                
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
                
                if render and episode == 0:
                    self.env.render()
            
            reached_goal = (terminated and 
                          self.env.maze[int(self.env.agent_pos[0]), 
                          int(self.env.agent_pos[1])] == self.env.GOAL)
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': steps,
                'reached_goal': reached_goal,
                'trajectory': self.env.get_trajectory()
            })
            
            print(f"Episode {episode+1}: Reward={episode_reward:.2f}, " + 
                  f"Steps={steps}, Goal={'YES' if reached_goal else 'NO'}")
        
        self.policy_net.train()
        return results

if __name__ == '__main__':
    # Initialize environment and solver
    env = MazeEnv()
    solver = DQNMazeSolver(env)
    
    # Train
    solver.train(num_episodes=500, verbose=True)
    
    # Visualize training
    solver.visualize_training()
    
    # Evaluate
    results = solver.evaluate(num_episodes=10, render=True)
