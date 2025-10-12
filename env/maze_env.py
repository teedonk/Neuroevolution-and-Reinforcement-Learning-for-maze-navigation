import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, Optional, Dict, List

class MazeEnv(gym.Env):
    """
    Custom Maze Environment with misleading paths and deceptive rewards.
    Designed to test both Neuroevolution and RL approaches.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    # Cell types
    EMPTY = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    MISLEAD = 4  # Looks promising but leads nowhere
    
    def __init__(self, maze_layout: np.ndarray = None, render_mode: Optional[str] = None,
                 max_steps: int = 500, use_distance_reward: bool = True):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.use_distance_reward = use_distance_reward
        
        # Default challenging maze with misleading paths
        if maze_layout is None:
            self.maze = self._create_default_maze()
        else:
            self.maze = maze_layout
            
        self.height, self.width = self.maze.shape
        
        # Find start and goal positions
        self.start_pos = self._find_position(self.EMPTY)
        self.goal_pos = self._find_position(self.GOAL)
        
        # State: [agent_x, agent_y]
        self.agent_pos = np.array(self.start_pos, dtype=np.float32)
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: normalized position + local view
        # [agent_x, agent_y, goal_x, goal_y] + 8 surrounding cells
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.float32
        )
        
        self.steps = 0
        self.trajectory = [self.agent_pos.copy()]
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        # For visualization
        self.fig = None
        self.ax = None
        
    def _create_default_maze(self) -> np.ndarray:
        """Create a challenging maze with multiple misleading paths."""
        maze = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],  # Misleading path
            [1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 2, 0],  # Trap before goal
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        return maze
    
    def _find_position(self, cell_type: int) -> Tuple[int, int]:
        """Find the first occurrence of a cell type."""
        positions = np.argwhere(self.maze == cell_type)
        if len(positions) == 0:
            if cell_type == self.EMPTY:
                return (0, 0)
            elif cell_type == self.GOAL:
                return (self.height - 2, self.width - 2)
        return tuple(positions[0])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with normalized coordinates and local view."""
        x, y = self.agent_pos
        
        # Normalized agent and goal positions
        norm_agent_x = x / (self.height - 1)
        norm_agent_y = y / (self.width - 1)
        norm_goal_x = self.goal_pos[0] / (self.height - 1)
        norm_goal_y = self.goal_pos[1] / (self.width - 1)
        
        # Get 8 surrounding cells (normalized)
        surrounding = []
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = int(x + dx), int(y + dy)
            if 0 <= nx < self.height and 0 <= ny < self.width:
                cell_value = self.maze[nx, ny] / 4.0  # Normalize
            else:
                cell_value = 0.25  # Wall indicator
            surrounding.append(cell_value)
        
        obs = np.array([norm_agent_x, norm_agent_y, norm_goal_x, norm_goal_y] + surrounding,
                       dtype=np.float32)
        return obs
    
    def _calculate_reward(self, new_pos: np.ndarray, hit_wall: bool) -> float:
        """Calculate reward based on action outcome."""
        x, y = int(new_pos[0]), int(new_pos[1])
        cell_type = self.maze[x, y]
        
        # Goal reached
        if cell_type == self.GOAL:
            return 100.0
        
        # Hit a trap
        if cell_type == self.TRAP:
            return -10.0
        
        # Hit a wall
        if hit_wall:
            return -1.0
        
        # Misleading path (looks good but isn't)
        if cell_type == self.MISLEAD:
            return 0.5  # Small positive to make it tempting
        
        # Distance-based shaping reward
        if self.use_distance_reward:
            old_dist = np.linalg.norm(self.agent_pos - self.goal_pos)
            new_dist = np.linalg.norm(new_pos - self.goal_pos)
            distance_reward = (old_dist - new_dist) * 0.5
        else:
            distance_reward = 0.0
        
        # Penalty for revisiting cells
        if tuple(new_pos) in self.visited_cells:
            revisit_penalty = -0.2
        else:
            revisit_penalty = 0.1  # Small reward for exploration
        
        # Small time penalty to encourage efficiency
        time_penalty = -0.01
        
        return distance_reward + revisit_penalty + time_penalty
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state."""
        self.steps += 1
        
        # Action directions: 0=up, 1=right, 2=down, 3=left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]
        
        new_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        x, y = int(new_pos[0]), int(new_pos[1])
        
        # Check boundaries and walls
        hit_wall = False
        if not (0 <= x < self.height and 0 <= y < self.width):
            new_pos = self.agent_pos.copy()
            hit_wall = True
        elif self.maze[x, y] == self.WALL:
            new_pos = self.agent_pos.copy()
            hit_wall = True
        
        # Calculate reward
        reward = self._calculate_reward(new_pos, hit_wall)
        
        # Update position
        self.agent_pos = new_pos
        self.trajectory.append(self.agent_pos.copy())
        self.visited_cells.add(tuple(self.agent_pos))
        
        # Check termination conditions
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        terminated = self.maze[x, y] == self.GOAL
        truncated = self.steps >= self.max_steps
        
        obs = self._get_observation()
        info = {
            'steps': self.steps,
            'distance_to_goal': np.linalg.norm(self.agent_pos - self.goal_pos),
            'trajectory_length': len(self.trajectory),
            'unique_cells_visited': len(self.visited_cells)
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.agent_pos = np.array(self.start_pos, dtype=np.float32)
        self.steps = 0
        self.trajectory = [self.agent_pos.copy()]
        self.visited_cells = set()
        self.visited_cells.add(tuple(self.agent_pos))
        
        return self._get_observation(), {}
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                plt.ion()
            
            self.ax.clear()
            self._draw_maze()
            plt.pause(0.01)
            
        elif self.render_mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(8, 8))
            self._draw_maze(ax)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
    
    def _draw_maze(self, ax=None):
        """Draw the current maze state."""
        if ax is None:
            ax = self.ax
        
        # Color map
        colors = {
            self.EMPTY: 'white',
            self.WALL: 'black',
            self.GOAL: 'gold',
            self.TRAP: 'red',
            self.MISLEAD: 'orange'
        }
        
        # Draw maze
        for i in range(self.height):
            for j in range(self.width):
                color = colors[self.maze[i, j]]
                rect = Rectangle((j, self.height - 1 - i), 1, 1,
                               facecolor=color, edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 1] + 0.5, self.height - traj[:, 0] - 0.5,
                   'b-', alpha=0.3, linewidth=2, label='Trajectory')
        
        # Draw agent
        x, y = self.agent_pos
        ax.plot(y + 0.5, self.height - x - 0.5, 'bo', markersize=15, label='Agent')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(f'Steps: {self.steps} | Distance to Goal: {np.linalg.norm(self.agent_pos - self.goal_pos):.2f}')
        ax.legend(loc='upper right')
        ax.axis('off')
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Return the agent's trajectory."""
        return self.trajectory
    
    def save_maze(self, filename: str):
        """Save maze configuration to JSON."""
        config = {
            'maze': self.maze.tolist(),
            'start_pos': self.start_pos,
            'goal_pos': self.goal_pos,
            'height': self.height,
            'width': self.width
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_maze(filename: str) -> 'MazeEnv':
        """Load maze configuration from JSON."""
        with open(filename, 'r') as f:
            config = json.load(f)
        maze = np.array(config['maze'])
        return MazeEnv(maze_layout=maze)
