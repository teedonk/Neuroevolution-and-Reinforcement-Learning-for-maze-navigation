# Environment Design Guide

## Overview

`MazeEnv` is a Gymnasium-compatible environment for maze navigation with misleading paths.

## Cell Types

```python
EMPTY = 0      # Navigable space (white)
WALL = 1       # Impassable barrier (black)
GOAL = 2       # Target destination (gold)
TRAP = 3       # Dangerous cell (red)
MISLEAD = 4    # Deceptive path (orange)
```

## Observation Space

**12-dimensional vector:**
```python
[
    agent_x_norm,      # Normalized X position [0, 1]
    agent_y_norm,      # Normalized Y position [0, 1]
    goal_x_norm,       # Normalized goal X [0, 1]
    goal_y_norm,       # Normalized goal Y [0, 1]
    cell_NW,           # Northwest cell [0, 1]
    cell_N,            # North cell [0, 1]
    cell_NE,           # Northeast cell [0, 1]
    cell_W,            # West cell [0, 1]
    cell_E,            # East cell [0, 1]
    cell_SW,           # Southwest cell [0, 1]
    cell_S,            # South cell [0, 1]
    cell_SE            # Southeast cell [0, 1]
]
```

**Why this design?**
- Compact: Fits in small networks
- Informative: Position + local view + goal
- Normalized: Easier to learn

## Action Space

**Discrete(4):**
```python
0 = UP    (row - 1)
1 = RIGHT (col + 1)
2 = DOWN  (row + 1)
3 = LEFT  (col - 1)
```

## Reward Function

```python
# Goal reached
if cell == GOAL:
    reward = +100

# Hit trap
elif cell == TRAP:
    reward = -10

# Hit wall
elif invalid_move:
    reward = -1

# Misleading path
elif cell == MISLEAD:
    reward = +0.5  # Tempting but suboptimal

# Normal movement
else:
    # Distance-based shaping
    reward = (old_distance - new_distance) * 0.5
    
    # Exploration bonus
    if cell_not_visited:
        reward += 0.1
    else:
        reward -= 0.2  # Revisit penalty
    
    # Time penalty
    reward -= 0.01
```

## Creating Custom Mazes

### Method 1: NumPy Array
```python
import numpy as np
from env.maze_env import MazeEnv

custom_maze = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 3],
    [0, 0, 0, 0, 2]
])

env = MazeEnv(maze_layout=custom_maze)
```

### Method 2: JSON File
```python
# Create maze
env = MazeEnv()
env.save_maze('my_maze.json')

# Load maze
env = MazeEnv.load_maze('my_maze.json')
```

### Method 3: Programmatic
```python
def create_complex_maze(size=15):
    maze = np.zeros((size, size), dtype=int)
    
    # Add walls (create corridors)
    for i in range(2, size-2, 3):
        maze[i, :] = 1
        maze[:, i] = 1
    
    # Add openings
    for i in range(2, size-2, 3):
        opening = np.random.randint(0, size)
        maze[i, opening] = 0
    
    # Add traps
    num_traps = size // 3
    for _ in range(num_traps):
        x, y = np.random.randint(0, size, 2)
        if maze[x, y] == 0:
            maze[x, y] = 3
    
    # Add misleading paths
    num_mislead = size // 4
    for _ in range(num_mislead):
        x, y = np.random.randint(0, size, 2)
        if maze[x, y] == 0:
            maze[x, y] = 4
    
    # Set goal
    maze[size-2, size-2] = 2
    maze[0, 0] = 0  # Ensure start is clear
    
    return maze
```

## Maze Design Principles

### Easy Maze
```python
✓ Direct path to goal
✓ Few obstacles
✓ No misleading paths
✗ Not challenging
```

### Medium Maze (Default)
```python
✓ Multiple paths
✓ Some dead ends
✓ Few misleading paths
✓ Strategic traps
✓ Balanced challenge
```

### Hard Maze
```python
✓ Many dead ends
✓ Multiple misleading paths
✓ Traps near goal
✓ Long optimal path
✗ May be too difficult
```

## Customizing Rewards

### Sparse Rewards
```python
env = MazeEnv(use_distance_reward=False)

# Only +100 for goal, -1 for walls
# Harder to learn, but clearer objective
```

### Dense Rewards
```python
env = MazeEnv(use_distance_reward=True)

# Continuous feedback via distance
# Easier to learn, but may encourage greediness
```

### Custom Reward Function
```python
class CustomMazeEnv(MazeEnv):
    def _calculate_reward(self, new_pos, hit_wall):
        # Your custom logic here
        if self.maze[new_pos] == self.GOAL:
            return 1000  # Huge reward
        
        # Penalize based on Manhattan distance
        manhattan = abs(new_pos[0] - self.goal_pos[0]) + \
                   abs(new_pos[1] - self.goal_pos[1])
        return -manhattan * 0.1
```

## Environment Parameters

```python
env = MazeEnv(
    maze_layout=None,              # Custom maze or None for default
    render_mode='human',           # 'human' or 'rgb_array'
    max_steps=500,                 # Episode timeout
    use_distance_reward=True       # Distance-based shaping
)
```

## Testing Your Environment

```python
# Basic test
env = MazeEnv()
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print(f"Maze size: {env.height}x{env.width}")

# Random agent test
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

# Render test
env.reset()
for _ in range(50):
    env.render()
    action = env.action_space.sample()
    env.step(action)
```

## Advanced Features

### State History
```python
# Access trajectory
trajectory = env.get_trajectory()
print(f"Path taken: {len(trajectory)} steps")

# Access visited cells
visited = env.visited_cells
print(f"Explored: {len(visited)} cells")
```

### Custom Rendering
```python
# Get RGB array
env = MazeEnv(render_mode='rgb_array')
obs, _ = env.reset()
rgb_array = env.render()  # Returns numpy array

# Save as image
from PIL import Image
img = Image.fromarray(rgb_array)
img.save('maze_state.png')
```

### Episode Info
```python
obs, reward, terminated, truncated, info = env.step(action)

print(info)
# {
#     'steps': 45,
#     'distance_to_goal': 5.2,
#     'trajectory_length': 45,
#     'unique_cells_visited': 23
# }
```

## Common Patterns

### Training Loop
```python
env = MazeEnv()

for episode in range(100):
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"Episode {episode}: Reward = {total_reward}")
```

### Evaluation
```python
def evaluate_agent(agent, env, num_episodes=10):
    successes = 0
    total_steps = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if terminated and env.maze[tuple(env.agent_pos)] == env.GOAL:
            successes += 1
        
        total_steps.append(steps)
    
    return {
        'success_rate': successes / num_episodes,
        'avg_steps': np.mean(total_steps)
    }
```

## Tips & Tricks

### Debugging
```python
# Visualize maze structure
env = MazeEnv()
env.reset()
env.render()

# Check if goal reachable
from scipy.ndimage import label
labeled, num_features = label(env.maze != 1)
start_region = labeled[tuple(env.start_pos)]
goal_region = labeled[tuple(env.goal_pos)]
assert start_region == goal_region, "Goal not reachable!"
```

### Performance
```python
# Disable rendering for faster training
env = MazeEnv(render_mode=None)

# Reduce max steps if agents timeout frequently
env = MazeEnv(max_steps=200)
```

### Curriculum Learning
```python
# Start with easy maze
easy_maze = create_maze(size=5, complexity=0.1)
env = MazeEnv(maze_layout=easy_maze)

# Train until 80% success rate
# Then increase difficulty

medium_maze = create_maze(size=10, complexity=0.3)
env = MazeEnv(maze_layout=medium_maze)
```

## Quick Reference

```python
# Create
env = MazeEnv()

# Reset
obs, info = env.reset(seed=42)

# Step
obs, reward, terminated, truncated, info = env.step(action)

# Render
env.render()

# Save
env.save_maze('my_maze.json')

# Load
env = MazeEnv.load_maze('my_maze.json')
```

---

**Next:** [Visualization Guide](visualization_guide.md) | [NEAT Tutorial](neat_tutorial.md)
