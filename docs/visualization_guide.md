# Visualization Guide

## Overview

This project includes **10+ visualization types** to understand agent behavior and performance.

## Quick Start

```python
from analysis.visualize_training import TrainingVisualizer

viz = TrainingVisualizer(
    neat_log_dir='logs/neat',
    dqn_log_dir='logs/dqn'
)

# Generate all visualizations
viz.create_comparison_dashboard()
viz.visualize_decision_boundaries()
viz.create_live_comparison()
viz.create_adaptation_animation()
```

## Visualization Types

### 1. Training Curves
**What:** Shows learning progress over time  
**When:** After training completes  
**Shows:** Fitness/reward, success rate, convergence

```python
# NEAT
neat_solver.visualize_training()

# DQN
dqn_solver.visualize_training()
```

**Output:** `training_curves.png`

**Interpretation:**
- Upward trend = learning
- Flat line = converged or stuck
- Oscillation = unstable training

### 2. Comparison Dashboard
**What:** Side-by-side performance comparison  
**When:** Compare trained agents  
**Shows:** All key metrics together

```python
viz.create_comparison_dashboard()
```

**Output:** `comparison_dashboard.png`

**Components:**
- Learning curves (NEAT vs DQN)
- Success rate bars
- Efficiency comparison
- Convergence analysis

### 3. Decision Boundaries
**What:** Heatmap of preferred actions  
**When:** Understand decision strategy  
**Shows:** What action agent prefers in each position

```python
viz.visualize_decision_boundaries()
```

**Output:** `decision_boundaries.png`

**Interpretation:**
- Color = preferred action
- Patterns = strategy
- Differences = NEAT vs DQN approaches

### 4. Live Comparison (Interactive)
**What:** Real-time side-by-side navigation  
**When:** Present results dynamically  
**Shows:** Both agents solving simultaneously

```bash
open analysis/interactive_dashboard.html
```

**Features:**
- Play/pause controls
- Speed adjustment
- Live Q-values
- Real-time metrics

### 5. Adaptation Animation
**What:** GIF showing strategy evolution  
**When:** Show learning process  
**Shows:** How behavior improves over time

```python
viz.create_adaptation_animation(save_path='adaptation.gif')
```

**Output:** `adaptation.gif`

**Use:** Presentations, social media

### 6. Robustness Plots
**What:** Performance under challenging conditions  
**When:** Test agent reliability  
**Shows:** Noise sensitivity, generalization

```python
from analysis.robustness_tests import RobustnessTestSuite

suite = RobustnessTestSuite(...)
suite.test_noise_sensitivity()
suite.test_generalization()
suite.compute_robustness_score()
```

**Outputs:**
- `noise_sensitivity.png`
- `generalization.png`
- `failure_modes.png`
- `robustness_scores.png`

## Custom Visualizations

### Plot Training Metrics
```python
import matplotlib.pyplot as plt
import json

# Load training data
with open('logs/neat/final_stats.json') as f:
    data = json.load(f)

# Plot custom metric
plt.figure(figsize=(10, 6))
plt.plot(data['generation'], data['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('NEAT Training Progress')
plt.grid(True)
plt.savefig('custom_plot.png')
plt.show()
```

### Trajectory Visualization
```python
from env.maze_env import MazeEnv
import matplotlib.pyplot as plt

env = MazeEnv()
obs, _ = env.reset()

# Run agent
trajectory = []
for _ in range(100):
    action = agent.select_action(obs)
    obs, _, done, _, _ = env.step(action)
    trajectory.append(env.agent_pos.copy())
    if done:
        break

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
env._draw_maze(ax)
traj = np.array(trajectory)
ax.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=2, alpha=0.6)
plt.savefig('trajectory.png')
```

### Heatmap of Visits
```python
import seaborn as sns

# Track visit counts
visit_count = np.zeros((env.height, env.width))

for episode in range(100):
    obs, _ = env.reset()
    for _ in range(500):
        action = agent.select_action(obs)
        obs, _, done, _, _ = env.step(action)
        x, y = env.agent_pos
        visit_count[int(x), int(y)] += 1
        if done:
            break

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(visit_count, cmap='YlOrRd', annot=False)
plt.title('Cell Visit Frequency')
plt.savefig('visit_heatmap.png')
```

### Q-Value Evolution
```python
# During DQN training
q_values_over_time = []

for episode in episodes:
    episode_q = []
    obs, _ = env.reset()
    
    for step in steps:
        q_vals = dqn_model(obs).detach().numpy()
        episode_q.append(q_vals.max())
        action = select_action(q_vals)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    q_values_over_time.append(np.mean(episode_q))

# Plot
plt.plot(q_values_over_time)
plt.xlabel('Episode')
plt.ylabel('Average Max Q-Value')
plt.title('Q-Value Evolution')
plt.savefig('q_evolution.png')
```

### Success Rate Timeline
```python
def plot_success_timeline(results, window=50):
    """Plot rolling success rate."""
    successes = [1 if r['reached_goal'] else 0 for r in results]
    
    rolling_success = []
    for i in range(len(successes)):
        start = max(0, i - window)
        rolling_success.append(np.mean(successes[start:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_success, linewidth=2)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Target')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate ({window}-episode window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('success_timeline.png')
```

## Styling Tips

### Professional Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Use colors
colors = {
    'neat': '#3498db',  # Blue
    'dqn': '#e74c3c',   # Red
    'success': '#27ae60',  # Green
    'fail': '#e67e22'   # Orange
}
```

### Publication Quality
```python
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Your plotting code
ax.plot(x, y, linewidth=2, label='My Data')

# Styling
ax.set_xlabel('X Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=14, fontweight='bold')
ax.set_title('Title', fontsize=16, fontweight='bold')
ax.legend(frameon=True, shadow=True)
ax.grid(True, alpha=0.3)

# Save high-res
plt.tight_layout()
plt.savefig('publication_plot.png', dpi=300, bbox_inches='tight')
```

### Dark Theme
```python
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, color='#00ff00', linewidth=2)
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#0d0d0d')
```

## Animation Guide

### Create Training GIF
```python
import matplotlib.animation as animation
from PIL import Image

fig, ax = plt.subplots()

def animate(frame):
    ax.clear()
    # Draw maze state at frame
    draw_maze_state(ax, frame)
    return ax,

anim = animation.FuncAnimation(
    fig, animate, frames=100, interval=50, blit=True
)

# Save
anim.save('training.gif', writer='pillow', fps=20)
```

### Video Export
```python
# Requires ffmpeg
anim.save('training.mp4', writer='ffmpeg', fps=30, bitrate=1800)
```

## Interactive Plots

### Plotly (Web-based)
```python
import plotly.graph_objects as go

# Create interactive plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=episodes, y=rewards,
    mode='lines',
    name='Reward',
    line=dict(color='blue', width=2)
))

fig.update_layout(
    title='Training Progress',
    xaxis_title='Episode',
    yaxis_title='Reward',
    hovermode='x'
)

fig.write_html('interactive_plot.html')
```

### Jupyter Widgets
```python
from ipywidgets import interact
import matplotlib.pyplot as plt

def plot_episode(episode_num):
    plt.figure(figsize=(10, 6))
    # Plot data for episode_num
    plt.plot(data[episode_num])
    plt.title(f'Episode {episode_num}')
    plt.show()

interact(plot_episode, episode_num=(0, 500, 10))
```

## Exporting Results

### Save All Figures
```python
def save_all_visualizations(output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Training curves
    neat_solver.visualize_training()
    plt.savefig(f'{output_dir}/neat_training.png')
    plt.close()
    
    dqn_solver.visualize_training()
    plt.savefig(f'{output_dir}/dqn_training.png')
    plt.close()
    
    # Comparison
    viz.create_comparison_dashboard()
    plt.savefig(f'{output_dir}/comparison.png')
    plt.close()
    
    print(f"All figures saved to {output_dir}/")
```

### Generate Report with Images
```python
from markdown2 import markdown

report = f"""
# Training Report

## NEAT Results
![NEAT Training](figures/neat_training.png)

## DQN Results
![DQN Training](figures/dqn_training.png)

## Comparison
![Comparison](figures/comparison.png)

## Summary
- NEAT Success Rate: {neat_success:.1%}
- DQN Success Rate: {dqn_success:.1%}
"""

# Save as HTML
with open('report.html', 'w') as f:
    f.write(markdown(report))
```

## Common Visualizations

### 1. Learning Curve
```python
plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
```

### 2. Comparison Bar Chart
```python
methods = ['NEAT', 'DQN']
values = [neat_score, dqn_score]
plt.bar(methods, values, color=['blue', 'red'])
plt.ylabel('Score')
plt.title('Performance Comparison')
```

### 3. Box Plot (Variability)
```python
data = [neat_results, dqn_results]
plt.boxplot(data, labels=['NEAT', 'DQN'])
plt.ylabel('Reward')
plt.title('Performance Distribution')
```

### 4. Scatter Plot (Correlation)
```python
plt.scatter(exploration_rates, success_rates)
plt.xlabel('Exploration Rate')
plt.ylabel('Success Rate')
plt.title('Exploration vs Success')
```

### 5. Confusion Matrix
```python
import seaborn as sns

# For action predictions
confusion = np.array([[TP, FP], [FN, TN]])
sns.heatmap(confusion, annot=True, fmt='d')
plt.title('Action Prediction Accuracy')
```

## Troubleshooting

### Plots Not Showing?
```python
# In Jupyter
%matplotlib inline

# In script
plt.show()

# Or save directly
plt.savefig('plot.png')
```

### Memory Issues?
```python
# Close figures after saving
plt.close('all')

# Clear figure
plt.clf()

# Reduce resolution
plt.savefig('plot.png', dpi=150)  # Instead of 300
```

### Slow Rendering?
```python
# Disable interactive mode
plt.ioff()

# Use faster backend
import matplotlib
matplotlib.use('Agg')
```

### Bad Quality?
```python
# Increase DPI
plt.savefig('plot.png', dpi=300)

# Use vector format
plt.savefig('plot.pdf')  # or .svg

# Set figure size appropriately
plt.figure(figsize=(10, 6))  # Width, Height in inches
```

## Best Practices

### ✅ Do
- Label axes clearly
- Add titles
- Include legends
- Use consistent colors
- Save high-resolution
- Add grid for readability
- Use error bars when appropriate

### ❌ Don't
- Overcrowd plots
- Use too many colors
- Forget axis labels
- Use default figure sizes
- Mix visualization styles
- Skip legends

## Quick Reference

```python
# Basic plot
plt.plot(x, y, label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x1, y1)
axes[0, 1].plot(x2, y2)
plt.tight_layout()

# Styling
plt.style.use('seaborn')  # or 'ggplot', 'bmh'
sns.set_palette("husl")

# Animation
anim = animation.FuncAnimation(fig, animate_func, frames=100)
anim.save('output.gif', writer='pillow', fps=10)

# Interactive
from analysis.visualize_training import TrainingVisualizer
viz = TrainingVisualizer()
viz.create_comparison_dashboard()
```

## Resources

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Examples](https://seaborn.pydata.org/examples/index.html)
- [Plotly Documentation](https://plotly.com/python/)
- [Animation Tutorial](https://matplotlib.org/stable/api/animation_api.html)

---

**Related:** [NEAT Tutorial](neat_tutorial.md) | [DQN Tutorial](dqn_tutorial.md) | [Environment Guide](environment_guide.md)
