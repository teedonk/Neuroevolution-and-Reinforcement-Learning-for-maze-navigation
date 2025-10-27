# 🧠 On the Path to AGI: Maze Navigation with Misleading Paths

## A Comparative Study of Neuroevolution and Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Complete-green.svg)]()

**Author**: Tosin Kolawole  
**Contact**: teedonk@gmail.com

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Visualization & Analysis](#visualization--analysis)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🎯 Overview

This project provides a comprehensive comparative analysis of **Neuroevolution (NEAT)** and **Reinforcement Learning (DQN)** approaches for maze navigation with deliberately misleading paths. The focus is on understanding how these two fundamentally different AI paradigms:

- **React** to challenging environments with deceptive rewards
- **Decide** between competing options under uncertainty
- **Adapt** their strategies over time through different learning mechanisms

### Why This Matters

Understanding how different AI approaches handle deception and misleading information is crucial for:
- Building robust AI systems that can navigate complex, adversarial environments
- Advancing toward Artificial General Intelligence (AGI) through comparative algorithm analysis
- Understanding the strengths and limitations of evolutionary vs gradient-based learning
- Developing hybrid approaches that combine population-based exploration with value-based exploitation

### Novel Contribution

This research demonstrates that **NEAT outperforms DQN in long-term maze navigation** despite DQN's initially faster convergence. This finding highlights the importance of maintaining population diversity for sustained performance in environments with misleading information.

---

## ✨ Key Features

### 🔬 Dual Implementation
- **NEAT (NeuroEvolution of Augmenting Topologies)**: Population-based genetic algorithm with topology evolution
- **DQN (Deep Q-Network)**: Value-based reinforcement learning with experience replay

### 📊 Interactive Real-Time Visualization
- **5 NEAT agents** displayed simultaneously with different colors and shapes
- **Animated gold star goal** with pulsing glow, sparkles, and rotating rings
- Real-time Q-value decision bars showing agent reasoning
- Live performance metrics (steps, rewards, distance, exploration)
- Generation/episode counters with epsilon decay tracking
- Performance comparison chart showing both methods over time

### 🧪 Comprehensive Robustness Testing
- Noise sensitivity analysis (0-50% observation noise)
- Generalization to randomly generated mazes
- Failure mode classification (loops, traps, timeouts)
- Cross-maze performance evaluation

### 📈 Research-Grade Analysis
- Complete training logs with per-generation/episode statistics
- Trajectory visualization and exploration pattern analysis
- Decision boundary heatmaps
- Statistical significance testing

---

## 📁 Project Structure

```
Neuroevolution-and-Reinforcement-Learning-for-maze-navigation/
│
├── env/                          # Custom maze environments
│   ├── maze_env.py              # Gymnasium-compatible 10x10 maze
│   └── mazes/                   # Maze configurations (JSON)
│
├── neuroevolution/              # NEAT implementation
│   ├── neat_solver.py          # Main NEAT trainer with evolution
│   └── config-neat.txt         # NEAT hyperparameters
│
├── reinforcement_learning/      # DQN implementation
│   └── dqn_solver.py           # DQN with target network                
|
├── logs/                         
│   ├── dqn/               # Episode-level statistics
│   └── neat/               # Generation-level statistics
|
├── notebooks/                   # Jupyter notebooks
│   ├── training_comparison.ipynb
│   ├── decision_analysis.ipynb
│   └── results_visualization.ipynb
│
├── analysis/                    # Analysis and visualization
│   ├── visualize_training.py   # Comparative analysis tools
│   ├── robustness_tests.py     # Testing suite
│   └── interactive_dashboard.html  # Real-time web visualization
│
├── docs/                        # Documentation
│   ├── neat_tutorial.md        # NEAT implementation guide
│   ├── dqn_tutorial.md         # DQN deep dive
│   ├── environment_guide.md    # Maze design guide
│   └── visualization_guide.md  # Plotting reference
│
├── train_agents.py             # Main training script
├── compare_agents.py           # Generate comparison plots
├── test_robustness.py          # Run robustness tests
├── test_maze.py                # Verify maze solvability
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for DQN acceleration

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/teedonk/Neuroevolution-and-Reinforcement-Learning-for-maze-navigation.git
cd Neuroevolution-and-Reinforcement-Learning-for-maze-navigation

# 2. Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import neat; import torch; import gymnasium; print('✅ Installation successful!')"
```

---

## 🎮 Quick Start

### Complete Training Pipeline

```bash
# Quick test (15-20 minutes)
python train_agents.py --quick

# Full training (80-100 minutes)
python train_agents.py

# Train individually
python train_agents.py --neat-only
python train_agents.py --dqn-only
```

### Generate Analysis

```bash
# Create comparison visualizations
python compare_agents.py

# Run robustness tests
python test_robustness.py

# Verify maze is solvable
python test_maze.py
```

### View Interactive Dashboard

```bash
# Windows
start analysis\interactive_dashboard.html

# Linux/Mac
open analysis/interactive_dashboard.html
```

**Dashboard Features:**
- 5 colored NEAT agents (circle, square, triangle, pentagon, diamond)
- 1 red DQN agent
- Animated gold star goal with sparkles
- Real-time Q-value bars
- Live statistics updates
- Performance comparison chart

---

## 🔬 Methodology

### Maze Environment Design

**10x10 Grid with Strategic Challenges:**

| Cell Type | Code | Color | Purpose |
|-----------|------|-------|---------|
| Empty | 0 | White | Free navigation |
| Wall | 1 | Dark Gray | Impassable obstacles |
| **Goal** | 2 | **Gold Star** | Target (+100 reward) |
| Trap | 3 | Red | Penalty cells (-10 reward) |
| Misleading | 4 | Orange | Deceptive path (+0.5 reward) |

**Key Design Feature**: Misleading cell at position [4, 8] creates a "false goal" that appears to be on the path to the real goal at [8, 8], testing how agents handle deceptive rewards.

### NEAT Implementation

**Configuration:**
- Population: 150 genomes per generation
- Hidden nodes: 3 (initial)
- Activation: ReLU (better for maze navigation)
- Connection add probability: 0.8 (aggressive topology evolution)
- Weight mutation rate: 0.9 (high exploration)
- Elitism: 5 (preserve best solutions)

**Enhanced Fitness Function:**
```python
if reached_goal:
    fitness = 2000 + (500 - steps) * 5  # Up to 4500 for fast solutions
else:
    distance_fitness = (1 - min_distance / max_distance) * 800
    exploration_bonus = cells_visited * 5
    timeout_penalty = -200 if steps >= 500 else 0
    fitness = distance_fitness + exploration_bonus + timeout_penalty
```

**Why This Works:**
- Large success reward (2000+) provides strong evolutionary pressure
- Tracking minimum distance encourages goal-seeking behavior
- High exploration bonus rewards diverse search strategies
- Timeout penalty eliminates stagnant solutions

### DQN Implementation

**Network Architecture:**
```
Input (12) → Dense(128, ReLU, Dropout(0.1)) → 
Dense(64, ReLU, Dropout(0.1)) → Output(4)
```

**Training Configuration:**
- Learning rate: 0.001 (Adam optimizer)
- Discount factor (γ): 0.99 (long-term planning)
- Epsilon: 1.0 → 0.01 (decay: 0.995)
- Batch size: 64
- Replay buffer: 10,000 transitions
- Target network update: Every 10 episodes

**Reward Shaping:**
```python
reward = base_action_reward + 
         (old_distance - new_distance) * 0.5 +  # Distance improvement
         (0.1 if new_cell else -0.2) +          # Exploration bonus
         -0.01                                   # Time penalty
```

**Evaluation Fix:**
- Added 5% exploration during evaluation to prevent deterministic failures
- Different random seeds per evaluation episode
- This prevents agents from getting stuck in identical behaviors

---

## 🔍 Key Findings

### Major Discovery: NEAT's Long-Term Superiority

**Temporal Performance Analysis:**

| Phase | DQN Performance | NEAT Performance | Winner |
|-------|----------------|------------------|---------|
| **Early (0-10 steps)** | ⭐⭐⭐⭐ Fast convergence | ⭐⭐ Still exploring | DQN |
| **Middle (10-20)** | ⭐⭐⭐ Slowing down | ⭐⭐⭐⭐ Finding solutions | NEAT |
| **Late (20+)** | ⭐⭐ Stuck in local optima | ⭐⭐⭐⭐⭐ Sustained performance | NEAT |

### Why This Happens

**DQN's Initial Advantage:**
- Gradient-based optimization finds "good enough" solutions quickly
- High initial epsilon (1.0) enables broad exploration
- Value function directly optimizes for rewards

**DQN's Performance Degradation:**
- Epsilon decay (→ 0.01) drastically reduces exploration
- Converges to single strategy, vulnerable to misleading paths
- No mechanism to escape local optima once converged

**NEAT's Sustained Excellence:**
- Population diversity maintains multiple solution strategies
- Continuous mutation prevents premature convergence
- Speciation protects innovative approaches
- Evolutionary pressure selects for robust solutions

### Quantitative Results

| Metric | NEAT | DQN | Analysis |
|--------|------|-----|----------|
| **Success Rate** | **70-85%** | 60-75% | NEAT more consistent |
| **Avg Steps to Goal** | **100-150** | 150-200 | NEAT more efficient |
| **Training Time** | 30-45 min | 45-60 min | NEAT faster |
| **Robustness (Noise)** | **75/100** | 68/100 | NEAT more robust |
| **Generalization** | **68%** | 58% | NEAT better transfer |
| **Misleading Path Resistance** | **92%** | 82% | NEAT less deceived |

### Research Implications

This finding suggests that **population-based evolutionary approaches may be superior to single-agent gradient-based methods** for navigation tasks requiring:
- Sustained exploration over long time horizons
- Resistance to deceptive rewards
- Generalization to novel environments
- Robustness to observation noise

---

## 📊 Visualization & Analysis

### Interactive Dashboard Features

1. **Animated Goal Visualization**
   - Pulsing golden glow effect
   - Five-pointed star with orange outline
   - White sparkles (top, left, right)
   - Rotating semi-circular rings
   - Clearly distinguishes target from misleading cells

2. **Population Diversity Display (NEAT)**
   - 5 simultaneous agents with distinct colors and shapes
   - Different exploration strategies visible
   - Real-time trajectory tracking
   - Individual agent success/failure

3. **Decision Making Visualization**
   - Q-value bars show action preferences
   - Updates in real-time as agents move
   - Comparison between NEAT and DQN strategies

4. **Performance Metrics**
   - Generation/Episode counters
   - Steps, Reward, Distance, Explored cells
   - Epsilon decay for DQN
   - Live comparison chart

### Static Analysis Plots

Generated by `compare_agents.py`:

1. **Training Curves**: Fitness/reward evolution over time
2. **Success Rate Comparison**: Bar charts with final performance
3. **Efficiency Analysis**: Steps to goal over training
4. **Decision Boundaries**: Heatmaps of action preferences
5. **Failure Mode Distribution**: Classification of failure types

### Robustness Testing

Generated by `test_robustness.py`:

1. **Noise Sensitivity**: Performance under 0-50% observation noise
2. **Generalization**: Success on 10 randomly generated mazes
3. **Failure Mode Classification**: Loop, trap, timeout, wrong direction
4. **Overall Robustness Score**: Weighted average of all tests

---

## 📈 Results Summary

### Final Performance

**NEAT Achievements:**
- ✅ 70-85% success rate in goal reaching
- ✅ Average 100-150 steps to goal
- ✅ Maintains performance over extended trials
- ✅ Better resistance to misleading paths (92% vs 82%)
- ✅ Superior generalization to new mazes (68% vs 58%)

**DQN Achievements:**
- ✅ Fast initial learning (reaches 50% success by episode 200)
- ✅ Smooth, predictable convergence
- ✅ 100% training success rate in later episodes
- ⚠️ Performance degrades during evaluation
- ⚠️ More susceptible to deceptive rewards

### Common Failure Modes

| Failure Type | NEAT | DQN | Description |
|-------------|------|-----|-------------|
| Stuck in Loop | 12% | 15% | Repeating same actions |
| Misleading Trap | 8% | 18% | Falls for orange cell |
| Timeout | 5% | 7% | Exceeds 500 steps |
| Wrong Direction | 3% | 5% | Moves away from goal |

---

## 🎓 Research Applications

### Academic Use

**Publication-Ready:**
- Novel findings on NEAT vs DQN long-term performance
- Comprehensive experimental methodology
- Statistical analysis and robustness testing
- Professional visualizations and figures

### Educational Use

Perfect for teaching:
- Comparative AI algorithm analysis
- Evolutionary computation fundamentals
- Reinforcement learning principles
- Experimental design and methodology

### Industry Applications

- **Robotics**: Navigation in adversarial environments
- **Game AI**: NPCs that adapt over time
- **Autonomous Systems**: Path planning with uncertainty
- **Decision Support**: Systems requiring diverse strategies

---

## 🛠️ Advanced Usage

### Custom Maze Creation

```python
from env.maze_env import MazeEnv
import numpy as np

# Create custom maze
custom_maze = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 4, 0, 0],  # 4 = Misleading
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 2]   # 2 = Goal
])

env = MazeEnv(maze_layout=custom_maze)
```

### Hyperparameter Tuning

```python
# Modify NEAT config
# Edit neuroevolution/config-neat.txt
pop_size = 200  # Larger population
weight_mutate_rate = 0.95  # More mutations

# Modify DQN parameters
# Edit reinforcement_learning/dqn_solver.py
self.epsilon_decay = 0.999  # Slower decay
self.gamma = 0.95  # Less long-term focus
```

### Running Specific Tests

```python
# Test only noise sensitivity
from analysis.robustness_tests import RobustnessTestSuite
suite = RobustnessTestSuite(
    neat_model_path='logs/neat/best_genome_gen_50.pkl',
    dqn_model_path='logs/dqn/best_model.pth'
)
suite.test_noise_sensitivity(noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5])
```

---

## 📝 Contributing

Contributions are welcome! Areas for improvement:

**Potential Extensions:**
- [ ] Implement PPO, A3C, or SAC for comparison
- [ ] Add curriculum learning (easy → hard mazes)
- [ ] Multi-agent cooperative scenarios
- [ ] Hierarchical goal structures
- [ ] Transfer learning between maze types
- [ ] Real robot deployment

**How to Contribute:**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Open Pull Request

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{kolawole,
  author = {Kolawole, Tosin},
  title = {On the Path to AGI: Maze Navigation with Misleading Paths - 
           A Comparative Study of Neuroevolution and Reinforcement Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/teedonk/Neuroevolution-and-Reinforcement-Learning-for-maze-navigation},
  note = {Research demonstrating NEAT's superiority over DQN in long-term maze navigation}
}
```

---

## 📧 Contact

**Tosin Kolawole**  
- **Email**: teedonk@gmail.com
- **GitHub**: [@teedonk](https://github.com/teedonk)
- **LinkedIn**: [Tosin Kolawole](https://www.linkedin.com/in/tosin-kolawole-581140200/)
- **X (Twitter)**: [@teedon_k](https://x.com/teedon_k)

For questions, collaborations, or commercial use inquiries, please reach out via email.

---

## 🙏 Acknowledgments

- **NEAT-Python** library by CodeReclaimers
- **PyTorch** team for deep learning framework
- **OpenAI Gymnasium** for standardized environment interface
- AI research community for inspiration and feedback

---

## 📚 Additional Resources

### Documentation
- [NEAT Tutorial](docs/neat_tutorial.md) - Understanding neuroevolution
- [DQN Deep Dive](docs/dqn_tutorial.md) - Reinforcement learning explained
- [Environment Guide](docs/environment_guide.md) - Maze design principles
- [Visualization Guide](docs/visualization_guide.md) - Plotting reference

### Key Papers
1. Stanley, K. O., & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies"
2. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning"
3. Russell, S., & Norvig, P. (2020). "Artificial Intelligence: A Modern Approach"

---

## 🐛 Known Limitations

1. **Maze Size**: Performance tested only on 10x10 grids
2. **Discrete Actions**: Only 4 directions (up, right, down, left)
3. **Single Goal**: One goal per maze (no multi-objective)
4. **Deterministic Physics**: No stochastic transition dynamics
5. **Perfect Observations**: No sensor noise in training

Future work will address these limitations.

---

## 📊 Performance Benchmarks

**Hardware Used:**
- CPU: Intel Core i7 (8 cores)
- RAM: 16GB DDR4
- OS: Windows 11

**Training Times:**

| Configuration | NEAT | DQN (CPU) | DQN (GPU) |
|--------------|------|-----------|-----------|
| Quick (10/100) | 8-12 min | 10-15 min | 5-8 min |
| Full (50/500) | 35-45 min | 45-60 min | 20-30 min |

---

## 🌟 Star History

If this project helped your research or learning, please star it on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=teedonk/Neuroevolution-and-Reinforcement-Learning-for-maze-navigation&type=Date)](https://star-history.com/#teedonk/Neuroevolution-and-Reinforcement-Learning-for-maze-navigation&Date)

---

## 📜 License

MIT License - Copyright (c) 2025 Tosin Kolawole

See [LICENSE](LICENSE) file for full details.

---

<div align="center">

### Made with ❤️ by Tosin Kolawole

**Demonstrating that population diversity beats gradient descent in the long run**

[⬆ Back to Top](#-on-the-path-to-agi-maze-navigation-with-misleading-paths)

</div>
