# DQN Deep Dive

## What is DQN?

**Deep Q-Network** - combines Q-learning with deep neural networks. Learns to estimate the value (Q-value) of taking actions in different states.

## Core Idea

```
Q(state, action) = Expected future reward
```

Agent learns: "How good is action A in state S?"

## Key Components

### 1. Q-Network
```python
Input (12) → [128, ReLU] → [64, ReLU] → Output (4)
         ↓
    Q-values for each action
```

### 2. Experience Replay
```python
# Store transitions
memory = [(s, a, r, s', done), ...]

# Learn from random samples
batch = memory.sample(64)
```

**Why?** Breaks correlation between consecutive samples.

### 3. Target Network
```python
policy_net  # Updated every step
target_net  # Updated every 10 episodes
```

**Why?** Stabilizes training by providing consistent targets.

### 4. ε-Greedy Exploration
```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = best_q_action()  # Exploit
```

Epsilon: 1.0 → 0.01 over training

## The Algorithm

### Training Loop
```python
for episode in episodes:
    state = env.reset()
    
    for step in steps:
        # 1. Choose action
        action = epsilon_greedy(state)
        
        # 2. Take action
        next_state, reward, done = env.step(action)
        
        # 3. Store transition
        memory.push(state, action, reward, next_state, done)
        
        # 4. Learn from batch
        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            loss = compute_td_loss(batch)
            optimize(loss)
        
        # 5. Update target network
        if episode % 10 == 0:
            target_net.copy(policy_net)
        
        state = next_state
```

### Loss Function (TD Error)
```python
# Current Q-value
Q_current = policy_net(state)[action]

# Target Q-value
Q_target = reward + gamma * max(target_net(next_state))

# Loss
loss = (Q_current - Q_target)²
```

## Hyperparameters

### Learning Rate
```
Low (0.0001): Stable, slow
Medium (0.001): Balanced ✓
High (0.01): Fast, unstable
```

### Discount Factor (γ)
```
0.9: Short-term focus
0.99: Long-term planning ✓
0.999: Very long-term (can be unstable)
```

### Epsilon Decay
```
Fast (0.99): Quick exploitation
Medium (0.995): Balanced ✓
Slow (0.999): Extended exploration
```

### Batch Size
```
Small (32): Noisy updates, faster
Medium (64): Balanced ✓
Large (128): Stable, slower
```

### Replay Buffer
```
Small (1000): Recent experience only
Medium (10000): Good memory ✓
Large (100000): Diverse experience, more RAM
```

## Improvements in This Project

### 1. Reward Shaping
```python
# Not just sparse goal reward
reward = base + distance_improvement + exploration - time_penalty
```

### 2. Dropout Regularization
```python
layers = [
    Linear(128),
    ReLU(),
    Dropout(0.1),  # Prevents overfitting
    ...
]
```

### 3. Gradient Clipping
```python
loss.backward()
clip_grad_norm_(parameters, max_norm=1.0)  # Prevents exploding gradients
optimizer.step()
```

## Debugging Guide

### Not Learning?
```python
# Check if Q-values changing
print(f"Q-values: {q_values.mean():.2f}")

# Verify loss decreasing
print(f"Loss: {loss.item():.4f}")

# Ensure exploration happening
print(f"Epsilon: {epsilon:.3f}")
```

### Unstable Training?
- Lower learning rate: `lr = 0.0005`
- Increase target update frequency: `freq = 20`
- Smaller batch size: `batch = 32`

### Overfitting?
- Add more dropout: `p = 0.2`
- Increase replay buffer: `size = 20000`
- Use regularization: `weight_decay = 1e-5`

### Too Slow?
- Use GPU: `device = 'cuda'`
- Reduce network size: `hidden = [64, 32]`
- Smaller buffer: `size = 5000`

## Advantages

✅ **Proven method** - works on many domains  
✅ **Sample efficient** - reuses experiences  
✅ **Stable training** - target network + replay  
✅ **Continuous learning** - online updates  
✅ **GPU acceleration** - fast with CUDA

## Disadvantages

❌ **Overestimation bias** - Q-values often too high  
❌ **Brittle** - sensitive to hyperparameters  
❌ **Exploration challenge** - ε-greedy is simple  
❌ **Discrete actions only** - can't handle continuous  
❌ **Correlation issues** - even with replay

## Variants to Try

### Double DQN
```python
# Use policy net to select action
action = policy_net(next_state).argmax()

# Use target net to evaluate
Q_target = reward + gamma * target_net(next_state)[action]
```

**Benefit:** Reduces overestimation

### Dueling DQN
```python
# Split network into value and advantage streams
V(s) = state_value_stream(features)
A(s,a) = advantage_stream(features)

Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
```

**Benefit:** Better state value estimation

### Prioritized Replay
```python
# Sample high-error transitions more often
priority = abs(td_error) + epsilon
batch = memory.sample(batch_size, priorities)
```

**Benefit:** Learns from important experiences faster

## Code Example

```python
from dqn_solver import DQNMazeSolver
from env.maze_env import MazeEnv

# Setup
env = MazeEnv()
solver = DQNMazeSolver(env, log_dir='logs/dqn')

# Train
solver.train(num_episodes=500, verbose=True)

# Evaluate
results = solver.evaluate(num_episodes=10)

# Visualize
solver.visualize_training()
```

## Quick Tips

### Faster Convergence
```python
solver.gamma = 0.95          # Less long-term
solver.epsilon_decay = 0.99  # Exploit sooner
solver.lr = 0.005            # Learn faster
```

### More Stable
```python
solver.gamma = 0.99          # More planning
solver.epsilon_decay = 0.995 # Explore longer
solver.target_update = 20    # Less frequent updates
```

### Better Exploration
```python
solver.epsilon_min = 0.05    # Keep exploring
solver.epsilon_decay = 0.999 # Decay slower
```

## Common Mistakes

### ❌ No Target Network
- Training becomes unstable
- Q-values explode or collapse

### ❌ Too Small Replay Buffer
- Overfits to recent experiences
- Forgets important lessons

### ❌ Wrong Reward Scale
- If rewards > 100: normalize
- If sparse: add shaping

### ❌ Learning Too Fast
- High LR causes oscillation
- Reduce to 0.0001 or lower

### ❌ Not Enough Exploration
- Gets stuck in local optima
- Increase epsilon_min or decay slower

## Monitoring Training

### Key Metrics
```python
# Should increase
avg_reward = mean(episode_rewards[-100:])

# Should decrease  
loss = mean(losses[-100:])

# Should decrease
epsilon = current_epsilon

# Should increase
success_rate = wins / total_episodes
```

### Good Signs
- Reward trending upward
- Loss trending downward
- Success rate improving
- Q-values stabilizing

### Bad Signs
- Reward not improving after 200 episodes
- Loss staying high or increasing
- Q-values exploding (>1000)
- No successful episodes

## Further Reading

- [Original DQN Paper](https://www.nature.com/articles/nature14236)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298) - Combines improvements
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

---

**Next:** [Environment Design](environment_guide.md) | [Visualization Guide](visualization_guide.md)
