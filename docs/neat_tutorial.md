# NEAT Explained

## What is NEAT?

**NeuroEvolution of Augmenting Topologies** - evolves neural networks using genetic algorithms. Instead of training weights via backpropagation, it evolves both network structure and weights through natural selection.

## Core Concepts

### 1. Genome = Neural Network
- Each genome encodes a neural network
- Genes define nodes and connections
- Networks start simple, grow complex

### 2. Evolution Process
```
Generation N:
  ├── Evaluate fitness (solve maze)
  ├── Select best performers
  ├── Create offspring (mutation + crossover)
  └── Repeat
```

### 3. Key Innovations

**Speciation**: Groups similar networks together
- Protects innovation
- Allows new structures to develop
- Prevents premature convergence

**Historical Markings**: Tracks gene ancestry
- Enables meaningful crossover
- Aligns genes from parents
- Preserves building blocks

**Start Minimal**: Networks begin with no hidden layers
- Complexity emerges as needed
- Avoids unnecessary computation
- Natural regularization

## How It Works in This Project

### Initialization
```python
# Population: 150 networks
# Structure: Input (12) → Output (4)
# No hidden layers initially
```

### Fitness Function
```python
if reached_goal:
    fitness = 100 + (500 - steps)  # Reward speed
else:
    fitness = distance_fitness + exploration_bonus
```

### Mutations
- **Weight mutation** (80%): Tweak existing connections
- **Add connection** (50%): Wire new nodes together
- **Add node** (30%): Insert neuron in connection
- **Change activation** (5%): Switch function type

### Selection
- Top genomes survive (elitism)
- Species compete for resources
- Weak species eliminated
- Best genome of each species protected

## Configuration Tuning

### Population Size
```
Small (50-100): Faster, less diversity
Medium (150-200): Balanced ✓
Large (300+): Slow, more exploration
```

### Mutation Rates
```
Conservative: weight=0.5, structure=0.2
Balanced: weight=0.8, structure=0.5 ✓
Aggressive: weight=0.9, structure=0.8
```

### Compatibility Threshold
```
Low (2.0): Many species, slower
Medium (3.0): Balanced ✓
High (5.0): Few species, faster convergence
```

## Advantages

✅ **No gradient needed** - works where backprop fails  
✅ **Discovers topology** - finds optimal architecture  
✅ **Diverse solutions** - multiple strategies emerge  
✅ **Good exploration** - avoids local optima  
✅ **Robust** - handles noise well

## Disadvantages

❌ **Slow** - evaluates many networks  
❌ **No guarantees** - stochastic process  
❌ **Hard to parallelize** - Python GIL limits  
❌ **Memory intensive** - stores whole population  
❌ **Tuning required** - many hyperparameters

## When to Use NEAT

**Best for:**
- Problems where topology matters
- No clear network architecture
- Need diverse solutions
- Small to medium problems

**Avoid for:**
- Very large state spaces
- Need fast training
- Limited compute resources
- Continuous actions

## Quick Tips

### Speed Up Training
```python
# Reduce population
config.pop_size = 100

# Stricter selection
config.survival_threshold = 0.1

# Faster stagnation limit
config.max_stagnation = 10
```

### Improve Exploration
```python
# Increase mutation
config.weight_mutate_rate = 0.9
config.conn_add_prob = 0.7

# More species
config.compatibility_threshold = 2.5
```

### Better Convergence
```python
# Higher elitism
config.elitism = 5

# Stronger selection
config.survival_threshold = 0.3
```

## Debugging

### Poor Performance?
1. Check fitness function - is it rewarding right behavior?
2. Increase population size
3. Lower compatibility threshold (more species)
4. Increase mutation rates

### Premature Convergence?
1. Increase species protection (lower threshold)
2. Raise stagnation limit
3. Increase population diversity

### Too Slow?
1. Reduce population size
2. Simplify fitness evaluation
3. Use multiprocessing
4. Reduce max generations

## Code Example

```python
from neat_solver import NEATMazeSolver, create_neat_config

# Create config
config_path = create_neat_config('config-neat.txt')

# Initialize solver
solver = NEATMazeSolver(config_path)

# Train
winner = solver.train(generations=50)

# Evaluate
results = solver.evaluate_best(num_episodes=10)

# Visualize
solver.visualize_training()
```

## Further Reading

- [Original NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [NEAT-Python Docs](https://neat-python.readthedocs.io/)
- [Tutorial Videos](https://www.youtube.com/results?search_query=NEAT+algorithm)

---

**Next:** [DQN Deep Dive](dqn_tutorial.md) | [Environment Design](environment_guide.md)
