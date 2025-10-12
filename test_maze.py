"""
Test if maze is solvable using simple BFS pathfinding.
Run: python test_maze.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.maze_env import MazeEnv
from collections import deque
import numpy as np

def bfs_solve_maze(env):
    """Use BFS to find if maze is solvable."""
    start = tuple(env.start_pos)
    goal = tuple(env.goal_pos)
    
    queue = deque([(start, [start])])
    visited = {start}
    
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        (y, x), path = queue.popleft()
        
        if (y, x) == goal:
            return True, path
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < env.height and 0 <= nx < env.width and
                env.maze[ny, nx] != env.WALL and (ny, nx) not in visited):
                
                visited.add((ny, nx))
                queue.append(((ny, nx), path + [(ny, nx)]))
    
    return False, []

def main():
    print("="*70)
    print("MAZE SOLVABILITY TEST")
    print("="*70)
    print()
    
    # Create environment
    env = MazeEnv()
    
    print("Maze Configuration:")
    print(f"  Size: {env.height}x{env.width}")
    print(f"  Start: {env.start_pos}")
    print(f"  Goal: {env.goal_pos}")
    print()
    
    # Visualize maze
    print("Maze Layout:")
    symbols = {0: '·', 1: '█', 2: '★', 3: '×', 4: '○'}
    for i in range(env.height):
        for j in range(env.width):
            cell = env.maze[i, j]
            if (i, j) == tuple(env.start_pos):
                print('S', end=' ')
            else:
                print(symbols.get(cell, '?'), end=' ')
        print()
    print()
    
    # Test solvability
    print("Testing solvability with BFS...")
    solvable, path = bfs_solve_maze(env)
    
    if solvable:
        print(f"✅ Maze is SOLVABLE!")
        print(f"   Optimal path length: {len(path)} steps")
        print(f"   Path exists from {env.start_pos} to {env.goal_pos}")
        print()
        
        # Show path
        print("Optimal path visualization:")
        path_set = set(path)
        for i in range(env.height):
            for j in range(env.width):
                if (i, j) in path_set:
                    print('•', end=' ')
                elif env.maze[i, j] == env.WALL:
                    print('█', end=' ')
                else:
                    print('·', end=' ')
            print()
        print()
        
        # Test with actual agent
        print("Testing with random agent...")
        obs, _ = env.reset()
        for step in range(len(path) * 2):  # Give it 2x optimal steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                if env.maze[int(env.agent_pos[0]), int(env.agent_pos[1])] == env.GOAL:
                    print(f"✅ Random agent reached goal in {step+1} steps!")
                    break
        else:
            print(f"⚠️  Random agent didn't reach goal (this is normal)")
        
        print()
        print("="*70)
        print("RECOMMENDATION: Maze is ready for training")
        print("="*70)
        print()
        print("Run training with:")
        print("  python train_agents.py --quick")
        
        return 0
        
    else:
        print(f"❌ Maze is NOT solvable!")
        print(f"   No path exists from {env.start_pos} to {env.goal_pos}")
        print()
        print("="*70)
        print("ERROR: Fix maze before training")
        print("="*70)
        print()
        print("Suggestions:")
        print("  1. Check if start and goal are connected")
        print("  2. Remove blocking walls")
        print("  3. Use simpler maze layout")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
