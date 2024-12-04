# Reinforcement Learning Algorithms Implementation

This project implements three different reinforcement learning algorithms to solve a decision-making problem: Monte Carlo, Q-Learning, and Value Iteration. Each algorithm approaches the problem of finding optimal policies in a stochastic environment with different methodologies.

## Problem Structure

The environment consists of states representing different time slots and locations (e.g., "RU_8p" for Regular Upstairs at 8pm), with possible actions:
- P (Practice)
- R (Rest)
- S (Study)

Each state-action pair has associated rewards and transition probabilities to next states.

## Algorithms

### 1. Monte Carlo Implementation (monteCarlo.py)

This implementation uses first-visit Monte Carlo methods to learn state values through experience.

Key features:
- Generates complete episodes through random action selection
- Updates state values based on actual returns
- Uses a learning rate (alpha) to gradually adjust state values
- Provides detailed episode tracking and final state values

Parameters:
- `num_episodes`: Number of episodes to run (default: 50)
- `alpha`: Learning rate (default: 0.1)

### 2. Q-Learning Implementation (q-Learning.py)

Implements the Q-learning algorithm, which learns action-values (Q-values) through temporal difference learning.

Key features:
- Off-policy learning algorithm
- Uses epsilon-greedy exploration strategy
- Implements learning rate and exploration rate decay
- Continues until Q-values converge

Parameters:
- `alpha`: Learning rate (initial: 0.2)
- `gamma`: Discount factor (0.99)
- `epsilon`: Initial exploration rate (1.0)
- `epsilon_decay`: Exploration decay rate (0.995)
- `alpha_decay`: Learning rate decay (0.995)

### 3. Value Iteration Implementation (valueInteraction.py)

Implements the value iteration algorithm to find optimal policies through dynamic programming.

Key features:
- Computes state values iteratively
- Uses a discount factor for future rewards
- Continues until values converge
- Determines optimal policy based on final values

Parameters:
- `gamma`: Discount factor (0.99)
- `theta`: Convergence threshold (0.001)

## Usage

Each algorithm can be run independently:

```python
# For Monte Carlo
python monteCarlo.py

# For Q-Learning
python q-Learning.py

# For Value Iteration
python valueInteraction.py
```

## Output

Each algorithm provides detailed output including:
- Episode-by-episode learning progress
- State values/Q-values
- Optimal policies
- Convergence information

## State Format

States are formatted as follows:
- First letter: R (Regular) or T (Tutorial)
- Second letter: U (Upstairs) or D (Downstairs)
- Time: 8a (8 AM), 10a (10 AM), 8p (8 PM), 10p (10 PM)

## Authors

- Bungein J Cheng

## Implementation Notes

- All implementations handle stochastic transitions
- Terminal states are handled appropriately in each algorithm
- State-action pairs with no defined transitions are excluded from consideration
- Each algorithm implements appropriate stopping conditions for convergence