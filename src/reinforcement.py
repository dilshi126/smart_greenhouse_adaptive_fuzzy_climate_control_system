"""
Reinforcement Learning Module for Fuzzy Rule Evolution
=======================================================
Implements Q-learning to automatically evolve and optimize fuzzy rules.

BONUS FEATURE: Reinforcement Learning
-------------------------------------
This module uses Q-learning to:
1. Learn optimal rule weights based on control performance
2. Adapt rules to changing environmental conditions
3. Evolve the fuzzy system over time

Q-LEARNING ALGORITHM:
=====================
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

Where:
- s: state (discretized temp, humidity, stage)
- a: action (rule weight adjustment)
- r: reward (negative error + energy bonus)
- α: learning rate
- γ: discount factor
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
import os

from .plant_database import PlantDatabase, GrowthStage
from .membership_functions import MembershipFunctions
from .mamdani_controller import MamdaniController
from .sugeno_controller import SugenoController


@dataclass
class RLState:
    """Represents a discretized state for Q-learning."""
    temp_level: int      # 0-4 (very_cold to hot)
    humidity_level: int  # 0-4 (very_dry to very_humid)
    stage_level: int     # 0-2 (seedling, vegetative, flowering)
    
    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.temp_level, self.humidity_level, self.stage_level)
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()


class FuzzyRLAgent:
    """
    Reinforcement Learning agent for fuzzy rule optimization.
    
    LEARNING APPROACH:
    ==================
    1. State: Discretized environmental conditions
    2. Actions: Adjust rule weights (increase/decrease/maintain)
    3. Reward: Based on control error, energy usage, and smoothness
    
    The agent learns which rule weight adjustments lead to better
    control performance in different environmental conditions.
    """
    
    def __init__(self, controller_type: str = "mamdani",
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.2,
                 num_rules: int = 30):
        """
        Initialize the RL agent.
        
        Args:
            controller_type: "mamdani" or "sugeno"
            learning_rate: α - how fast to learn
            discount_factor: γ - importance of future rewards
            epsilon: exploration rate for ε-greedy policy
            num_rules: number of fuzzy rules to optimize
        """
        self.controller_type = controller_type
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.num_rules = num_rules
        
        # State space: 5 temp levels × 5 humidity levels × 3 stages = 75 states
        self.num_states = 5 * 5 * 3
        
        # Action space: For each rule, can increase/decrease/maintain weight
        # Simplified: 3 actions per rule, but we'll use aggregate actions
        # Actions: 0=decrease_all, 1=maintain, 2=increase_all,
        #          3=increase_temp_rules, 4=increase_humidity_rules
        self.num_actions = 5
        
        # Initialize Q-table
        self.q_table = {}
        
        # Rule weights (initialized to 1.0)
        self.rule_weights = np.ones(num_rules)
        
        # Performance history
        self.episode_rewards = []
        self.episode_errors = []
        
        # Initialize controller
        self._init_controller()
    
    def _init_controller(self):
        """Initialize the fuzzy controller."""
        self.mf = MembershipFunctions()
        if self.controller_type == "mamdani":
            self.controller = MamdaniController(self.mf)
        else:
            self.controller = SugenoController(self.mf)
    
    def _discretize_state(self, temp: float, humidity: float, 
                          stage: float) -> RLState:
        """Convert continuous values to discrete state."""
        # Temperature: 0-9=very_cold, 10-17=cold, 18-27=optimal, 28-36=warm, 37+=hot
        if temp < 10:
            temp_level = 0
        elif temp < 18:
            temp_level = 1
        elif temp < 28:
            temp_level = 2
        elif temp < 37:
            temp_level = 3
        else:
            temp_level = 4
        
        # Humidity: 0-20=very_dry, 21-40=dry, 41-70=optimal, 71-85=humid, 86+=very_humid
        if humidity < 20:
            humidity_level = 0
        elif humidity < 40:
            humidity_level = 1
        elif humidity < 70:
            humidity_level = 2
        elif humidity < 85:
            humidity_level = 3
        else:
            humidity_level = 4
        
        # Stage: 0-33=seedling, 34-66=vegetative, 67+=flowering
        if stage < 33:
            stage_level = 0
        elif stage < 67:
            stage_level = 1
        else:
            stage_level = 2
        
        return RLState(temp_level, humidity_level, stage_level)

    def _get_q_value(self, state: RLState, action: int) -> float:
        """Get Q-value for state-action pair."""
        key = (state.to_tuple(), action)
        return self.q_table.get(key, 0.0)
    
    def _set_q_value(self, state: RLState, action: int, value: float):
        """Set Q-value for state-action pair."""
        key = (state.to_tuple(), action)
        self.q_table[key] = value
    
    def select_action(self, state: RLState, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: If True, use exploration; if False, exploit only
        
        Returns:
            Selected action index
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: best known action
            q_values = [self._get_q_value(state, a) for a in range(self.num_actions)]
            return int(np.argmax(q_values))
    
    def apply_action(self, action: int):
        """
        Apply action to modify rule weights.
        
        Actions:
        0: Decrease all weights by 5%
        1: Maintain current weights
        2: Increase all weights by 5%
        3: Increase temperature-related rules
        4: Increase humidity-related rules
        """
        adjustment = 0.05
        
        if action == 0:
            self.rule_weights *= (1 - adjustment)
        elif action == 1:
            pass  # Maintain
        elif action == 2:
            self.rule_weights *= (1 + adjustment)
        elif action == 3:
            # Increase first 15 rules (temperature-focused)
            self.rule_weights[:15] *= (1 + adjustment)
        elif action == 4:
            # Increase last 15 rules (humidity-focused)
            self.rule_weights[15:] *= (1 + adjustment)
        
        # Clip weights to valid range
        self.rule_weights = np.clip(self.rule_weights, 0.1, 2.0)
        
        # Apply weights to controller rules
        for i, rule in enumerate(self.controller.rules):
            if i < len(self.rule_weights):
                rule.weight = self.rule_weights[i]
    
    def compute_reward(self, temp: float, humidity: float,
                       optimal_temp: float, optimal_humidity: float,
                       heater_output: float, misting_output: float,
                       prev_heater: float = None, prev_misting: float = None) -> float:
        """
        Compute reward based on control performance.
        
        Reward components:
        1. Negative error (closer to optimal = higher reward)
        2. Energy efficiency bonus (lower output = bonus)
        3. Smoothness bonus (smaller changes = bonus)
        """
        # Error penalty
        temp_error = abs(temp - optimal_temp) / 45  # Normalized
        humidity_error = abs(humidity - optimal_humidity) / 100
        error_penalty = -(temp_error + humidity_error) * 10
        
        # Energy efficiency bonus
        heater_deviation = abs(heater_output - 50) / 50  # Deviation from neutral
        energy_penalty = -(heater_deviation + misting_output / 100) * 2
        
        # Smoothness bonus
        smoothness_bonus = 0
        if prev_heater is not None and prev_misting is not None:
            heater_change = abs(heater_output - prev_heater) / 100
            misting_change = abs(misting_output - prev_misting) / 100
            smoothness_bonus = -(heater_change + misting_change) * 3
        
        # Total reward
        reward = error_penalty + energy_penalty + smoothness_bonus
        
        # Bonus for being in optimal range
        if temp_error < 0.1 and humidity_error < 0.1:
            reward += 5
        
        return reward

    def train_episode(self, plant: str = "Tomato", 
                      stage: GrowthStage = GrowthStage.VEGETATIVE,
                      num_steps: int = 100) -> Dict[str, float]:
        """
        Train for one episode.
        
        Args:
            plant: Plant type
            stage: Growth stage
            num_steps: Steps per episode
        
        Returns:
            Dict with episode metrics
        """
        # Get optimal values
        requirements = PlantDatabase.get_requirements(plant, stage)
        optimal_temp = requirements.temp_optimal
        optimal_humidity = requirements.humidity_optimal
        stage_value = PlantDatabase.get_stage_numeric(stage)
        
        # Initialize episode
        total_reward = 0
        total_error = 0
        prev_heater = 50
        prev_misting = 25
        
        # Generate random starting conditions
        temp = optimal_temp + np.random.uniform(-10, 10)
        humidity = optimal_humidity + np.random.uniform(-20, 20)
        
        for step in range(num_steps):
            # Get current state
            state = self._discretize_state(temp, humidity, stage_value)
            
            # Select and apply action
            action = self.select_action(state, training=True)
            self.apply_action(action)
            
            # Get controller output
            result = self.controller.infer(temp, humidity, stage_value)
            heater = result['heater']
            misting = result['misting']
            
            # Compute reward
            reward = self.compute_reward(
                temp, humidity, optimal_temp, optimal_humidity,
                heater, misting, prev_heater, prev_misting
            )
            
            # Simulate environment response (simplified)
            # Heater affects temperature, misting affects humidity
            temp_change = (heater - 50) * 0.05  # Positive heater increases temp
            humidity_change = misting * 0.03 - 1  # Misting increases humidity
            
            # Add random disturbance
            temp += temp_change + np.random.normal(0, 0.5)
            humidity += humidity_change + np.random.normal(0, 1)
            
            # Clip to valid ranges
            temp = np.clip(temp, 0, 45)
            humidity = np.clip(humidity, 0, 100)
            
            # Get next state
            next_state = self._discretize_state(temp, humidity, stage_value)
            
            # Q-learning update
            current_q = self._get_q_value(state, action)
            max_next_q = max(self._get_q_value(next_state, a) 
                           for a in range(self.num_actions))
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self._set_q_value(state, action, new_q)
            
            # Track metrics
            total_reward += reward
            total_error += abs(temp - optimal_temp) + abs(humidity - optimal_humidity)
            prev_heater = heater
            prev_misting = misting
        
        # Record episode results
        avg_error = total_error / num_steps
        self.episode_rewards.append(total_reward)
        self.episode_errors.append(avg_error)
        
        return {
            'total_reward': total_reward,
            'avg_error': avg_error,
            'final_temp': temp,
            'final_humidity': humidity
        }

    def train(self, num_episodes: int = 500, 
              plants: List[str] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            plants: List of plants to train on (cycles through)
            verbose: Print progress
        
        Returns:
            Training history
        """
        if plants is None:
            plants = ["Tomato", "Lettuce", "Orchid"]
        
        stages = [GrowthStage.SEEDLING, GrowthStage.VEGETATIVE, GrowthStage.FLOWERING]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"REINFORCEMENT LEARNING TRAINING")
            print(f"Controller: {self.controller_type}")
            print(f"Episodes: {num_episodes}")
            print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Cycle through plants and stages
            plant = plants[episode % len(plants)]
            stage = stages[(episode // len(plants)) % len(stages)]
            
            # Decay epsilon over time
            self.epsilon = max(0.05, self.epsilon * 0.995)
            
            # Train episode
            metrics = self.train_episode(plant, stage)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_error = np.mean(self.episode_errors[-50:])
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Avg Reward = {avg_reward:.2f}, "
                      f"Avg Error = {avg_error:.2f}, "
                      f"ε = {self.epsilon:.3f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"Final Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"Final Avg Error: {np.mean(self.episode_errors[-100:]):.2f}")
            print(f"Q-table size: {len(self.q_table)} entries")
            print(f"{'='*60}\n")
        
        return {
            'rewards': self.episode_rewards,
            'errors': self.episode_errors
        }
    
    def get_optimized_weights(self) -> np.ndarray:
        """Get the learned rule weights."""
        return self.rule_weights.copy()
    
    def save(self, filepath: str):
        """Save the trained agent."""
        data = {
            'q_table': self.q_table,
            'rule_weights': self.rule_weights,
            'episode_rewards': self.episode_rewards,
            'episode_errors': self.episode_errors,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load a trained agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.rule_weights = data['rule_weights']
        self.episode_rewards = data['episode_rewards']
        self.episode_errors = data['episode_errors']
        self.epsilon = data['epsilon']
        
        # Apply loaded weights to controller
        for i, rule in enumerate(self.controller.rules):
            if i < len(self.rule_weights):
                rule.weight = self.rule_weights[i]
        
        print(f"Agent loaded from: {filepath}")

    def evaluate(self, plant: str = "Tomato",
                 stage: GrowthStage = GrowthStage.VEGETATIVE,
                 num_steps: int = 100) -> Dict[str, float]:
        """
        Evaluate the trained agent (no learning).
        
        Returns performance metrics.
        """
        requirements = PlantDatabase.get_requirements(plant, stage)
        optimal_temp = requirements.temp_optimal
        optimal_humidity = requirements.humidity_optimal
        stage_value = PlantDatabase.get_stage_numeric(stage)
        
        # Start at suboptimal conditions
        temp = optimal_temp + 10
        humidity = optimal_humidity - 15
        
        total_error = 0
        total_energy = 0
        outputs = []
        
        for step in range(num_steps):
            state = self._discretize_state(temp, humidity, stage_value)
            action = self.select_action(state, training=False)
            self.apply_action(action)
            
            result = self.controller.infer(temp, humidity, stage_value)
            heater = result['heater']
            misting = result['misting']
            
            outputs.append((heater, misting))
            
            # Track metrics
            total_error += abs(temp - optimal_temp) + abs(humidity - optimal_humidity)
            total_energy += abs(heater - 50) + misting
            
            # Simulate environment
            temp += (heater - 50) * 0.05 + np.random.normal(0, 0.3)
            humidity += misting * 0.03 - 1 + np.random.normal(0, 0.5)
            temp = np.clip(temp, 0, 45)
            humidity = np.clip(humidity, 0, 100)
        
        # Calculate smoothness
        heater_changes = [abs(outputs[i][0] - outputs[i-1][0]) 
                         for i in range(1, len(outputs))]
        smoothness = np.std(heater_changes)
        
        return {
            'avg_error': total_error / num_steps,
            'avg_energy': total_energy / num_steps,
            'smoothness': smoothness,
            'final_temp': temp,
            'final_humidity': humidity
        }


def compare_with_without_rl(num_episodes: int = 200):
    """
    Compare controller performance with and without RL optimization.
    """
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING COMPARISON")
    print("="*70)
    
    # Without RL
    print("\n[1] Baseline (No RL):")
    baseline_agent = FuzzyRLAgent(controller_type="mamdani")
    baseline_metrics = baseline_agent.evaluate()
    print(f"  Avg Error: {baseline_metrics['avg_error']:.4f}")
    print(f"  Avg Energy: {baseline_metrics['avg_energy']:.4f}")
    print(f"  Smoothness: {baseline_metrics['smoothness']:.4f}")
    
    # With RL
    print(f"\n[2] Training RL Agent ({num_episodes} episodes)...")
    rl_agent = FuzzyRLAgent(controller_type="mamdani")
    rl_agent.train(num_episodes=num_episodes, verbose=True)
    
    print("\n[3] After RL Training:")
    rl_metrics = rl_agent.evaluate()
    print(f"  Avg Error: {rl_metrics['avg_error']:.4f}")
    print(f"  Avg Energy: {rl_metrics['avg_energy']:.4f}")
    print(f"  Smoothness: {rl_metrics['smoothness']:.4f}")
    
    # Improvement
    print("\n[4] Improvement:")
    error_improvement = (baseline_metrics['avg_error'] - rl_metrics['avg_error']) / baseline_metrics['avg_error'] * 100
    energy_improvement = (baseline_metrics['avg_energy'] - rl_metrics['avg_energy']) / baseline_metrics['avg_energy'] * 100
    print(f"  Error Reduction: {error_improvement:.1f}%")
    print(f"  Energy Reduction: {energy_improvement:.1f}%")
    
    print("\n" + "="*70)
    
    return {
        'baseline': baseline_metrics,
        'rl_optimized': rl_metrics,
        'error_improvement': error_improvement,
        'energy_improvement': energy_improvement
    }


def plot_training_history(agent: FuzzyRLAgent, save_path: str = None):
    """Plot training history."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    ax = axes[0]
    ax.plot(agent.episode_rewards, alpha=0.3, color='blue')
    # Moving average
    window = 50
    if len(agent.episode_rewards) >= window:
        ma = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(agent.episode_rewards)), ma, color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards')
    ax.grid(True, alpha=0.3)
    
    # Errors
    ax = axes[1]
    ax.plot(agent.episode_errors, alpha=0.3, color='red')
    if len(agent.episode_errors) >= window:
        ma = np.convolve(agent.episode_errors, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(agent.episode_errors)), ma, color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Error')
    ax.set_title('Training Errors')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Demo: Train and evaluate RL agent
    results = compare_with_without_rl(num_episodes=300)
    
    # Train a full agent and plot
    print("\nTraining full agent for visualization...")
    agent = FuzzyRLAgent(controller_type="mamdani")
    agent.train(num_episodes=500, verbose=True)
    
    # Plot training history
    plot_training_history(agent, "rl_training_history.png")
    
    # Save trained agent
    agent.save("trained_rl_agent.pkl")
