"""
PSO Optimizer Module
====================
Implements Particle Swarm Optimization for fuzzy membership function tuning.

Part 6: Optimization
--------------------
Uses PSO to optimize membership function parameters to minimize:
- Control error (deviation from setpoints)
- Energy usage
- Output oscillation

PSO ALGORITHM:
==============
1. Initialize swarm of particles (candidate solutions)
2. Each particle represents a set of MF parameters
3. Evaluate fitness using simulation
4. Update particle velocities and positions
5. Track personal best and global best
6. Iterate until convergence

WHY PSO:
========
- Works well with continuous parameter spaces
- No gradient required (fuzzy systems are non-differentiable)
- Good balance between exploration and exploitation
- Easy to implement and tune
- Handles multi-objective optimization naturally
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import copy

from .membership_functions import MembershipFunctions, FuzzySet
from .mamdani_controller import MamdaniController
from .sugeno_controller import SugenoController
from .plant_database import PlantDatabase, GrowthStage


@dataclass
class Particle:
    """Represents a particle in the swarm."""
    position: np.ndarray      # Current parameter values
    velocity: np.ndarray      # Current velocity
    best_position: np.ndarray # Personal best position
    best_fitness: float       # Personal best fitness
    
    def __init__(self, dimensions: int, bounds: Tuple[np.ndarray, np.ndarray]):
        """Initialize particle with random position within bounds."""
        lower, upper = bounds
        self.position = lower + np.random.random(dimensions) * (upper - lower)
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')


class PSOOptimizer:
    """
    Particle Swarm Optimization for fuzzy system parameter tuning.
    
    OPTIMIZATION TARGET:
    ====================
    Minimize a weighted combination of:
    - Control error: How well the system maintains setpoints
    - Energy usage: Total control effort
    - Smoothness: Output variation (oscillation)
    
    PARAMETERS OPTIMIZED:
    =====================
    - Temperature MF centers and widths
    - Humidity MF centers and widths
    - (Optionally) Rule weights
    
    PSO PARAMETERS:
    ===============
    - w: Inertia weight (balances exploration/exploitation)
    - c1: Cognitive coefficient (personal best attraction)
    - c2: Social coefficient (global best attraction)
    """
    
    def __init__(self, controller_type: str = "mamdani",
                 num_particles: int = 20,
                 max_iterations: int = 50,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO optimizer.
        
        Args:
            controller_type: "mamdani" or "sugeno"
            num_particles: Swarm size
            max_iterations: Maximum iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        self.controller_type = controller_type
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parameter bounds
        self._setup_parameter_space()
        
        # Optimization history
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_params': []
        }
    
    def _setup_parameter_space(self):
        """Define the parameter space to optimize."""
        # We optimize the centers of the 'optimal' MFs for temp and humidity
        # and the widths of these MFs
        
        # Parameters: [temp_optimal_center, temp_optimal_width,
        #              humidity_optimal_center, humidity_optimal_width]
        
        self.param_names = [
            'temp_optimal_center',
            'temp_optimal_width',
            'humidity_optimal_center', 
            'humidity_optimal_width'
        ]
        
        self.dimensions = len(self.param_names)
        
        # Bounds for each parameter
        self.lower_bounds = np.array([15, 3, 50, 5])
        self.upper_bounds = np.array([30, 10, 80, 15])
        
        self.bounds = (self.lower_bounds, self.upper_bounds)
    
    def _decode_parameters(self, position: np.ndarray) -> Dict:
        """Convert particle position to MF parameters."""
        return {
            'temp_optimal_center': position[0],
            'temp_optimal_width': position[1],
            'humidity_optimal_center': position[2],
            'humidity_optimal_width': position[3]
        }
    
    def _apply_parameters(self, mf: MembershipFunctions, params: Dict):
        """Apply optimized parameters to membership functions."""
        # Update temperature optimal MF
        center = params['temp_optimal_center']
        width = params['temp_optimal_width']
        mf.temp_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular',
            (center - width, center, center + width)
        )
        
        # Update humidity optimal MF
        center = params['humidity_optimal_center']
        width = params['humidity_optimal_width']
        mf.humidity_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular',
            (center - width, center, center + width)
        )
    
    def _evaluate_fitness(self, position: np.ndarray, 
                         plant: str, stage: GrowthStage) -> float:
        """
        Evaluate fitness of a parameter set.
        
        Runs a short simulation and computes weighted fitness score.
        """
        params = self._decode_parameters(position)
        
        # Create controller with these parameters
        mf = MembershipFunctions()
        self._apply_parameters(mf, params)
        
        if self.controller_type == "mamdani":
            controller = MamdaniController(mf)
        else:
            controller = SugenoController(mf)
        
        # Get plant requirements
        requirements = PlantDatabase.get_requirements(plant, stage)
        optimal_temp = requirements.temp_optimal
        optimal_humidity = requirements.humidity_optimal
        
        # Run mini-simulation
        num_samples = 50
        temps = np.random.normal(optimal_temp, 5, num_samples)
        humidities = np.random.normal(optimal_humidity, 10, num_samples)
        stage_value = PlantDatabase.get_stage_numeric(stage)
        
        total_error = 0
        total_energy = 0
        outputs_heater = []
        outputs_misting = []
        
        for i in range(num_samples):
            result = controller.infer(temps[i], humidities[i], stage_value)
            
            heater = result['heater']
            misting = result['misting']
            
            outputs_heater.append(heater)
            outputs_misting.append(misting)
            
            # Error: deviation from neutral when at optimal conditions
            temp_error = abs(temps[i] - optimal_temp)
            humidity_error = abs(humidities[i] - optimal_humidity)
            
            # Expected output: closer to neutral (50) when error is low
            expected_heater = 50  # Neutral
            expected_misting = 25  # Low maintenance
            
            if temp_error < 3 and humidity_error < 5:
                # Near optimal - should have low output
                total_error += abs(heater - expected_heater) + abs(misting - expected_misting)
            else:
                # Away from optimal - should respond proportionally
                total_error += max(0, 50 - abs(heater - 50)) + max(0, 25 - misting)
            
            # Energy usage
            total_energy += abs(heater - 50) + misting
        
        # Smoothness (output variation)
        smoothness = np.std(np.diff(outputs_heater)) + np.std(np.diff(outputs_misting))
        
        # Weighted fitness (lower is better)
        fitness = (
            0.4 * (total_error / num_samples) +
            0.3 * (total_energy / num_samples) +
            0.3 * smoothness
        )
        
        return fitness
    
    def optimize(self, plant: str = "Tomato", 
                stage: GrowthStage = GrowthStage.VEGETATIVE,
                verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run PSO optimization.
        
        Args:
            plant: Plant type for fitness evaluation
            stage: Growth stage for fitness evaluation
            verbose: Print progress
        
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"PSO OPTIMIZATION")
            print(f"Controller: {self.controller_type}")
            print(f"Plant: {plant}, Stage: {stage.name}")
            print(f"Particles: {self.num_particles}, Iterations: {self.max_iterations}")
            print(f"{'='*60}\n")
        
        # Initialize swarm
        swarm = [Particle(self.dimensions, self.bounds) 
                 for _ in range(self.num_particles)]
        
        # Global best
        global_best_position = None
        global_best_fitness = float('inf')
        
        # Initial evaluation
        for particle in swarm:
            fitness = self._evaluate_fitness(particle.position, plant, stage)
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
            
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            for particle in swarm:
                # Update velocity
                r1, r2 = np.random.random(2)
                
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (global_best_position - particle.position)
                
                particle.velocity = (self.w * particle.velocity + 
                                    cognitive + social)
                
                # Update position
                particle.position = particle.position + particle.velocity
                
                # Enforce bounds
                particle.position = np.clip(particle.position, 
                                           self.lower_bounds, 
                                           self.upper_bounds)
                
                # Evaluate fitness
                fitness = self._evaluate_fitness(particle.position, plant, stage)
                
                # Update personal best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle.position.copy()
            
            # Record history
            avg_fitness = np.mean([p.best_fitness for p in swarm])
            self.history['best_fitness'].append(global_best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_params'].append(self._decode_parameters(global_best_position))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}: "
                      f"Best Fitness = {global_best_fitness:.4f}, "
                      f"Avg Fitness = {avg_fitness:.4f}")
        
        best_params = self._decode_parameters(global_best_position)
        
        if verbose:
            print(f"\n{'='*60}")
            print("OPTIMIZATION COMPLETE")
            print(f"Best Fitness: {global_best_fitness:.4f}")
            print(f"Best Parameters:")
            for name, value in best_params.items():
                print(f"  {name}: {value:.2f}")
            print(f"{'='*60}\n")
        
        return best_params, global_best_fitness
    
    def get_optimized_controller(self, best_params: Dict):
        """Create a controller with optimized parameters."""
        mf = MembershipFunctions()
        self._apply_parameters(mf, best_params)
        
        if self.controller_type == "mamdani":
            return MamdaniController(mf)
        else:
            return SugenoController(mf)
    
    def compare_before_after(self, plant: str, stage: GrowthStage,
                            best_params: Dict) -> Dict:
        """
        Compare controller performance before and after optimization.
        
        Returns dict with before/after metrics.
        """
        # Default controller
        mf_default = MembershipFunctions()
        if self.controller_type == "mamdani":
            controller_default = MamdaniController(mf_default)
        else:
            controller_default = SugenoController(mf_default)
        
        # Optimized controller
        mf_optimized = MembershipFunctions()
        self._apply_parameters(mf_optimized, best_params)
        if self.controller_type == "mamdani":
            controller_optimized = MamdaniController(mf_optimized)
        else:
            controller_optimized = SugenoController(mf_optimized)
        
        # Evaluate both
        requirements = PlantDatabase.get_requirements(plant, stage)
        stage_value = PlantDatabase.get_stage_numeric(stage)
        
        # Test scenarios
        test_temps = [15, 20, 25, 30, 35]
        test_humidities = [40, 55, 70, 85]
        
        default_error = 0
        optimized_error = 0
        
        for temp in test_temps:
            for humidity in test_humidities:
                result_default = controller_default.infer(temp, humidity, stage_value)
                result_optimized = controller_optimized.infer(temp, humidity, stage_value)
                
                # Calculate error from expected behavior
                temp_dev = abs(temp - requirements.temp_optimal)
                humidity_dev = abs(humidity - requirements.humidity_optimal)
                
                # Expected: more control action when further from optimal
                expected_heater_action = min(50, temp_dev * 3)
                expected_misting_action = min(50, humidity_dev * 2)
                
                default_error += abs(abs(result_default['heater'] - 50) - expected_heater_action)
                default_error += abs(result_default['misting'] - expected_misting_action)
                
                optimized_error += abs(abs(result_optimized['heater'] - 50) - expected_heater_action)
                optimized_error += abs(result_optimized['misting'] - expected_misting_action)
        
        num_tests = len(test_temps) * len(test_humidities)
        
        return {
            'default_error': default_error / num_tests,
            'optimized_error': optimized_error / num_tests,
            'improvement': (default_error - optimized_error) / default_error * 100
        }


def plot_optimization_history(optimizer: PSOOptimizer, save_path: str = None):
    """Plot optimization convergence history."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    iterations = range(1, len(optimizer.history['best_fitness']) + 1)
    
    # Fitness convergence
    ax = axes[0]
    ax.plot(iterations, optimizer.history['best_fitness'], 'b-', 
            label='Best Fitness', linewidth=2)
    ax.plot(iterations, optimizer.history['avg_fitness'], 'r--',
            label='Average Fitness', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness (lower is better)')
    ax.set_title('PSO Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Parameter evolution
    ax = axes[1]
    params_history = optimizer.history['best_params']
    param_names = list(params_history[0].keys())
    
    for name in param_names:
        values = [p[name] for p in params_history]
        # Normalize for visualization
        values_norm = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)
        ax.plot(iterations, values_norm, label=name.replace('_', ' ').title())
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Normalized Parameter Value')
    ax.set_title('Parameter Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Optimization history plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run optimization demo
    print("Running PSO Optimization Demo...")
    
    # Optimize Mamdani controller
    optimizer_mamdani = PSOOptimizer(
        controller_type="mamdani",
        num_particles=15,
        max_iterations=30
    )
    
    best_params_mamdani, best_fitness_mamdani = optimizer_mamdani.optimize(
        plant="Tomato",
        stage=GrowthStage.VEGETATIVE
    )
    
    # Compare before/after
    comparison = optimizer_mamdani.compare_before_after(
        "Tomato", GrowthStage.VEGETATIVE, best_params_mamdani
    )
    
    print("\nBEFORE vs AFTER OPTIMIZATION:")
    print(f"  Default Error: {comparison['default_error']:.4f}")
    print(f"  Optimized Error: {comparison['optimized_error']:.4f}")
    print(f"  Improvement: {comparison['improvement']:.2f}%")
    
    # Plot convergence
    plot_optimization_history(optimizer_mamdani, "optimization_history.png")
    
    # Optimize Sugeno controller
    print("\n" + "="*60)
    optimizer_sugeno = PSOOptimizer(
        controller_type="sugeno",
        num_particles=15,
        max_iterations=30
    )
    
    best_params_sugeno, best_fitness_sugeno = optimizer_sugeno.optimize(
        plant="Tomato",
        stage=GrowthStage.VEGETATIVE
    )
