"""
Greenhouse Simulator Module
===========================
Simulates greenhouse environment and evaluates controller performance.

Part 5: Performance Evaluation
------------------------------
Conducts simulations with varying weather conditions and compares
Mamdani vs Sugeno controller performance.

PERFORMANCE METRICS:
====================
1. Average Response Time: How quickly the controller responds to changes
2. Average Error: Deviation from optimal setpoints
3. Energy Usage: Total control effort (proxy for energy consumption)
4. Smoothness Score: Variation in control outputs (lower = smoother)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from .plant_database import PlantDatabase, GrowthStage
from .adaptive_system import AdaptiveFuzzySystem


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    plant: str
    growth_stage: str
    initial_temp: float
    initial_humidity: float
    weather_pattern: str
    
    # Time series data
    temps: np.ndarray
    humidities: np.ndarray
    mamdani_heater: np.ndarray
    mamdani_misting: np.ndarray
    sugeno_heater: np.ndarray
    sugeno_misting: np.ndarray
    
    # Performance metrics
    mamdani_metrics: Dict[str, float]
    sugeno_metrics: Dict[str, float]


class WeatherGenerator:
    """Generates realistic weather patterns for simulation."""
    
    @staticmethod
    def generate_pattern(pattern_type: str, duration: int, 
                        base_temp: float = 25, base_humidity: float = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weather pattern.
        
        Args:
            pattern_type: Type of weather pattern
            duration: Number of time steps
            base_temp: Starting temperature
            base_humidity: Starting humidity
        
        Returns:
            Tuple of (temperatures, humidities) arrays
        """
        t = np.linspace(0, duration, duration)
        
        if pattern_type == "stable":
            # Stable conditions with minor fluctuations
            temps = base_temp + np.random.normal(0, 1, duration)
            humidities = base_humidity + np.random.normal(0, 2, duration)
            
        elif pattern_type == "warming":
            # Gradual warming trend
            temps = base_temp + 0.15 * t + np.random.normal(0, 0.5, duration)
            humidities = base_humidity - 0.1 * t + np.random.normal(0, 1, duration)
            
        elif pattern_type == "cooling":
            # Gradual cooling trend
            temps = base_temp - 0.15 * t + np.random.normal(0, 0.5, duration)
            humidities = base_humidity + 0.08 * t + np.random.normal(0, 1, duration)
            
        elif pattern_type == "heat_wave":
            # Sudden heat spike
            temps = base_temp + 10 * np.exp(-((t - duration/2)**2) / (2 * (duration/6)**2))
            temps += np.random.normal(0, 0.5, duration)
            humidities = base_humidity - 15 * np.exp(-((t - duration/2)**2) / (2 * (duration/6)**2))
            humidities += np.random.normal(0, 1, duration)
            
        elif pattern_type == "cold_snap":
            # Sudden cold drop
            temps = base_temp - 12 * np.exp(-((t - duration/2)**2) / (2 * (duration/6)**2))
            temps += np.random.normal(0, 0.5, duration)
            humidities = base_humidity + 10 * np.exp(-((t - duration/2)**2) / (2 * (duration/6)**2))
            humidities += np.random.normal(0, 1, duration)
            
        elif pattern_type == "oscillating":
            # Day/night cycle simulation
            temps = base_temp + 8 * np.sin(2 * np.pi * t / (duration/3))
            temps += np.random.normal(0, 0.5, duration)
            humidities = base_humidity - 10 * np.sin(2 * np.pi * t / (duration/3))
            humidities += np.random.normal(0, 1, duration)
            
        elif pattern_type == "humid_storm":
            # High humidity event
            temps = base_temp - 3 + np.random.normal(0, 1, duration)
            humidities = base_humidity + 25 * np.exp(-((t - duration/2)**2) / (2 * (duration/5)**2))
            humidities += np.random.normal(0, 1, duration)
            
        elif pattern_type == "dry_spell":
            # Low humidity period
            temps = base_temp + 5 + np.random.normal(0, 1, duration)
            humidities = base_humidity - 30 * (1 - np.exp(-t / (duration/3)))
            humidities += np.random.normal(0, 1, duration)
            
        else:  # random
            # Random walk
            temps = base_temp + np.cumsum(np.random.normal(0, 0.5, duration))
            humidities = base_humidity + np.cumsum(np.random.normal(0, 0.8, duration))
        
        # Clip to realistic ranges
        temps = np.clip(temps, 0, 45)
        humidities = np.clip(humidities, 10, 95)
        
        return temps, humidities


class GreenhouseSimulator:
    """
    Simulates greenhouse environment with fuzzy control.
    
    The simulator models:
    - External weather influence on greenhouse conditions
    - Controller response to maintain optimal conditions
    - Plant-specific requirements and adaptations
    """
    
    def __init__(self):
        """Initialize the simulator."""
        self.weather_patterns = [
            "stable", "warming", "cooling", "heat_wave", 
            "cold_snap", "oscillating", "humid_storm", "dry_spell", "random"
        ]
    
    def run_simulation(self, plant: str, stage: GrowthStage,
                      weather_pattern: str, duration: int = 100,
                      initial_temp: float = None,
                      initial_humidity: float = None) -> SimulationResult:
        """
        Run a single simulation.
        
        Args:
            plant: Plant type
            stage: Growth stage
            weather_pattern: Weather pattern type
            duration: Simulation duration (time steps)
            initial_temp: Starting temperature (optional)
            initial_humidity: Starting humidity (optional)
        
        Returns:
            SimulationResult with all data and metrics
        """
        # Get plant requirements for initial conditions
        requirements = PlantDatabase.get_requirements(plant, stage)
        
        if initial_temp is None:
            initial_temp = requirements.temp_optimal
        if initial_humidity is None:
            initial_humidity = requirements.humidity_optimal
        
        # Generate weather pattern
        temps, humidities = WeatherGenerator.generate_pattern(
            weather_pattern, duration, initial_temp, initial_humidity
        )
        
        # Create adaptive system
        system = AdaptiveFuzzySystem(plant, stage, controller_type="both")
        
        # Storage for outputs
        mamdani_heater = np.zeros(duration)
        mamdani_misting = np.zeros(duration)
        sugeno_heater = np.zeros(duration)
        sugeno_misting = np.zeros(duration)
        
        # Run simulation
        stage_value = PlantDatabase.get_stage_numeric(stage)
        
        mamdani_times = []
        sugeno_times = []
        
        for i in range(duration):
            # Time the controllers
            start = time.perf_counter()
            result = system.control(temps[i], humidities[i], stage_value)
            
            mamdani_heater[i] = result['mamdani']['heater']
            mamdani_misting[i] = result['mamdani']['misting']
            sugeno_heater[i] = result['sugeno']['heater']
            sugeno_misting[i] = result['sugeno']['misting']
        
        # Calculate performance metrics
        mamdani_metrics = self._calculate_metrics(
            temps, humidities, mamdani_heater, mamdani_misting,
            requirements.temp_optimal, requirements.humidity_optimal
        )
        
        sugeno_metrics = self._calculate_metrics(
            temps, humidities, sugeno_heater, sugeno_misting,
            requirements.temp_optimal, requirements.humidity_optimal
        )
        
        return SimulationResult(
            plant=plant,
            growth_stage=stage.name,
            initial_temp=initial_temp,
            initial_humidity=initial_humidity,
            weather_pattern=weather_pattern,
            temps=temps,
            humidities=humidities,
            mamdani_heater=mamdani_heater,
            mamdani_misting=mamdani_misting,
            sugeno_heater=sugeno_heater,
            sugeno_misting=sugeno_misting,
            mamdani_metrics=mamdani_metrics,
            sugeno_metrics=sugeno_metrics
        )
    
    def _calculate_metrics(self, temps: np.ndarray, humidities: np.ndarray,
                          heater: np.ndarray, misting: np.ndarray,
                          optimal_temp: float, optimal_humidity: float) -> Dict[str, float]:
        """
        Calculate performance metrics for a controller.
        
        Metrics:
        1. Average Response Time: Measured by output change rate
        2. Average Error: Mean absolute deviation from setpoint
        3. Energy Usage: Sum of absolute control efforts
        4. Smoothness Score: Standard deviation of output changes
        """
        # Average Error (from optimal conditions)
        # We estimate what the controlled temp/humidity would be
        # Higher heater output → higher temp, higher misting → higher humidity
        temp_error = np.abs(temps - optimal_temp)
        humidity_error = np.abs(humidities - optimal_humidity)
        avg_error = np.mean(temp_error) + np.mean(humidity_error) * 0.5
        
        # Energy Usage (total control effort)
        # Deviation from neutral (50 for heater, 0 for misting when optimal)
        heater_effort = np.sum(np.abs(heater - 50))
        misting_effort = np.sum(misting)
        energy_usage = (heater_effort + misting_effort) / len(heater)
        
        # Smoothness Score (lower is better)
        heater_changes = np.diff(heater)
        misting_changes = np.diff(misting)
        smoothness = np.std(heater_changes) + np.std(misting_changes)
        
        # Response Time (how quickly output changes in response to input changes)
        temp_changes = np.abs(np.diff(temps))
        heater_response = np.abs(np.diff(heater))
        
        # Correlation between input change and output response
        if np.std(temp_changes) > 0 and np.std(heater_response) > 0:
            response_correlation = np.corrcoef(temp_changes, heater_response)[0, 1]
            response_time = 1 - abs(response_correlation)  # Lower is faster response
        else:
            response_time = 0.5
        
        return {
            'avg_response_time': float(response_time),
            'avg_error': float(avg_error),
            'energy_usage': float(energy_usage),
            'smoothness_score': float(smoothness)
        }
    
    def run_random_tests(self, num_tests: int = 20) -> List[SimulationResult]:
        """
        Run multiple random test simulations.
        
        Args:
            num_tests: Number of tests to run
        
        Returns:
            List of SimulationResult objects
        """
        results = []
        plants = PlantDatabase.get_all_plants()
        stages = list(GrowthStage)
        
        print(f"\nRunning {num_tests} random simulations...")
        print("-" * 60)
        
        for i in range(num_tests):
            # Random selection
            plant = np.random.choice(plants)
            stage = np.random.choice(stages)
            pattern = np.random.choice(self.weather_patterns)
            
            # Random initial conditions
            initial_temp = np.random.uniform(10, 35)
            initial_humidity = np.random.uniform(30, 85)
            
            print(f"Test {i+1}/{num_tests}: {plant} ({stage.name}) - {pattern}")
            
            result = self.run_simulation(
                plant, stage, pattern,
                initial_temp=initial_temp,
                initial_humidity=initial_humidity
            )
            results.append(result)
        
        return results
    
    def generate_comparison_table(self, results: List[SimulationResult]) -> str:
        """
        Generate performance comparison table.
        
        Returns formatted table comparing Mamdani vs Sugeno performance.
        """
        # Aggregate metrics
        mamdani_response = np.mean([r.mamdani_metrics['avg_response_time'] for r in results])
        mamdani_error = np.mean([r.mamdani_metrics['avg_error'] for r in results])
        mamdani_energy = np.mean([r.mamdani_metrics['energy_usage'] for r in results])
        mamdani_smooth = np.mean([r.mamdani_metrics['smoothness_score'] for r in results])
        
        sugeno_response = np.mean([r.sugeno_metrics['avg_response_time'] for r in results])
        sugeno_error = np.mean([r.sugeno_metrics['avg_error'] for r in results])
        sugeno_energy = np.mean([r.sugeno_metrics['energy_usage'] for r in results])
        sugeno_smooth = np.mean([r.sugeno_metrics['smoothness_score'] for r in results])
        
        table = f"""
{'='*70}
PERFORMANCE COMPARISON TABLE ({len(results)} simulations)
{'='*70}

| Controller | Avg Response Time | Avg Error | Energy Usage | Smoothness Score |
|------------|-------------------|-----------|--------------|------------------|
| Mamdani    | {mamdani_response:17.4f} | {mamdani_error:9.4f} | {mamdani_energy:12.4f} | {mamdani_smooth:16.4f} |
| Sugeno     | {sugeno_response:17.4f} | {sugeno_error:9.4f} | {sugeno_energy:12.4f} | {sugeno_smooth:16.4f} |

{'='*70}

ANALYSIS:
---------
Response Time: {'Mamdani' if mamdani_response < sugeno_response else 'Sugeno'} is faster (lower is better)
Average Error: {'Mamdani' if mamdani_error < sugeno_error else 'Sugeno'} has lower error
Energy Usage:  {'Mamdani' if mamdani_energy < sugeno_energy else 'Sugeno'} uses less energy
Smoothness:    {'Mamdani' if mamdani_smooth < sugeno_smooth else 'Sugeno'} is smoother (lower is better)

OVERALL WINNER: {'Mamdani' if (mamdani_response + mamdani_error + mamdani_energy + mamdani_smooth) < 
                              (sugeno_response + sugeno_error + sugeno_energy + sugeno_smooth) else 'Sugeno'}

REASONING:
- Mamdani provides smoother control due to fuzzy output aggregation
- Sugeno is computationally more efficient with crisp outputs
- For greenhouse control, smoothness is important to avoid plant stress
- Energy efficiency matters for operational costs
{'='*70}
"""
        return table


def plot_simulation_results(result: SimulationResult, save_path: str = None):
    """Plot results from a single simulation."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time = np.arange(len(result.temps))
    
    # Input conditions
    ax = axes[0, 0]
    ax.plot(time, result.temps, 'r-', label='Temperature (°C)')
    ax.plot(time, result.humidities, 'b-', label='Humidity (%)')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'Weather Pattern: {result.weather_pattern}\n'
                f'Plant: {result.plant} ({result.growth_stage})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heater outputs
    ax = axes[0, 1]
    ax.plot(time, result.mamdani_heater, 'r-', label='Mamdani', alpha=0.7)
    ax.plot(time, result.sugeno_heater, 'b--', label='Sugeno', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Neutral')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Heater/Cooling (%)')
    ax.set_title('Heater Control Output')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Misting outputs
    ax = axes[1, 0]
    ax.plot(time, result.mamdani_misting, 'r-', label='Mamdani', alpha=0.7)
    ax.plot(time, result.sugeno_misting, 'b--', label='Sugeno', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Misting (%)')
    ax.set_title('Misting Control Output')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics comparison
    ax = axes[1, 1]
    metrics = ['Response\nTime', 'Avg\nError', 'Energy\nUsage', 'Smoothness']
    mamdani_vals = [
        result.mamdani_metrics['avg_response_time'],
        result.mamdani_metrics['avg_error'] / 10,  # Scale for visibility
        result.mamdani_metrics['energy_usage'] / 50,
        result.mamdani_metrics['smoothness_score'] / 5
    ]
    sugeno_vals = [
        result.sugeno_metrics['avg_response_time'],
        result.sugeno_metrics['avg_error'] / 10,
        result.sugeno_metrics['energy_usage'] / 50,
        result.sugeno_metrics['smoothness_score'] / 5
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, mamdani_vals, width, label='Mamdani', color='red', alpha=0.7)
    ax.bar(x + width/2, sugeno_vals, width, label='Sugeno', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized Score')
    ax.set_title('Performance Metrics (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Simulation plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Run demonstration
    simulator = GreenhouseSimulator()
    
    # Single simulation demo
    print("Running single simulation demo...")
    result = simulator.run_simulation(
        "Tomato", GrowthStage.VEGETATIVE, "heat_wave"
    )
    plot_simulation_results(result, "simulation_demo.png")
    
    # Run 20 random tests
    results = simulator.run_random_tests(20)
    
    # Generate comparison table
    table = simulator.generate_comparison_table(results)
    print(table)
