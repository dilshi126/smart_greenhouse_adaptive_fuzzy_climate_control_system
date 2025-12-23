"""
Adaptive Fuzzy System Module
============================
Implements dynamic adaptation mechanism for the fuzzy controllers.

Part 4: Dynamic Adaptation
--------------------------
The adaptive system automatically adjusts:
1. Membership function parameters based on plant type
2. Rule weights based on growth stage
3. Output scaling based on environmental conditions

ADAPTATION LOGIC:
=================
1. PLANT-BASED ADAPTATION:
   - Shifts "optimal" membership functions to match plant requirements
   - Adjusts sensitivity based on plant tolerance ranges
   
2. GROWTH STAGE ADAPTATION:
   - Seedling: Tighter control, smaller output changes
   - Vegetative: Normal control, standard responses
   - Flowering: Precise control, avoid stress
   
3. ENVIRONMENTAL ADAPTATION:
   - Extreme conditions trigger emergency responses
   - Gradual changes use smooth transitions
   - Rapid changes trigger faster responses
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .plant_database import PlantDatabase, GrowthStage, ClimateRequirements
from .membership_functions import MembershipFunctions, FuzzySet
from .mamdani_controller import MamdaniController
from .sugeno_controller import SugenoController


class AdaptationMode(Enum):
    """Adaptation modes for different scenarios."""
    NORMAL = "normal"           # Standard operation
    CONSERVATIVE = "conservative"  # Reduced output changes (seedlings)
    AGGRESSIVE = "aggressive"   # Faster response (emergencies)
    PRECISE = "precise"         # Tight control (flowering)


@dataclass
class AdaptationState:
    """Current state of the adaptive system."""
    plant_name: str
    growth_stage: GrowthStage
    mode: AdaptationMode
    optimal_temp: float
    optimal_humidity: float
    temp_tolerance: float
    humidity_tolerance: float
    output_scale: float  # Multiplier for output intensity


class AdaptiveFuzzySystem:
    """
    Adaptive fuzzy control system that adjusts to plant and environmental changes.
    
    ADAPTATION MECHANISM:
    =====================
    
    1. When plant type changes:
       - Load new climate requirements from database
       - Shift membership functions to new optimal values
       - Adjust tolerance ranges
    
    2. When growth stage changes:
       - Modify rule weights for stage-specific rules
       - Adjust output scaling (conservative for seedlings)
       - Update optimal setpoints
    
    3. Continuous environmental adaptation:
       - Monitor rate of change in conditions
       - Switch to aggressive mode if rapid changes detected
       - Smooth transitions during normal operation
    """
    
    def __init__(self, initial_plant: str = "Tomato",
                 initial_stage: GrowthStage = GrowthStage.VEGETATIVE,
                 controller_type: str = "both"):
        """
        Initialize the adaptive fuzzy system.
        
        Args:
            initial_plant: Starting plant type
            initial_stage: Starting growth stage
            controller_type: "mamdani", "sugeno", or "both"
        """
        self.controller_type = controller_type
        
        # Initialize membership functions
        self.mf = MembershipFunctions()
        
        # Initialize controllers
        self.mamdani = MamdaniController(self.mf) if controller_type in ["mamdani", "both"] else None
        self.sugeno = SugenoController(self.mf) if controller_type in ["sugeno", "both"] else None
        
        # Initialize adaptation state
        self.state = self._create_initial_state(initial_plant, initial_stage)
        
        # Apply initial adaptation
        self._adapt_to_current_state()
        
        # History for rate-of-change detection
        self.temp_history = []
        self.humidity_history = []
        self.history_window = 10  # Number of samples to track
        
        print(f"Adaptive Fuzzy System initialized for {initial_plant} ({initial_stage.name})")
    
    def _create_initial_state(self, plant_name: str, 
                              stage: GrowthStage) -> AdaptationState:
        """Create initial adaptation state from plant database."""
        requirements = PlantDatabase.get_requirements(plant_name, stage)
        
        # Determine adaptation mode based on growth stage
        mode_map = {
            GrowthStage.SEEDLING: AdaptationMode.CONSERVATIVE,
            GrowthStage.VEGETATIVE: AdaptationMode.NORMAL,
            GrowthStage.FLOWERING: AdaptationMode.PRECISE,
        }
        
        # Calculate tolerances
        temp_tolerance = (requirements.temp_max - requirements.temp_min) / 2
        humidity_tolerance = (requirements.humidity_max - requirements.humidity_min) / 2
        
        # Output scale based on mode
        scale_map = {
            AdaptationMode.CONSERVATIVE: 0.7,
            AdaptationMode.NORMAL: 1.0,
            AdaptationMode.PRECISE: 0.85,
            AdaptationMode.AGGRESSIVE: 1.3,
        }
        
        return AdaptationState(
            plant_name=plant_name,
            growth_stage=stage,
            mode=mode_map[stage],
            optimal_temp=requirements.temp_optimal,
            optimal_humidity=requirements.humidity_optimal,
            temp_tolerance=temp_tolerance,
            humidity_tolerance=humidity_tolerance,
            output_scale=scale_map[mode_map[stage]]
        )
    
    def _adapt_to_current_state(self):
        """Apply adaptations based on current state."""
        # Adapt membership functions
        self.mf.adapt_to_plant(
            self.state.plant_name,
            self.state.optimal_temp,
            self.state.optimal_humidity
        )
        
        # Adjust membership function widths based on tolerance
        self._adjust_mf_widths()
        
        print(f"Adapted to: {self.state.plant_name} - {self.state.growth_stage.name}")
        print(f"  Optimal: {self.state.optimal_temp}°C, {self.state.optimal_humidity}%")
        print(f"  Mode: {self.state.mode.value}, Scale: {self.state.output_scale}")
    
    def _adjust_mf_widths(self):
        """Adjust membership function widths based on plant tolerance."""
        # Wider tolerance → wider optimal MF
        # Tighter tolerance → narrower optimal MF
        
        base_width = 5  # Base half-width for optimal MF
        temp_width = base_width * (self.state.temp_tolerance / 3)  # Normalize
        humidity_width = base_width * (self.state.humidity_tolerance / 5)
        
        # Update temperature optimal MF
        opt_temp = self.state.optimal_temp
        self.mf.temp_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular',
            (opt_temp - temp_width, opt_temp, opt_temp + temp_width)
        )
        
        # Update humidity optimal MF
        opt_hum = self.state.optimal_humidity
        self.mf.humidity_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular',
            (opt_hum - humidity_width, opt_hum, opt_hum + humidity_width)
        )
    
    def change_plant(self, plant_name: str):
        """
        Change the plant type and adapt the system.
        
        Args:
            plant_name: New plant type
        """
        if plant_name not in PlantDatabase.get_all_plants():
            raise ValueError(f"Unknown plant: {plant_name}")
        
        print(f"\n{'='*50}")
        print(f"PLANT CHANGE: {self.state.plant_name} → {plant_name}")
        print(f"{'='*50}")
        
        self.state = self._create_initial_state(plant_name, self.state.growth_stage)
        self._adapt_to_current_state()
    
    def change_growth_stage(self, stage: GrowthStage):
        """
        Change the growth stage and adapt the system.
        
        Args:
            stage: New growth stage
        """
        print(f"\n{'='*50}")
        print(f"GROWTH STAGE CHANGE: {self.state.growth_stage.name} → {stage.name}")
        print(f"{'='*50}")
        
        self.state = self._create_initial_state(self.state.plant_name, stage)
        self._adapt_to_current_state()
    
    def _detect_rapid_change(self, temp: float, humidity: float) -> bool:
        """Detect if environmental conditions are changing rapidly."""
        self.temp_history.append(temp)
        self.humidity_history.append(humidity)
        
        # Keep only recent history
        if len(self.temp_history) > self.history_window:
            self.temp_history.pop(0)
            self.humidity_history.pop(0)
        
        if len(self.temp_history) < 3:
            return False
        
        # Calculate rate of change
        temp_change = abs(self.temp_history[-1] - self.temp_history[0])
        humidity_change = abs(self.humidity_history[-1] - self.humidity_history[0])
        
        # Thresholds for rapid change
        rapid_temp_change = 5.0  # °C over window
        rapid_humidity_change = 15.0  # % over window
        
        return temp_change > rapid_temp_change or humidity_change > rapid_humidity_change
    
    def _check_emergency(self, temp: float, humidity: float) -> bool:
        """Check if conditions require emergency response."""
        # Emergency thresholds
        temp_emergency_low = self.state.optimal_temp - 15
        temp_emergency_high = self.state.optimal_temp + 15
        humidity_emergency_low = 20
        humidity_emergency_high = 95
        
        return (temp < temp_emergency_low or temp > temp_emergency_high or
                humidity < humidity_emergency_low or humidity > humidity_emergency_high)
    
    def control(self, temp: float, humidity: float, 
                growth_stage_value: float = None) -> Dict[str, Dict[str, float]]:
        """
        Compute control outputs with adaptation.
        
        Args:
            temp: Current temperature (°C)
            humidity: Current humidity (%)
            growth_stage_value: Optional override for growth stage (0-100)
        
        Returns:
            Dict with 'mamdani' and/or 'sugeno' results, each containing
            'heater' and 'misting' values
        """
        # Use current growth stage if not specified
        if growth_stage_value is None:
            growth_stage_value = PlantDatabase.get_stage_numeric(self.state.growth_stage)
        
        # Check for mode changes
        original_mode = self.state.mode
        
        if self._check_emergency(temp, humidity):
            self.state.mode = AdaptationMode.AGGRESSIVE
            self.state.output_scale = 1.3
        elif self._detect_rapid_change(temp, humidity):
            self.state.mode = AdaptationMode.AGGRESSIVE
            self.state.output_scale = 1.2
        else:
            # Restore mode based on growth stage
            mode_map = {
                GrowthStage.SEEDLING: AdaptationMode.CONSERVATIVE,
                GrowthStage.VEGETATIVE: AdaptationMode.NORMAL,
                GrowthStage.FLOWERING: AdaptationMode.PRECISE,
            }
            self.state.mode = mode_map[self.state.growth_stage]
            scale_map = {
                AdaptationMode.CONSERVATIVE: 0.7,
                AdaptationMode.NORMAL: 1.0,
                AdaptationMode.PRECISE: 0.85,
            }
            self.state.output_scale = scale_map[self.state.mode]
        
        if self.state.mode != original_mode:
            print(f"Mode changed: {original_mode.value} → {self.state.mode.value}")
        
        results = {}
        
        # Get Mamdani output
        if self.mamdani:
            mamdani_result = self.mamdani.infer(temp, humidity, growth_stage_value)
            results['mamdani'] = {
                'heater': self._scale_output(mamdani_result['heater']),
                'misting': self._scale_output(mamdani_result['misting']),
                'raw_heater': mamdani_result['heater'],
                'raw_misting': mamdani_result['misting'],
            }
        
        # Get Sugeno output
        if self.sugeno:
            sugeno_result = self.sugeno.infer(temp, humidity, growth_stage_value)
            results['sugeno'] = {
                'heater': self._scale_output(sugeno_result['heater']),
                'misting': self._scale_output(sugeno_result['misting']),
                'raw_heater': sugeno_result['heater'],
                'raw_misting': sugeno_result['misting'],
            }
        
        # Add state information
        results['state'] = {
            'plant': self.state.plant_name,
            'stage': self.state.growth_stage.name,
            'mode': self.state.mode.value,
            'scale': self.state.output_scale,
            'optimal_temp': self.state.optimal_temp,
            'optimal_humidity': self.state.optimal_humidity,
        }
        
        return results
    
    def _scale_output(self, value: float) -> float:
        """Apply output scaling based on adaptation mode."""
        # Scale around the neutral point (50 for heater)
        if value >= 50:
            # Heating/high misting - scale the deviation from neutral
            deviation = value - 50
            scaled_deviation = deviation * self.state.output_scale
            return min(100, 50 + scaled_deviation)
        else:
            # Cooling/low misting - scale the deviation from neutral
            deviation = 50 - value
            scaled_deviation = deviation * self.state.output_scale
            return max(0, 50 - scaled_deviation)
    
    def get_setpoints(self) -> Dict[str, float]:
        """Get current optimal setpoints."""
        return {
            'temperature': self.state.optimal_temp,
            'humidity': self.state.optimal_humidity,
        }
    
    def get_adaptation_info(self) -> Dict:
        """Get current adaptation state information."""
        return {
            'plant': self.state.plant_name,
            'growth_stage': self.state.growth_stage.name,
            'mode': self.state.mode.value,
            'optimal_temp': self.state.optimal_temp,
            'optimal_humidity': self.state.optimal_humidity,
            'temp_tolerance': self.state.temp_tolerance,
            'humidity_tolerance': self.state.humidity_tolerance,
            'output_scale': self.state.output_scale,
        }


def demonstrate_adaptation():
    """Demonstrate the adaptive system behavior."""
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("ADAPTIVE FUZZY SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Create adaptive system
    system = AdaptiveFuzzySystem("Tomato", GrowthStage.SEEDLING)
    
    # Simulate changing conditions
    time_steps = 100
    temps = np.concatenate([
        np.linspace(20, 35, 30),    # Warming
        np.linspace(35, 15, 40),    # Cooling
        np.linspace(15, 25, 30),    # Return to normal
    ])
    
    humidities = np.concatenate([
        np.linspace(70, 40, 30),    # Drying
        np.linspace(40, 85, 40),    # Humidifying
        np.linspace(85, 65, 30),    # Return to normal
    ])
    
    # Storage for results
    mamdani_heater = []
    mamdani_misting = []
    sugeno_heater = []
    sugeno_misting = []
    modes = []
    
    # Simulate with plant and stage changes
    for i in range(time_steps):
        # Change plant at step 33
        if i == 33:
            system.change_plant("Lettuce")
        
        # Change growth stage at step 66
        if i == 66:
            system.change_growth_stage(GrowthStage.FLOWERING)
        
        result = system.control(temps[i], humidities[i])
        
        mamdani_heater.append(result['mamdani']['heater'])
        mamdani_misting.append(result['mamdani']['misting'])
        sugeno_heater.append(result['sugeno']['heater'])
        sugeno_misting.append(result['sugeno']['misting'])
        modes.append(result['state']['mode'])
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    time = np.arange(time_steps)
    
    # Temperature and Humidity inputs
    ax = axes[0, 0]
    ax.plot(time, temps, 'r-', label='Temperature (°C)')
    ax.plot(time, humidities, 'b-', label='Humidity (%)')
    ax.axvline(x=33, color='g', linestyle='--', alpha=0.5, label='Plant Change')
    ax.axvline(x=66, color='m', linestyle='--', alpha=0.5, label='Stage Change')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title('Input Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heater outputs comparison
    ax = axes[0, 1]
    ax.plot(time, mamdani_heater, 'r-', label='Mamdani', alpha=0.7)
    ax.plot(time, sugeno_heater, 'b--', label='Sugeno', alpha=0.7)
    ax.axvline(x=33, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=66, color='m', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Neutral')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Heater/Cooling (%)')
    ax.set_title('Heater Control Output')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Misting outputs comparison
    ax = axes[1, 0]
    ax.plot(time, mamdani_misting, 'r-', label='Mamdani', alpha=0.7)
    ax.plot(time, sugeno_misting, 'b--', label='Sugeno', alpha=0.7)
    ax.axvline(x=33, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=66, color='m', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Misting (%)')
    ax.set_title('Misting Control Output')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Output difference (Mamdani - Sugeno)
    ax = axes[1, 1]
    heater_diff = np.array(mamdani_heater) - np.array(sugeno_heater)
    misting_diff = np.array(mamdani_misting) - np.array(sugeno_misting)
    ax.plot(time, heater_diff, 'r-', label='Heater Diff', alpha=0.7)
    ax.plot(time, misting_diff, 'b-', label='Misting Diff', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Difference (%)')
    ax.set_title('Controller Difference (Mamdani - Sugeno)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mode visualization
    ax = axes[2, 0]
    mode_values = {'normal': 1, 'conservative': 0.5, 'precise': 0.75, 'aggressive': 1.5}
    mode_numeric = [mode_values[m] for m in modes]
    ax.plot(time, mode_numeric, 'g-', linewidth=2)
    ax.axvline(x=33, color='g', linestyle='--', alpha=0.5)
    ax.axvline(x=66, color='m', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mode Level')
    ax.set_title('Adaptation Mode')
    ax.set_yticks([0.5, 0.75, 1.0, 1.5])
    ax.set_yticklabels(['Conservative', 'Precise', 'Normal', 'Aggressive'])
    ax.grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('adaptation_demo.png', dpi=150, bbox_inches='tight')
    print("\nAdaptation demonstration plot saved to: adaptation_demo.png")
    plt.show()


if __name__ == "__main__":
    demonstrate_adaptation()
