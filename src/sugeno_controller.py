"""
Takagi-Sugeno Fuzzy Controller Module
=====================================
Implements a Takagi-Sugeno (TS) fuzzy inference system for greenhouse climate control.

Part 2: Fuzzy System Design - Sugeno Controller
-----------------------------------------------
The Sugeno controller uses:
- Same fuzzification as Mamdani
- Linear output functions instead of fuzzy sets
- Weighted average defuzzification

JUSTIFICATION FOR SUGENO:
=========================
1. Computationally efficient - no need to defuzzify fuzzy output
2. Works well with optimization techniques (linear parameters)
3. Suitable for mathematical analysis and stability proofs
4. Better for systems requiring precise numerical output
5. Easier to integrate with other control systems

COMPARISON WITH MAMDANI:
========================
| Aspect              | Mamdani           | Sugeno              |
|---------------------|-------------------|---------------------|
| Output              | Fuzzy set         | Crisp (linear func) |
| Defuzzification     | Centroid/COG      | Weighted average    |
| Computation         | More intensive    | More efficient      |
| Interpretability    | More intuitive    | Less intuitive      |
| Optimization        | Harder            | Easier              |
| Control surface     | Smoother          | Can be discontinuous|
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from .membership_functions import MembershipFunctions


class SugenoRule:
    """
    Represents a Takagi-Sugeno fuzzy rule.
    
    In Sugeno systems, the consequent is a function of inputs:
    - Zero-order: output = constant
    - First-order: output = a*temp + b*humidity + c*stage + d
    """
    
    def __init__(self, rule_id: int, antecedents: Dict[str, str],
                 heater_coeffs: Tuple[float, ...], 
                 misting_coeffs: Tuple[float, ...],
                 weight: float = 1.0):
        """
        Initialize a Sugeno rule.
        
        Args:
            rule_id: Unique identifier
            antecedents: Dict mapping input variable to fuzzy set name
            heater_coeffs: (a, b, c, d) for heater = a*temp + b*hum + c*stage + d
            misting_coeffs: (a, b, c, d) for misting = a*temp + b*hum + c*stage + d
            weight: Rule weight (0-1)
        """
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.heater_coeffs = heater_coeffs
        self.misting_coeffs = misting_coeffs
        self.weight = weight
    
    def compute_heater_output(self, temp: float, humidity: float, stage: float) -> float:
        """Compute heater output using linear function."""
        a, b, c, d = self.heater_coeffs
        output = a * temp + b * humidity + c * stage + d
        return np.clip(output, 0, 100)
    
    def compute_misting_output(self, temp: float, humidity: float, stage: float) -> float:
        """Compute misting output using linear function."""
        a, b, c, d = self.misting_coeffs
        output = a * temp + b * humidity + c * stage + d
        return np.clip(output, 0, 100)
    
    def __repr__(self):
        ant_str = " AND ".join([f"{k} IS {v}" for k, v in self.antecedents.items()])
        return (f"Rule {self.rule_id}: IF {ant_str} THEN "
                f"heater={self.heater_coeffs}, misting={self.misting_coeffs}")


class SugenoController:
    """
    Takagi-Sugeno fuzzy inference system for greenhouse control.
    
    RULE OUTPUT FUNCTION DESIGN:
    ============================
    
    Each rule has linear output functions:
    heater = a*temp + b*humidity + c*stage + d
    misting = a*temp + b*humidity + c*stage + d
    
    Coefficient design rationale:
    - Temperature coefficient (a): Negative for heater (higher temp → less heating)
    - Humidity coefficient (b): Negative for misting (higher humidity → less misting)
    - Stage coefficient (c): Modifies sensitivity based on growth phase
    - Constant (d): Base output level for the rule
    
    The coefficients are designed to produce outputs in 0-100% range.
    """
    
    def __init__(self, membership_functions: MembershipFunctions = None):
        """Initialize the Sugeno controller."""
        self.mf = membership_functions or MembershipFunctions()
        self.rules = self._create_rule_base()
    
    def _create_rule_base(self) -> List[SugenoRule]:
        """
        Create the Sugeno rule base with 27+ rules.
        
        Output coefficients are designed based on:
        - Physical relationships (temp↑ → cooling↑, humidity↓ → misting↑)
        - Normalized to produce 0-100% outputs
        """
        rules = []
        rule_id = 1
        
        # ============== TEMPERATURE-HUMIDITY RULES ==============
        # Coefficients: (temp_coeff, humidity_coeff, stage_coeff, constant)
        
        # VERY COLD temperature rules (5 rules) - Need maximum heating
        rules.append(SugenoRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'very_dry'},
            heater_coeffs=(-1.5, 0.0, 0.0, 95),      # High heating
            misting_coeffs=(0.5, -0.8, 0.0, 80)))    # High misting
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'dry'},
            heater_coeffs=(-1.5, 0.0, 0.0, 95),
            misting_coeffs=(0.3, -0.6, 0.0, 60)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'optimal'},
            heater_coeffs=(-1.5, 0.0, 0.0, 95),
            misting_coeffs=(0.1, -0.3, 0.0, 30)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'humid'},
            heater_coeffs=(-1.5, 0.1, 0.0, 90),
            misting_coeffs=(0.0, -0.2, 0.0, 15)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'very_humid'},
            heater_coeffs=(-1.5, 0.15, 0.0, 85),
            misting_coeffs=(0.0, 0.0, 0.0, 5)))
        rule_id += 1
        
        # COLD temperature rules (5 rules) - Need moderate heating
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'humidity': 'very_dry'},
            heater_coeffs=(-1.0, 0.0, 0.0, 75),
            misting_coeffs=(0.4, -0.7, 0.0, 75)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'humidity': 'dry'},
            heater_coeffs=(-1.0, 0.0, 0.0, 75),
            misting_coeffs=(0.3, -0.5, 0.0, 55)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'humidity': 'optimal'},
            heater_coeffs=(-1.0, 0.0, 0.0, 75),
            misting_coeffs=(0.1, -0.2, 0.0, 25)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'humidity': 'humid'},
            heater_coeffs=(-1.0, 0.1, 0.0, 70),
            misting_coeffs=(0.0, -0.1, 0.0, 10)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'humidity': 'very_humid'},
            heater_coeffs=(-1.0, 0.15, 0.0, 65),
            misting_coeffs=(0.0, 0.0, 0.0, 5)))
        rule_id += 1
        
        # OPTIMAL temperature rules (5 rules) - Maintain conditions
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'very_dry'},
            heater_coeffs=(0.0, 0.0, 0.0, 50),       # Neutral
            misting_coeffs=(0.3, -0.6, 0.0, 70)))    # High misting
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'dry'},
            heater_coeffs=(0.0, 0.0, 0.0, 50),
            misting_coeffs=(0.2, -0.4, 0.0, 50)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'optimal'},
            heater_coeffs=(0.0, 0.0, 0.0, 50),       # Perfect - maintain
            misting_coeffs=(0.0, 0.0, 0.0, 25)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'humid'},
            heater_coeffs=(0.0, 0.0, 0.0, 50),
            misting_coeffs=(0.0, -0.1, 0.0, 10)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'very_humid'},
            heater_coeffs=(0.0, 0.0, 0.0, 50),
            misting_coeffs=(0.0, 0.0, 0.0, 5)))
        rule_id += 1
        
        # WARM temperature rules (5 rules) - Need cooling
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'humidity': 'very_dry'},
            heater_coeffs=(1.0, 0.0, 0.0, 10),       # Cooling
            misting_coeffs=(0.5, -0.5, 0.0, 85)))    # Maximum misting
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'humidity': 'dry'},
            heater_coeffs=(1.0, 0.0, 0.0, 15),
            misting_coeffs=(0.4, -0.4, 0.0, 70)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'humidity': 'optimal'},
            heater_coeffs=(1.0, 0.0, 0.0, 20),
            misting_coeffs=(0.2, -0.2, 0.0, 45)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'humidity': 'humid'},
            heater_coeffs=(1.0, 0.1, 0.0, 25),
            misting_coeffs=(0.1, -0.1, 0.0, 25)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'humidity': 'very_humid'},
            heater_coeffs=(1.0, 0.15, 0.0, 30),
            misting_coeffs=(0.0, 0.0, 0.0, 10)))
        rule_id += 1
        
        # HOT temperature rules (5 rules) - Need maximum cooling
        rules.append(SugenoRule(rule_id,
            {'temperature': 'hot', 'humidity': 'very_dry'},
            heater_coeffs=(1.5, 0.0, 0.0, 0),        # Maximum cooling
            misting_coeffs=(0.6, -0.4, 0.0, 90)))    # Maximum misting
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'hot', 'humidity': 'dry'},
            heater_coeffs=(1.5, 0.0, 0.0, 0),
            misting_coeffs=(0.5, -0.3, 0.0, 85)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'hot', 'humidity': 'optimal'},
            heater_coeffs=(1.5, 0.0, 0.0, 5),
            misting_coeffs=(0.3, -0.2, 0.0, 65)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'hot', 'humidity': 'humid'},
            heater_coeffs=(1.5, 0.1, 0.0, 10),
            misting_coeffs=(0.2, -0.1, 0.0, 45)))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'hot', 'humidity': 'very_humid'},
            heater_coeffs=(1.5, 0.15, 0.0, 15),
            misting_coeffs=(0.1, 0.0, 0.0, 25)))
        rule_id += 1
        
        # ============== GROWTH STAGE MODIFIER RULES ==============
        
        # Seedling rules - more conservative, stage affects output
        rules.append(SugenoRule(rule_id,
            {'temperature': 'cold', 'growth_stage': 'early_seedling'},
            heater_coeffs=(-1.2, 0.0, -0.1, 85),
            misting_coeffs=(0.2, -0.3, 0.05, 40),
            weight=0.8))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'growth_stage': 'early_seedling'},
            heater_coeffs=(0.8, 0.0, -0.1, 25),
            misting_coeffs=(0.3, -0.3, 0.05, 60),
            weight=0.8))
        rule_id += 1
        
        # Flowering rules - tighter control
        rules.append(SugenoRule(rule_id,
            {'humidity': 'humid', 'growth_stage': 'flowering'},
            heater_coeffs=(0.0, 0.05, -0.05, 50),
            misting_coeffs=(0.0, -0.15, -0.05, 15),
            weight=0.9))
        rule_id += 1
        
        rules.append(SugenoRule(rule_id,
            {'temperature': 'optimal', 'growth_stage': 'flowering'},
            heater_coeffs=(0.0, 0.0, -0.05, 50),
            misting_coeffs=(0.05, -0.1, -0.05, 20),
            weight=0.9))
        rule_id += 1
        
        # Vegetative growth - more tolerant
        rules.append(SugenoRule(rule_id,
            {'temperature': 'warm', 'growth_stage': 'late_vegetative'},
            heater_coeffs=(0.5, 0.0, 0.0, 35),
            misting_coeffs=(0.2, -0.2, 0.0, 40),
            weight=0.7))
        rule_id += 1
        
        print(f"Created {len(rules)} fuzzy rules for Sugeno controller")
        return rules
    
    def evaluate_rule(self, rule: SugenoRule,
                      fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate a single rule's firing strength.
        
        Uses MIN operator for AND conjunction (same as Mamdani).
        """
        firing_strength = 1.0
        
        for var_name, fuzzy_set_name in rule.antecedents.items():
            if var_name in fuzzified_inputs:
                membership = fuzzified_inputs[var_name].get(fuzzy_set_name, 0.0)
                firing_strength = min(firing_strength, membership)
        
        return firing_strength * rule.weight
    
    def infer(self, temp: float, humidity: float, growth_stage: float) -> Dict[str, float]:
        """
        Perform Takagi-Sugeno fuzzy inference.
        
        Args:
            temp: Temperature in °C
            humidity: Humidity in %
            growth_stage: Growth stage (0-100 encoded)
        
        Returns:
            Dict with 'heater' and 'misting' crisp output values (0-100%)
        
        DEFUZZIFICATION: Weighted Average
        ==================================
        output = Σ(wi * zi) / Σ(wi)
        
        where:
        - wi = firing strength of rule i
        - zi = output of rule i's consequent function
        
        JUSTIFICATION:
        - Computationally efficient (no integration)
        - Produces crisp output directly
        - Mathematically tractable for analysis
        - Well-suited for optimization
        """
        # Step 1: Fuzzification
        fuzzified = self.mf.fuzzify_all(temp, humidity, growth_stage)
        
        # Step 2: Rule evaluation and output computation
        heater_weighted_sum = 0.0
        misting_weighted_sum = 0.0
        total_weight = 0.0
        
        rule_activations = []  # For debugging/analysis
        
        for rule in self.rules:
            firing_strength = self.evaluate_rule(rule, fuzzified)
            
            if firing_strength > 0:
                # Compute rule outputs
                heater_output = rule.compute_heater_output(temp, humidity, growth_stage)
                misting_output = rule.compute_misting_output(temp, humidity, growth_stage)
                
                # Accumulate weighted outputs
                heater_weighted_sum += firing_strength * heater_output
                misting_weighted_sum += firing_strength * misting_output
                total_weight += firing_strength
                
                rule_activations.append({
                    'rule_id': rule.rule_id,
                    'firing_strength': firing_strength,
                    'heater_output': heater_output,
                    'misting_output': misting_output
                })
        
        # Step 3: Weighted average defuzzification
        if total_weight > 0:
            heater_final = heater_weighted_sum / total_weight
            misting_final = misting_weighted_sum / total_weight
        else:
            heater_final = 50.0  # Default neutral
            misting_final = 25.0
        
        return {
            'heater': float(np.clip(heater_final, 0, 100)),
            'misting': float(np.clip(misting_final, 0, 100)),
            'total_firing_strength': total_weight,
            'active_rules': len(rule_activations),
            'rule_activations': rule_activations
        }
    
    def get_control_surface(self, growth_stage: float = 50.0,
                           resolution: int = 50) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
        """
        Generate control surface for visualization.
        
        Returns:
            Tuple of (temp_grid, humidity_grid, heater_surface, misting_surface)
        """
        temps = np.linspace(0, 45, resolution)
        humidities = np.linspace(0, 100, resolution)
        
        heater_surface = np.zeros((resolution, resolution))
        misting_surface = np.zeros((resolution, resolution))
        
        for i, temp in enumerate(temps):
            for j, humidity in enumerate(humidities):
                result = self.infer(temp, humidity, growth_stage)
                heater_surface[j, i] = result['heater']
                misting_surface[j, i] = result['misting']
        
        temp_grid, humidity_grid = np.meshgrid(temps, humidities)
        return temp_grid, humidity_grid, heater_surface, misting_surface
    
    def update_rule_coefficients(self, rule_id: int, 
                                 heater_coeffs: Tuple[float, ...] = None,
                                 misting_coeffs: Tuple[float, ...] = None):
        """Update coefficients for a specific rule (used by optimizer)."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                if heater_coeffs is not None:
                    rule.heater_coeffs = heater_coeffs
                if misting_coeffs is not None:
                    rule.misting_coeffs = misting_coeffs
                break
    
    def print_rules(self):
        """Print all rules in human-readable format."""
        print("\n" + "="*80)
        print("SUGENO CONTROLLER RULE BASE")
        print("="*80)
        for rule in self.rules:
            print(rule)
        print("="*80 + "\n")


def plot_control_surface(controller: SugenoController, growth_stage: float = 50.0,
                        save_path: str = None):
    """Plot the control surface for the Sugeno controller."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    temp_grid, humidity_grid, heater_surface, misting_surface = \
        controller.get_control_surface(growth_stage)
    
    fig = plt.figure(figsize=(14, 6))
    
    # Heater control surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(temp_grid, humidity_grid, heater_surface,
                     cmap='coolwarm', alpha=0.8)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Humidity (%)')
    ax1.set_zlabel('Heater/Cooling (%)')
    ax1.set_title(f'Sugeno: Heater Control Surface\n(Growth Stage = {growth_stage})')
    
    # Misting control surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(temp_grid, humidity_grid, misting_surface,
                     cmap='Blues', alpha=0.8)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Humidity (%)')
    ax2.set_zlabel('Misting (%)')
    ax2.set_title(f'Sugeno: Misting Control Surface\n(Growth Stage = {growth_stage})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Control surface plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Demo: Create and test Sugeno controller
    controller = SugenoController()
    controller.print_rules()
    
    # Test inference
    test_cases = [
        (15, 40, 20),   # Cold, dry, seedling
        (25, 65, 50),   # Optimal conditions
        (35, 30, 80),   # Hot, dry, flowering
        (10, 85, 50),   # Very cold, very humid
    ]
    
    print("\nTest Inference Results:")
    print("-" * 60)
    for temp, humidity, stage in test_cases:
        result = controller.infer(temp, humidity, stage)
        print(f"Temp={temp}°C, Humidity={humidity}%, Stage={stage}")
        print(f"  → Heater: {result['heater']:.1f}%, Misting: {result['misting']:.1f}%")
        print(f"  → Active rules: {result['active_rules']}, "
              f"Total firing strength: {result['total_firing_strength']:.2f}")
    
    # Plot control surface
    plot_control_surface(controller)
