"""
Mamdani Fuzzy Controller Module
===============================
Implements a Mamdani-type fuzzy inference system for greenhouse climate control.

Part 2: Fuzzy System Design - Mamdani Controller
------------------------------------------------
The Mamdani controller uses:
- MIN operator for AND (conjunction)
- MAX operator for OR (disjunction)
- MIN for implication (clipping)
- MAX for aggregation
- CENTROID for defuzzification

JUSTIFICATION FOR MAMDANI:
==========================
1. Intuitive rule interpretation - rules directly map to human reasoning
2. Output is a fuzzy set - provides richer information about control action
3. Well-suited for systems where expert knowledge is available
4. Smooth control surface due to fuzzy output aggregation
5. Better for systems requiring gradual transitions (plant care)

RULE BASE DESIGN (27+ rules):
=============================
Rules are designed based on:
- Temperature deviation from optimal
- Humidity deviation from optimal
- Growth stage requirements (seedlings need more stable conditions)
"""

import numpy as np
from typing import Dict, List, Tuple
from .membership_functions import MembershipFunctions, FuzzySet


class FuzzyRule:
    """Represents a single fuzzy rule."""
    
    def __init__(self, rule_id: int, antecedents: Dict[str, str], 
                 consequents: Dict[str, str], weight: float = 1.0):
        """
        Initialize a fuzzy rule.
        
        Args:
            rule_id: Unique identifier for the rule
            antecedents: Dict mapping input variable to fuzzy set name
                        e.g., {'temperature': 'cold', 'humidity': 'dry'}
            consequents: Dict mapping output variable to fuzzy set name
                        e.g., {'heater': 'heat_high', 'misting': 'high'}
            weight: Rule weight (0-1), default 1.0
        """
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.consequents = consequents
        self.weight = weight
    
    def __repr__(self):
        ant_str = " AND ".join([f"{k} IS {v}" for k, v in self.antecedents.items()])
        con_str = ", ".join([f"{k} IS {v}" for k, v in self.consequents.items()])
        return f"Rule {self.rule_id}: IF {ant_str} THEN {con_str}"


class MamdaniController:
    """
    Mamdani-type fuzzy inference system for greenhouse control.
    
    FUZZY RULE BASE (27 rules covering all input combinations):
    ===========================================================
    
    The rules are organized by temperature condition, then humidity,
    with growth stage modifying the response intensity.
    
    Rule Design Philosophy:
    - Very Cold + Any Humidity → Maximum Heating (protect plants)
    - Cold + Dry → High Heating + Medium Misting
    - Cold + Humid → Medium Heating + Low Misting
    - Optimal + Optimal → Maintain (Off/Low)
    - Warm + Dry → Low Cooling + High Misting
    - Hot + Any → Maximum Cooling + Misting based on humidity
    - Seedling stage → More conservative control (smaller changes)
    - Flowering stage → Tighter control (plants are sensitive)
    """
    
    def __init__(self, membership_functions: MembershipFunctions = None):
        """Initialize the Mamdani controller."""
        self.mf = membership_functions or MembershipFunctions()
        self.rules = self._create_rule_base()
        self.output_resolution = 100  # Points for output universe
    
    def _create_rule_base(self) -> List[FuzzyRule]:
        """
        Create the fuzzy rule base with 27+ rules.
        
        Rules cover combinations of:
        - 5 temperature levels × 5 humidity levels = 25 base rules
        - Additional rules for growth stage modifications
        """
        rules = []
        rule_id = 1
        
        # ============== TEMPERATURE-HUMIDITY RULES ==============
        
        # VERY COLD temperature rules (5 rules)
        rules.append(FuzzyRule(rule_id, 
            {'temperature': 'very_cold', 'humidity': 'very_dry'},
            {'heater': 'heat_high', 'misting': 'high'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'dry'},
            {'heater': 'heat_high', 'misting': 'medium'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'optimal'},
            {'heater': 'heat_high', 'misting': 'low'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'humid'},
            {'heater': 'heat_high', 'misting': 'off'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'very_cold', 'humidity': 'very_humid'},
            {'heater': 'heat_high', 'misting': 'off'}))
        rule_id += 1
        
        # COLD temperature rules (5 rules)
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'humidity': 'very_dry'},
            {'heater': 'heat_low', 'misting': 'high'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'humidity': 'dry'},
            {'heater': 'heat_low', 'misting': 'medium'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'humidity': 'optimal'},
            {'heater': 'heat_low', 'misting': 'low'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'humidity': 'humid'},
            {'heater': 'heat_low', 'misting': 'off'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'humidity': 'very_humid'},
            {'heater': 'heat_low', 'misting': 'off'}))
        rule_id += 1
        
        # OPTIMAL temperature rules (5 rules)
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'very_dry'},
            {'heater': 'off', 'misting': 'high'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'dry'},
            {'heater': 'off', 'misting': 'medium'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'optimal'},
            {'heater': 'off', 'misting': 'low'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'humid'},
            {'heater': 'off', 'misting': 'off'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'humidity': 'very_humid'},
            {'heater': 'off', 'misting': 'off'}))
        rule_id += 1
        
        # WARM temperature rules (5 rules)
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'humidity': 'very_dry'},
            {'heater': 'cool_low', 'misting': 'maximum'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'humidity': 'dry'},
            {'heater': 'cool_low', 'misting': 'high'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'humidity': 'optimal'},
            {'heater': 'cool_low', 'misting': 'medium'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'humidity': 'humid'},
            {'heater': 'cool_low', 'misting': 'low'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'humidity': 'very_humid'},
            {'heater': 'cool_low', 'misting': 'off'}))
        rule_id += 1
        
        # HOT temperature rules (5 rules)
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'hot', 'humidity': 'very_dry'},
            {'heater': 'cool_high', 'misting': 'maximum'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'hot', 'humidity': 'dry'},
            {'heater': 'cool_high', 'misting': 'maximum'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'hot', 'humidity': 'optimal'},
            {'heater': 'cool_high', 'misting': 'high'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'hot', 'humidity': 'humid'},
            {'heater': 'cool_high', 'misting': 'medium'}))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'hot', 'humidity': 'very_humid'},
            {'heater': 'cool_high', 'misting': 'low'}))
        rule_id += 1
        
        # ============== GROWTH STAGE MODIFIER RULES ==============
        # These rules modify control intensity based on growth stage
        
        # Seedling rules - more conservative control
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'cold', 'growth_stage': 'early_seedling'},
            {'heater': 'heat_high', 'misting': 'medium'}, weight=0.8))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'growth_stage': 'early_seedling'},
            {'heater': 'cool_low', 'misting': 'high'}, weight=0.8))
        rule_id += 1
        
        # Flowering rules - tighter control for sensitive phase
        rules.append(FuzzyRule(rule_id,
            {'humidity': 'humid', 'growth_stage': 'flowering'},
            {'heater': 'off', 'misting': 'off'}, weight=0.9))
        rule_id += 1
        
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'optimal', 'growth_stage': 'flowering'},
            {'heater': 'off', 'misting': 'low'}, weight=0.9))
        rule_id += 1
        
        # Vegetative growth - can tolerate more variation
        rules.append(FuzzyRule(rule_id,
            {'temperature': 'warm', 'growth_stage': 'late_vegetative'},
            {'heater': 'off', 'misting': 'medium'}, weight=0.7))
        rule_id += 1
        
        print(f"Created {len(rules)} fuzzy rules for Mamdani controller")
        return rules
    
    def evaluate_rule(self, rule: FuzzyRule, 
                      fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate a single rule's firing strength.
        
        Uses MIN operator for AND conjunction of antecedents.
        """
        firing_strength = 1.0
        
        for var_name, fuzzy_set_name in rule.antecedents.items():
            if var_name in fuzzified_inputs:
                membership = fuzzified_inputs[var_name].get(fuzzy_set_name, 0.0)
                firing_strength = min(firing_strength, membership)
        
        return firing_strength * rule.weight
    
    def infer(self, temp: float, humidity: float, growth_stage: float) -> Dict[str, float]:
        """
        Perform Mamdani fuzzy inference.
        
        Args:
            temp: Temperature in °C
            humidity: Humidity in %
            growth_stage: Growth stage (0-100 encoded)
        
        Returns:
            Dict with 'heater' and 'misting' crisp output values (0-100%)
        """
        # Step 1: Fuzzification
        fuzzified = self.mf.fuzzify_all(temp, humidity, growth_stage)
        
        # Step 2: Rule evaluation and implication
        # Create output universes
        output_universe = np.linspace(0, 100, self.output_resolution)
        heater_aggregated = np.zeros(self.output_resolution)
        misting_aggregated = np.zeros(self.output_resolution)
        
        for rule in self.rules:
            firing_strength = self.evaluate_rule(rule, fuzzified)
            
            if firing_strength > 0:
                # Apply implication (MIN clipping) and aggregate (MAX)
                if 'heater' in rule.consequents:
                    heater_set_name = rule.consequents['heater']
                    heater_fs = self.mf.heater_mfs[heater_set_name]
                    for i, x in enumerate(output_universe):
                        mf_value = self.mf.calculate_membership(heater_fs, x)
                        clipped = min(firing_strength, mf_value)
                        heater_aggregated[i] = max(heater_aggregated[i], clipped)
                
                if 'misting' in rule.consequents:
                    misting_set_name = rule.consequents['misting']
                    misting_fs = self.mf.misting_mfs[misting_set_name]
                    for i, x in enumerate(output_universe):
                        mf_value = self.mf.calculate_membership(misting_fs, x)
                        clipped = min(firing_strength, mf_value)
                        misting_aggregated[i] = max(misting_aggregated[i], clipped)
        
        # Step 3: Defuzzification using centroid method
        heater_output = self._centroid_defuzzify(output_universe, heater_aggregated)
        misting_output = self._centroid_defuzzify(output_universe, misting_aggregated)
        
        return {
            'heater': heater_output,
            'misting': misting_output,
            'heater_aggregated': heater_aggregated,
            'misting_aggregated': misting_aggregated,
            'output_universe': output_universe
        }
    
    def _centroid_defuzzify(self, universe: np.ndarray, 
                            aggregated: np.ndarray) -> float:
        """
        Centroid (center of gravity) defuzzification.
        
        JUSTIFICATION:
        - Most commonly used method
        - Provides smooth, continuous output
        - Considers entire shape of aggregated fuzzy set
        - Physically interpretable as "center of mass"
        """
        total_area = np.sum(aggregated)
        if total_area == 0:
            return 50.0  # Default to middle if no rules fire
        
        centroid = np.sum(universe * aggregated) / total_area
        return float(centroid)
    
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
    
    def print_rules(self):
        """Print all rules in human-readable format."""
        print("\n" + "="*70)
        print("MAMDANI CONTROLLER RULE BASE")
        print("="*70)
        for rule in self.rules:
            print(rule)
        print("="*70 + "\n")


def plot_control_surface(controller: MamdaniController, growth_stage: float = 50.0,
                        save_path: str = None):
    """Plot the control surface for the Mamdani controller."""
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
    ax1.set_title(f'Mamdani: Heater Control Surface\n(Growth Stage = {growth_stage})')
    
    # Misting control surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(temp_grid, humidity_grid, misting_surface,
                     cmap='Blues', alpha=0.8)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Humidity (%)')
    ax2.set_zlabel('Misting (%)')
    ax2.set_title(f'Mamdani: Misting Control Surface\n(Growth Stage = {growth_stage})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Control surface plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Demo: Create and test Mamdani controller
    controller = MamdaniController()
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
    
    # Plot control surface
    plot_control_surface(controller)
