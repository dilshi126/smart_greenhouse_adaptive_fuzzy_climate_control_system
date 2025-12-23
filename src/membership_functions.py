"""
Membership Functions Module
===========================
Defines fuzzy membership functions for all input and output variables.

Part 2: Fuzzy System Design - Membership Functions
--------------------------------------------------
Each input variable has 5 fuzzy sets as required:

TEMPERATURE (0-45°C):
- Very Cold (VC): Below optimal, risk of frost damage
- Cold (C): Below optimal but safe
- Optimal (O): Ideal temperature range
- Warm (W): Above optimal but tolerable
- Hot (H): Above optimal, risk of heat stress

HUMIDITY (0-100%):
- Very Dry (VD): Risk of dehydration
- Dry (D): Below optimal moisture
- Optimal (O): Ideal humidity range
- Humid (H): Above optimal moisture
- Very Humid (VH): Risk of fungal diseases

GROWTH STAGE (0-100 encoded):
- Early Seedling (ES): Just germinated
- Late Seedling (LS): Established seedling
- Early Vegetative (EV): Beginning active growth
- Late Vegetative (LV): Peak growth phase
- Flowering (F): Reproductive phase

OUTPUT - Heater/Cooling Power (0-100%):
- Off, Low, Medium, High, Maximum

OUTPUT - Misting Intensity (0-100%):
- Off, Low, Medium, High, Maximum
"""

import numpy as np
from typing import Dict, Callable, Tuple
from dataclasses import dataclass


@dataclass
class FuzzySet:
    """Represents a fuzzy set with its membership function parameters."""
    name: str
    mf_type: str  # 'triangular', 'trapezoidal', 'gaussian'
    params: tuple  # Parameters for the membership function
    
    def __repr__(self):
        return f"FuzzySet({self.name}, {self.mf_type}, {self.params})"


class MembershipFunctions:
    """
    Fuzzy membership function definitions and calculations.
    
    JUSTIFICATION FOR MEMBERSHIP FUNCTION CHOICES:
    ==============================================
    
    1. TRIANGULAR MFs for Temperature and Humidity:
       - Simple and computationally efficient
       - Provides clear peak values for "optimal" conditions
       - Linear transitions match gradual climate changes
       - Easy to tune and interpret
    
    2. TRAPEZOIDAL MFs for extreme values (Very Cold, Very Hot, etc.):
       - Flat top represents range where condition is fully satisfied
       - Captures "saturation" behavior at extremes
       - Example: Below 5°C is equally "very cold" whether it's 5°C or 0°C
    
    3. GAUSSIAN MFs for Growth Stage:
       - Smooth transitions between stages
       - Represents natural biological progression
       - No sharp boundaries in plant development
    
    DEFUZZIFICATION METHOD JUSTIFICATION:
    =====================================
    - Mamdani: Centroid method - provides smooth, balanced output
    - Sugeno: Weighted average - computationally efficient, crisp output
    """
    
    def __init__(self, plant_name: str = "Tomato", growth_stage_value: float = 50.0):
        """
        Initialize membership functions.
        
        Args:
            plant_name: Name of the plant for adaptive parameter adjustment
            growth_stage_value: Numeric value of growth stage (0-100)
        """
        self.plant_name = plant_name
        self.growth_stage = growth_stage_value
        
        # Initialize default membership function parameters
        self._init_temperature_mfs()
        self._init_humidity_mfs()
        self._init_growth_stage_mfs()
        self._init_output_mfs()
    
    def _init_temperature_mfs(self):
        """Initialize temperature membership functions (5 sets)."""
        # Default parameters - can be adapted based on plant type
        self.temp_mfs = {
            'very_cold': FuzzySet('Very Cold', 'trapezoidal', (0, 0, 5, 12)),
            'cold': FuzzySet('Cold', 'triangular', (8, 14, 20)),
            'optimal': FuzzySet('Optimal', 'triangular', (18, 23, 28)),
            'warm': FuzzySet('Warm', 'triangular', (26, 32, 38)),
            'hot': FuzzySet('Hot', 'trapezoidal', (35, 40, 45, 45)),
        }
        self.temp_range = (0, 45)
    
    def _init_humidity_mfs(self):
        """Initialize humidity membership functions (5 sets)."""
        self.humidity_mfs = {
            'very_dry': FuzzySet('Very Dry', 'trapezoidal', (0, 0, 15, 30)),
            'dry': FuzzySet('Dry', 'triangular', (20, 35, 50)),
            'optimal': FuzzySet('Optimal', 'triangular', (45, 65, 80)),
            'humid': FuzzySet('Humid', 'triangular', (70, 80, 90)),
            'very_humid': FuzzySet('Very Humid', 'trapezoidal', (85, 92, 100, 100)),
        }
        self.humidity_range = (0, 100)
    
    def _init_growth_stage_mfs(self):
        """Initialize growth stage membership functions (5 sets)."""
        self.stage_mfs = {
            'early_seedling': FuzzySet('Early Seedling', 'gaussian', (10, 8)),
            'late_seedling': FuzzySet('Late Seedling', 'gaussian', (30, 8)),
            'early_vegetative': FuzzySet('Early Vegetative', 'gaussian', (50, 8)),
            'late_vegetative': FuzzySet('Late Vegetative', 'gaussian', (70, 8)),
            'flowering': FuzzySet('Flowering', 'gaussian', (90, 8)),
        }
        self.stage_range = (0, 100)
    
    def _init_output_mfs(self):
        """Initialize output membership functions for both outputs."""
        # Heater/Cooling Power (negative = cooling, positive = heating conceptually)
        # Output range 0-100%: 0-40 cooling, 40-60 off, 60-100 heating
        self.heater_mfs = {
            'cool_high': FuzzySet('Cool High', 'trapezoidal', (0, 0, 10, 25)),
            'cool_low': FuzzySet('Cool Low', 'triangular', (15, 30, 45)),
            'off': FuzzySet('Off', 'triangular', (40, 50, 60)),
            'heat_low': FuzzySet('Heat Low', 'triangular', (55, 70, 85)),
            'heat_high': FuzzySet('Heat High', 'trapezoidal', (75, 90, 100, 100)),
        }
        
        # Misting System Intensity
        self.misting_mfs = {
            'off': FuzzySet('Off', 'trapezoidal', (0, 0, 5, 15)),
            'low': FuzzySet('Low', 'triangular', (10, 25, 40)),
            'medium': FuzzySet('Medium', 'triangular', (35, 50, 65)),
            'high': FuzzySet('High', 'triangular', (60, 75, 90)),
            'maximum': FuzzySet('Maximum', 'trapezoidal', (85, 95, 100, 100)),
        }
        self.output_range = (0, 100)
    
    # ==================== Membership Function Calculations ====================
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function.
        
        Args:
            x: Input value
            a: Left foot
            b: Peak
            c: Right foot
        
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function.
        
        Args:
            x: Input value
            a: Left foot
            b: Left shoulder
            c: Right shoulder
            d: Right foot
        
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x: float, mean: float, sigma: float) -> float:
        """
        Gaussian membership function.
        
        Args:
            x: Input value
            mean: Center of the gaussian
            sigma: Standard deviation (width)
        
        Returns:
            Membership degree [0, 1]
        """
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    
    def calculate_membership(self, fuzzy_set: FuzzySet, x: float) -> float:
        """Calculate membership degree for a given fuzzy set and input value."""
        if fuzzy_set.mf_type == 'triangular':
            return self.triangular(x, *fuzzy_set.params)
        elif fuzzy_set.mf_type == 'trapezoidal':
            return self.trapezoidal(x, *fuzzy_set.params)
        elif fuzzy_set.mf_type == 'gaussian':
            return self.gaussian(x, *fuzzy_set.params)
        else:
            raise ValueError(f"Unknown membership function type: {fuzzy_set.mf_type}")
    
    def fuzzify_temperature(self, temp: float) -> Dict[str, float]:
        """Fuzzify temperature input."""
        return {name: self.calculate_membership(fs, temp) 
                for name, fs in self.temp_mfs.items()}
    
    def fuzzify_humidity(self, humidity: float) -> Dict[str, float]:
        """Fuzzify humidity input."""
        return {name: self.calculate_membership(fs, humidity) 
                for name, fs in self.humidity_mfs.items()}
    
    def fuzzify_growth_stage(self, stage: float) -> Dict[str, float]:
        """Fuzzify growth stage input."""
        return {name: self.calculate_membership(fs, stage) 
                for name, fs in self.stage_mfs.items()}
    
    def fuzzify_all(self, temp: float, humidity: float, stage: float) -> Dict[str, Dict[str, float]]:
        """Fuzzify all inputs at once."""
        return {
            'temperature': self.fuzzify_temperature(temp),
            'humidity': self.fuzzify_humidity(humidity),
            'growth_stage': self.fuzzify_growth_stage(stage),
        }
    
    def adapt_to_plant(self, plant_name: str, optimal_temp: float, optimal_humidity: float):
        """
        Adapt membership functions based on plant requirements.
        
        This shifts the 'optimal' membership functions to center on the
        plant's ideal conditions.
        """
        self.plant_name = plant_name
        
        # Shift temperature optimal MF
        temp_shift = optimal_temp - 23  # 23 is default optimal center
        self.temp_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular', 
            (18 + temp_shift, 23 + temp_shift, 28 + temp_shift)
        )
        
        # Shift humidity optimal MF
        humidity_shift = optimal_humidity - 65  # 65 is default optimal center
        self.humidity_mfs['optimal'] = FuzzySet(
            'Optimal', 'triangular',
            (45 + humidity_shift, 65 + humidity_shift, 80 + humidity_shift)
        )
    
    def get_mf_parameters(self) -> Dict:
        """Get all membership function parameters for optimization."""
        return {
            'temperature': {name: fs.params for name, fs in self.temp_mfs.items()},
            'humidity': {name: fs.params for name, fs in self.humidity_mfs.items()},
            'growth_stage': {name: fs.params for name, fs in self.stage_mfs.items()},
        }
    
    def set_mf_parameters(self, params: Dict):
        """Set membership function parameters (used by optimizer)."""
        if 'temperature' in params:
            for name, p in params['temperature'].items():
                if name in self.temp_mfs:
                    self.temp_mfs[name] = FuzzySet(
                        self.temp_mfs[name].name,
                        self.temp_mfs[name].mf_type,
                        tuple(p)
                    )
        
        if 'humidity' in params:
            for name, p in params['humidity'].items():
                if name in self.humidity_mfs:
                    self.humidity_mfs[name] = FuzzySet(
                        self.humidity_mfs[name].name,
                        self.humidity_mfs[name].mf_type,
                        tuple(p)
                    )


def plot_membership_functions(mf: MembershipFunctions, save_path: str = None):
    """Plot all membership functions for visualization."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Temperature MFs
    ax = axes[0, 0]
    x_temp = np.linspace(0, 45, 200)
    for name, fs in mf.temp_mfs.items():
        y = [mf.calculate_membership(fs, x) for x in x_temp]
        ax.plot(x_temp, y, label=name)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Temperature Membership Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Humidity MFs
    ax = axes[0, 1]
    x_hum = np.linspace(0, 100, 200)
    for name, fs in mf.humidity_mfs.items():
        y = [mf.calculate_membership(fs, x) for x in x_hum]
        ax.plot(x_hum, y, label=name)
    ax.set_xlabel('Humidity (%)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Humidity Membership Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Growth Stage MFs
    ax = axes[0, 2]
    x_stage = np.linspace(0, 100, 200)
    for name, fs in mf.stage_mfs.items():
        y = [mf.calculate_membership(fs, x) for x in x_stage]
        ax.plot(x_stage, y, label=name)
    ax.set_xlabel('Growth Stage (encoded)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Growth Stage Membership Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heater Output MFs
    ax = axes[1, 0]
    x_out = np.linspace(0, 100, 200)
    for name, fs in mf.heater_mfs.items():
        y = [mf.calculate_membership(fs, x) for x in x_out]
        ax.plot(x_out, y, label=name)
    ax.set_xlabel('Heater/Cooling Power (%)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Heater/Cooling Output MFs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Misting Output MFs
    ax = axes[1, 1]
    for name, fs in mf.misting_mfs.items():
        y = [mf.calculate_membership(fs, x) for x in x_out]
        ax.plot(x_out, y, label=name)
    ax.set_xlabel('Misting Intensity (%)')
    ax.set_ylabel('Membership Degree')
    ax.set_title('Misting Output MFs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hide empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Membership functions plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Demo: Create and visualize membership functions
    mf = MembershipFunctions()
    
    # Test fuzzification
    temp, humidity, stage = 25, 70, 50
    print(f"\nFuzzification of inputs: Temp={temp}°C, Humidity={humidity}%, Stage={stage}")
    fuzzy_values = mf.fuzzify_all(temp, humidity, stage)
    
    for var_name, memberships in fuzzy_values.items():
        print(f"\n{var_name}:")
        for set_name, degree in memberships.items():
            if degree > 0:
                print(f"  {set_name}: {degree:.3f}")
    
    # Plot membership functions
    plot_membership_functions(mf)
