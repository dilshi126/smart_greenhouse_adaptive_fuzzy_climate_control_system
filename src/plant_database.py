"""
Plant Database Module
=====================
Defines plant species, growth stages, and their ideal climate requirements.

Part 1: System Modeling
-----------------------
Three plant species selected:
1. Tomato - Common greenhouse crop with moderate requirements
2. Lettuce - Cool-season crop with lower temperature needs
3. Orchid - Tropical plant with high humidity requirements

Each plant has different needs across three growth stages:
- Seedling: Initial growth phase, requires stable conditions
- Vegetative: Active growth phase, higher nutrient/water needs
- Flowering: Reproductive phase, specific temperature triggers needed
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple


class GrowthStage(Enum):
    """Plant growth stages with numeric encoding for fuzzy input."""
    SEEDLING = 0      # 0-33% of growth stage input range
    VEGETATIVE = 1    # 34-66% of growth stage input range
    FLOWERING = 2     # 67-100% of growth stage input range


@dataclass
class ClimateRequirements:
    """Ideal climate requirements for a plant at a specific growth stage."""
    temp_min: float      # Minimum ideal temperature (°C)
    temp_optimal: float  # Optimal temperature (°C)
    temp_max: float      # Maximum ideal temperature (°C)
    humidity_min: float  # Minimum ideal humidity (%)
    humidity_optimal: float  # Optimal humidity (%)
    humidity_max: float  # Maximum ideal humidity (%)
    
    def __repr__(self):
        return (f"Temp: {self.temp_min}-{self.temp_optimal}-{self.temp_max}°C, "
                f"Humidity: {self.humidity_min}-{self.humidity_optimal}-{self.humidity_max}%")


class PlantDatabase:
    """
    Database of plant species and their climate requirements.
    
    COMPARISON TABLE - Climate Needs by Plant and Growth Stage:
    ============================================================
    
    | Plant   | Stage      | Temp Range (°C) | Optimal Temp | Humidity Range | Optimal Humidity |
    |---------|------------|-----------------|--------------|----------------|------------------|
    | Tomato  | Seedling   | 20-25          | 22           | 65-75%         | 70%              |
    | Tomato  | Vegetative | 22-28          | 25           | 60-70%         | 65%              |
    | Tomato  | Flowering  | 18-24          | 21           | 55-65%         | 60%              |
    | Lettuce | Seedling   | 15-20          | 18           | 70-80%         | 75%              |
    | Lettuce | Vegetative | 12-18          | 15           | 65-75%         | 70%              |
    | Lettuce | Flowering  | 10-16          | 13           | 60-70%         | 65%              |
    | Orchid  | Seedling   | 22-28          | 25           | 75-85%         | 80%              |
    | Orchid  | Vegetative | 20-26          | 23           | 70-80%         | 75%              |
    | Orchid  | Flowering  | 18-24          | 21           | 65-75%         | 70%              |
    
    WHY FUZZY LOGIC IS BETTER THAN CRISP LOGIC OR PID:
    ==================================================
    
    1. HANDLING UNCERTAINTY: Greenhouse conditions are inherently uncertain.
       Temperature sensors have noise, humidity varies spatially. Fuzzy logic
       naturally handles this imprecision through membership functions.
    
    2. MULTIPLE INPUTS/OUTPUTS: PID controllers are designed for SISO systems.
       Our system has 3 inputs and 2 outputs with complex interactions.
       Fuzzy logic handles MIMO systems elegantly through rule bases.
    
    3. NONLINEAR RELATIONSHIPS: The relationship between temperature/humidity
       and plant health is nonlinear. Fuzzy rules capture expert knowledge
       about these nonlinear relationships naturally.
    
    4. LINGUISTIC RULES: Domain experts (botanists) think in terms like
       "if temperature is high and humidity is low, increase misting."
       Fuzzy logic directly encodes this knowledge.
    
    5. SMOOTH TRANSITIONS: Unlike crisp logic with hard boundaries,
       fuzzy logic provides smooth control transitions, reducing
       equipment wear and plant stress.
    
    6. ADAPTABILITY: Fuzzy systems can be easily modified by adjusting
       membership functions or rules without redesigning the entire system.
    
    7. NO MATHEMATICAL MODEL REQUIRED: PID requires a mathematical model
       of the plant. Fuzzy logic works with qualitative understanding.
    """
    
    # Plant climate requirements database
    PLANTS: Dict[str, Dict[GrowthStage, ClimateRequirements]] = {
        "Tomato": {
            GrowthStage.SEEDLING: ClimateRequirements(20, 22, 25, 65, 70, 75),
            GrowthStage.VEGETATIVE: ClimateRequirements(22, 25, 28, 60, 65, 70),
            GrowthStage.FLOWERING: ClimateRequirements(18, 21, 24, 55, 60, 65),
        },
        "Lettuce": {
            GrowthStage.SEEDLING: ClimateRequirements(15, 18, 20, 70, 75, 80),
            GrowthStage.VEGETATIVE: ClimateRequirements(12, 15, 18, 65, 70, 75),
            GrowthStage.FLOWERING: ClimateRequirements(10, 13, 16, 60, 65, 70),
        },
        "Orchid": {
            GrowthStage.SEEDLING: ClimateRequirements(22, 25, 28, 75, 80, 85),
            GrowthStage.VEGETATIVE: ClimateRequirements(20, 23, 26, 70, 75, 80),
            GrowthStage.FLOWERING: ClimateRequirements(18, 21, 24, 65, 70, 75),
        },
    }
    
    @classmethod
    def get_requirements(cls, plant_name: str, stage: GrowthStage) -> ClimateRequirements:
        """Get climate requirements for a specific plant and growth stage."""
        if plant_name not in cls.PLANTS:
            raise ValueError(f"Unknown plant: {plant_name}. Available: {list(cls.PLANTS.keys())}")
        return cls.PLANTS[plant_name][stage]
    
    @classmethod
    def get_all_plants(cls) -> list:
        """Get list of all available plant names."""
        return list(cls.PLANTS.keys())
    
    @classmethod
    def get_stage_numeric(cls, stage: GrowthStage) -> float:
        """Convert growth stage to numeric value (0-100 scale)."""
        stage_map = {
            GrowthStage.SEEDLING: 16.5,    # Center of 0-33 range
            GrowthStage.VEGETATIVE: 50.0,  # Center of 34-66 range
            GrowthStage.FLOWERING: 83.5,   # Center of 67-100 range
        }
        return stage_map[stage]
    
    @classmethod
    def print_comparison_table(cls):
        """Print a formatted comparison table of all plant requirements."""
        print("\n" + "="*90)
        print("PLANT CLIMATE REQUIREMENTS COMPARISON TABLE")
        print("="*90)
        print(f"{'Plant':<10} {'Stage':<12} {'Temp Range':<15} {'Optimal Temp':<14} "
              f"{'Humidity Range':<16} {'Optimal Humidity'}")
        print("-"*90)
        
        for plant_name, stages in cls.PLANTS.items():
            for stage, req in stages.items():
                print(f"{plant_name:<10} {stage.name:<12} "
                      f"{req.temp_min}-{req.temp_max}°C{'':<7} "
                      f"{req.temp_optimal}°C{'':<10} "
                      f"{req.humidity_min}-{req.humidity_max}%{'':<8} "
                      f"{req.humidity_optimal}%")
        print("="*90 + "\n")


if __name__ == "__main__":
    # Demo: Print the comparison table
    PlantDatabase.print_comparison_table()
    
    # Example usage
    tomato_seedling = PlantDatabase.get_requirements("Tomato", GrowthStage.SEEDLING)
    print(f"Tomato Seedling Requirements: {tomato_seedling}")
