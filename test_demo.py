"""Simple test script to verify the system works."""
from src import (
    PlantDatabase, GrowthStage, MembershipFunctions,
    MamdaniController, SugenoController, AdaptiveFuzzySystem,
    GreenhouseSimulator
)

print("=" * 70)
print("SMART GREENHOUSE FUZZY CONTROL SYSTEM - DEMO")
print("=" * 70)

# 1. Plant Database
print("\n[1] PLANT DATABASE")
print("-" * 50)
PlantDatabase.print_comparison_table()

# 2. Membership Functions
print("\n[2] MEMBERSHIP FUNCTIONS TEST")
print("-" * 50)
mf = MembershipFunctions()
fuzzy = mf.fuzzify_all(25, 65, 50)
print("Fuzzification at Temp=25C, Humidity=65%, Stage=50:")
for var, vals in fuzzy.items():
    active = {k: round(v, 3) for k, v in vals.items() if v > 0.01}
    if active:
        print(f"  {var}: {active}")

# 3. Controllers
print("\n[3] CONTROLLER COMPARISON")
print("-" * 50)
mamdani = MamdaniController()
sugeno = SugenoController()

test_cases = [
    (15, 40, 20, "Cold & Dry (Seedling)"),
    (25, 65, 50, "Optimal (Vegetative)"),
    (35, 30, 80, "Hot & Dry (Flowering)"),
]

print(f"\n{'Condition':<25} {'Mamdani H':<12} {'Mamdani M':<12} {'Sugeno H':<12} {'Sugeno M'}")
print("-" * 75)
for temp, humidity, stage, desc in test_cases:
    m = mamdani.infer(temp, humidity, stage)
    s = sugeno.infer(temp, humidity, stage)
    print(f"{desc:<25} {m['heater']:>10.1f}% {m['misting']:>10.1f}% {s['heater']:>10.1f}% {s['misting']:>9.1f}%")

# 4. Adaptive System
print("\n[4] ADAPTIVE SYSTEM")
print("-" * 50)
adaptive = AdaptiveFuzzySystem("Tomato", GrowthStage.SEEDLING)
result = adaptive.control(20, 70)
print(f"Tomato Seedling at 20C, 70%:")
print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, Misting={result['mamdani']['misting']:.1f}%")
print(f"  Sugeno:  Heater={result['sugeno']['heater']:.1f}%, Misting={result['sugeno']['misting']:.1f}%")

adaptive.change_plant("Orchid")
result = adaptive.control(20, 70)
print(f"\nOrchid at 20C, 70%:")
print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, Misting={result['mamdani']['misting']:.1f}%")

# 5. Simulation
print("\n[5] SIMULATION")
print("-" * 50)
simulator = GreenhouseSimulator()
sim = simulator.run_simulation("Tomato", GrowthStage.VEGETATIVE, "heat_wave", duration=50)
print(f"Simulation: {sim.plant} ({sim.growth_stage}) - {sim.weather_pattern}")
print(f"Temp range: {sim.temps.min():.1f}C - {sim.temps.max():.1f}C")
print(f"\nMamdani Metrics:")
for k, v in sim.mamdani_metrics.items():
    print(f"  {k}: {v:.4f}")
print(f"\nSugeno Metrics:")
for k, v in sim.sugeno_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
