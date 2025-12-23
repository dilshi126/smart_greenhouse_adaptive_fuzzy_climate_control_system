"""
Complete Project Demonstration
==============================
Runs all components of the Smart Greenhouse Adaptive Fuzzy Control System.
"""

import sys
import numpy as np

print("=" * 80)
print("SMART GREENHOUSE ADAPTIVE FUZZY CLIMATE CONTROL SYSTEM")
print("Complete Project Demonstration")
print("=" * 80)

# Import all modules
from src import (
    PlantDatabase, GrowthStage, MembershipFunctions,
    MamdaniController, SugenoController, AdaptiveFuzzySystem,
    GreenhouseSimulator, PSOOptimizer, FuzzyRLAgent
)

# ============================================================================
# PART 1: SYSTEM MODELING
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: SYSTEM MODELING")
print("=" * 80)

print("\n[1.1] Plant Species and Climate Requirements:")
PlantDatabase.print_comparison_table()

print("\n[1.2] Why Fuzzy Logic is Better than PID:")
print("""
  1. UNCERTAINTY HANDLING: Fuzzy logic naturally handles sensor noise and imprecision
  2. MULTI-VARIABLE: Single rule base handles temp, humidity, and growth stage together
  3. NON-LINEAR: Captures complex plant-climate relationships through rules
  4. EXPERT KNOWLEDGE: Directly encodes linguistic rules from botanists
  5. SMOOTH CONTROL: Gradual transitions reduce plant stress and equipment wear
  6. ADAPTABILITY: Easy to modify rules without redesigning the system
""")

# ============================================================================
# PART 2: FUZZY SYSTEM DESIGN
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: FUZZY SYSTEM DESIGN")
print("=" * 80)

print("\n[2.1] Membership Functions (5 per input):")
mf = MembershipFunctions()
print("  Temperature: Very Cold, Cold, Optimal, Warm, Hot")
print("  Humidity: Very Dry, Dry, Optimal, Humid, Very Humid")
print("  Growth Stage: Early Seedling, Late Seedling, Early Vegetative, Late Vegetative, Flowering")

print("\n[2.2] Fuzzification Example (Temp=25°C, Humidity=65%, Stage=50):")
fuzzy = mf.fuzzify_all(25, 65, 50)
for var, vals in fuzzy.items():
    active = {k: f"{v:.3f}" for k, v in vals.items() if v > 0.01}
    if active:
        print(f"  {var}: {active}")

print("\n[2.3] Controllers Created:")
mamdani = MamdaniController()
sugeno = SugenoController()
print(f"  Mamdani: {len(mamdani.rules)} rules, Centroid defuzzification")
print(f"  Sugeno: {len(sugeno.rules)} rules, Weighted average defuzzification")

# ============================================================================
# PART 3: PROGRAMMING IMPLEMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: PROGRAMMING IMPLEMENTATION")
print("=" * 80)

print("\n[3.1] Controller Comparison Test:")
test_cases = [
    (15, 40, 20, "Cold & Dry (Seedling)"),
    (25, 65, 50, "Optimal (Vegetative)"),
    (35, 30, 80, "Hot & Dry (Flowering)"),
    (10, 90, 50, "Very Cold & Humid"),
]

print(f"\n{'Condition':<25} {'Mamdani H':<12} {'Mamdani M':<12} {'Sugeno H':<12} {'Sugeno M'}")
print("-" * 75)
for temp, humidity, stage, desc in test_cases:
    m = mamdani.infer(temp, humidity, stage)
    s = sugeno.infer(temp, humidity, stage)
    print(f"{desc:<25} {m['heater']:>10.1f}% {m['misting']:>10.1f}% {s['heater']:>10.1f}% {s['misting']:>9.1f}%")

# ============================================================================
# PART 4: DYNAMIC ADAPTATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: DYNAMIC ADAPTATION")
print("=" * 80)

print("\n[4.1] Adaptive System Initialization:")
adaptive = AdaptiveFuzzySystem("Tomato", GrowthStage.SEEDLING)
info = adaptive.get_adaptation_info()
print(f"  Plant: {info['plant']}, Stage: {info['growth_stage']}")
print(f"  Mode: {info['mode']}, Output Scale: {info['output_scale']}")
print(f"  Optimal: {info['optimal_temp']}°C, {info['optimal_humidity']}%")

print("\n[4.2] Control Output at 20°C, 70%:")
result = adaptive.control(20, 70)
print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, Misting={result['mamdani']['misting']:.1f}%")
print(f"  Sugeno:  Heater={result['sugeno']['heater']:.1f}%, Misting={result['sugeno']['misting']:.1f}%")

print("\n[4.3] Adaptation to Different Plant (Orchid):")
adaptive.change_plant("Orchid")
info = adaptive.get_adaptation_info()
print(f"  New Optimal: {info['optimal_temp']}°C, {info['optimal_humidity']}%")
result = adaptive.control(20, 70)
print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, Misting={result['mamdani']['misting']:.1f}%")

print("\n[4.4] Adaptation to Flowering Stage:")
adaptive.change_growth_stage(GrowthStage.FLOWERING)
info = adaptive.get_adaptation_info()
print(f"  Mode changed to: {info['mode']}, Scale: {info['output_scale']}")

# ============================================================================
# PART 5: PERFORMANCE EVALUATION (20+ Random Tests)
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: PERFORMANCE EVALUATION")
print("=" * 80)

print("\n[5.1] Running 20 Random Simulations...")
simulator = GreenhouseSimulator()
results = simulator.run_random_tests(20)

print("\n[5.2] Performance Comparison Table:")
table = simulator.generate_comparison_table(results)
print(table)

# ============================================================================
# PART 6: OPTIMIZATION (PSO)
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: PSO OPTIMIZATION")
print("=" * 80)

print("\n[6.1] Running PSO Optimization (30 iterations)...")
optimizer = PSOOptimizer(
    controller_type="mamdani",
    num_particles=15,
    max_iterations=30
)
best_params, best_fitness = optimizer.optimize("Tomato", GrowthStage.VEGETATIVE, verbose=True)

print("\n[6.2] Before vs After Optimization:")
comparison = optimizer.compare_before_after("Tomato", GrowthStage.VEGETATIVE, best_params)
print(f"  Default Error: {comparison['default_error']:.4f}")
print(f"  Optimized Error: {comparison['optimized_error']:.4f}")
print(f"  Improvement: {comparison['improvement']:.2f}%")

# ============================================================================
# BONUS: REINFORCEMENT LEARNING
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: REINFORCEMENT LEARNING")
print("=" * 80)

print("\n[B.1] Training RL Agent (100 episodes)...")
from src.reinforcement import compare_with_without_rl
rl_results = compare_with_without_rl(num_episodes=100)

# ============================================================================
# BONUS: GUI INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: GUI INTERFACE")
print("=" * 80)
print("""
[B.2] GUI Features:
  - Real-time temperature and humidity sliders
  - Plant type and growth stage selection
  - Live Mamdani vs Sugeno output comparison
  - Progress bars for heater/misting outputs
  - Automatic simulation with weather patterns
  - Status indicators (Optimal/Warning/Critical)

To launch GUI, run: python main.py --mode gui
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)
print("""
✅ PART 1: System Modeling
   - 3 plant species (Tomato, Lettuce, Orchid)
   - Climate requirements for 3 growth stages
   - Comparison table and fuzzy logic justification

✅ PART 2: Fuzzy System Design
   - 5 membership functions per input variable
   - Mamdani controller with 30 rules
   - Sugeno controller with 30 rules
   - Justified fuzzification/defuzzification methods

✅ PART 3: Programming Implementation
   - Full Python implementation (no external fuzzy libraries)
   - Custom membership function calculations
   - Complete inference engines for both controllers

✅ PART 4: Dynamic Adaptation
   - Automatic MF adjustment for plant type
   - Growth stage-based mode switching
   - Emergency detection and response

✅ PART 5: Performance Evaluation
   - 20+ random test simulations
   - Performance comparison table
   - Analysis of Mamdani vs Sugeno

✅ PART 6: Optimization
   - PSO implementation for MF parameter tuning
   - Before/after comparison results
   - Demonstrated improvement

✅ BONUS: GUI Interface
   - Interactive Tkinter dashboard
   - Real-time simulation capability

✅ BONUS: Reinforcement Learning
   - Q-learning for rule weight optimization
   - Demonstrated performance improvement

✅ FINAL REPORT: FINAL_REPORT.md
   - Comprehensive documentation of all parts
""")

print("=" * 80)
print("DEMONSTRATION COMPLETE!")
print("=" * 80)
