"""
Smart Greenhouse Adaptive Fuzzy Climate Control System
======================================================
Main entry point for the greenhouse fuzzy control system.

This system implements adaptive fuzzy control for greenhouse climate management,
supporting multiple plant species with varying temperature and humidity requirements
across different growth stages.

Features:
- Mamdani and Takagi-Sugeno fuzzy controllers
- Adaptive system that adjusts to plant type and growth stage
- PSO optimization for membership function tuning
- Comprehensive simulation and performance evaluation

Usage:
    python main.py [--mode MODE] [--plant PLANT] [--stage STAGE]

Modes:
    demo        - Run demonstration of all features (default)
    simulate    - Run simulation with specified plant/stage
    optimize    - Run PSO optimization
    compare     - Compare Mamdani vs Sugeno controllers
    visualize   - Generate visualization plots
"""

import argparse
import sys
import numpy as np

from src import (
    PlantDatabase,
    GrowthStage,
    MembershipFunctions,
    MamdaniController,
    SugenoController,
    AdaptiveFuzzySystem,
    GreenhouseSimulator,
    PSOOptimizer,
    FuzzyRLAgent
)


def run_demo():
    """Run a comprehensive demonstration of the system."""
    print("\n" + "=" * 70)
    print("SMART GREENHOUSE ADAPTIVE FUZZY CLIMATE CONTROL SYSTEM")
    print("=" * 70)
    
    # 1. Show plant database
    print("\n[1] PLANT DATABASE")
    print("-" * 50)
    PlantDatabase.print_comparison_table()
    
    # 2. Demonstrate membership functions
    print("\n[2] MEMBERSHIP FUNCTIONS")
    print("-" * 50)
    mf = MembershipFunctions()
    
    test_inputs = [(25, 65, 50), (15, 40, 20), (35, 85, 80)]
    for temp, humidity, stage in test_inputs:
        print(f"\nFuzzification of: Temp={temp}°C, Humidity={humidity}%, Stage={stage}")
        fuzzy_values = mf.fuzzify_all(temp, humidity, stage)
        for var_name, memberships in fuzzy_values.items():
            active = {k: f"{v:.3f}" for k, v in memberships.items() if v > 0.01}
            if active:
                print(f"  {var_name}: {active}")
    
    # 3. Test both controllers
    print("\n[3] CONTROLLER COMPARISON")
    print("-" * 50)
    
    mamdani = MamdaniController()
    sugeno = SugenoController()
    
    test_cases = [
        (15, 40, 20, "Cold & Dry (Seedling)"),
        (25, 65, 50, "Optimal (Vegetative)"),
        (35, 30, 80, "Hot & Dry (Flowering)"),
        (10, 90, 50, "Very Cold & Humid"),
    ]
    
    print(f"\n{'Condition':<25} {'Mamdani Heater':<16} {'Mamdani Mist':<14} "
          f"{'Sugeno Heater':<15} {'Sugeno Mist'}")
    print("-" * 90)
    
    for temp, humidity, stage, desc in test_cases:
        m_result = mamdani.infer(temp, humidity, stage)
        s_result = sugeno.infer(temp, humidity, stage)
        print(f"{desc:<25} {m_result['heater']:>14.1f}% {m_result['misting']:>12.1f}% "
              f"{s_result['heater']:>13.1f}% {s_result['misting']:>10.1f}%")
    
    # 4. Demonstrate adaptive system
    print("\n[4] ADAPTIVE SYSTEM")
    print("-" * 50)
    
    adaptive = AdaptiveFuzzySystem("Tomato", GrowthStage.SEEDLING)
    print(f"\nInitial state: {adaptive.get_adaptation_info()}")
    
    # Simulate control
    result = adaptive.control(20, 70)
    print(f"\nControl at 20°C, 70% humidity:")
    print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, "
          f"Misting={result['mamdani']['misting']:.1f}%")
    print(f"  Sugeno:  Heater={result['sugeno']['heater']:.1f}%, "
          f"Misting={result['sugeno']['misting']:.1f}%")
    
    # Change plant
    adaptive.change_plant("Orchid")
    result = adaptive.control(20, 70)
    print(f"\nAfter changing to Orchid:")
    print(f"  Mamdani: Heater={result['mamdani']['heater']:.1f}%, "
          f"Misting={result['mamdani']['misting']:.1f}%")
    
    # 5. Run simulation
    print("\n[5] SIMULATION")
    print("-" * 50)
    
    simulator = GreenhouseSimulator()
    sim_result = simulator.run_simulation(
        "Tomato", GrowthStage.VEGETATIVE, "heat_wave", duration=50
    )
    
    print(f"\nSimulation: {sim_result.plant} ({sim_result.growth_stage})")
    print(f"Weather: {sim_result.weather_pattern}")
    print(f"\nMamdani Metrics:")
    for metric, value in sim_result.mamdani_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nSugeno Metrics:")
    for metric, value in sim_result.sugeno_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nFor more options, run: python main.py --help")


def run_simulation(plant: str, stage: GrowthStage, weather: str, duration: int):
    """Run a single simulation."""
    print(f"\nRunning simulation: {plant} ({stage.name}) - {weather}")
    print("-" * 50)
    
    simulator = GreenhouseSimulator()
    result = simulator.run_simulation(plant, stage, weather, duration)
    
    print(f"\nResults:")
    print(f"Temperature range: {result.temps.min():.1f}°C - {result.temps.max():.1f}°C")
    print(f"Humidity range: {result.humidities.min():.1f}% - {result.humidities.max():.1f}%")
    
    print(f"\nMamdani Performance:")
    for metric, value in result.mamdani_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nSugeno Performance:")
    for metric, value in result.sugeno_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Try to plot if matplotlib is available
    try:
        from src.simulator import plot_simulation_results
        plot_simulation_results(result, f"simulation_{plant}_{stage.name}.png")
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")


def run_optimization(plant: str, stage: GrowthStage, controller_type: str):
    """Run PSO optimization."""
    print(f"\nRunning PSO optimization for {controller_type} controller")
    print(f"Plant: {plant}, Stage: {stage.name}")
    print("-" * 50)
    
    optimizer = PSOOptimizer(
        controller_type=controller_type,
        num_particles=20,
        max_iterations=50
    )
    
    best_params, best_fitness = optimizer.optimize(plant, stage)
    
    # Compare before/after
    comparison = optimizer.compare_before_after(plant, stage, best_params)
    
    print(f"\nOptimization Results:")
    print(f"  Best Fitness: {best_fitness:.4f}")
    print(f"  Default Error: {comparison['default_error']:.4f}")
    print(f"  Optimized Error: {comparison['optimized_error']:.4f}")
    print(f"  Improvement: {comparison['improvement']:.2f}%")
    
    # Try to plot if matplotlib is available
    try:
        from src.optimizer import plot_optimization_history
        plot_optimization_history(optimizer, f"optimization_{controller_type}.png")
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")


def run_comparison(num_tests: int = 20):
    """Run comprehensive comparison between controllers."""
    print(f"\nRunning {num_tests} random simulations for comparison")
    print("-" * 50)
    
    simulator = GreenhouseSimulator()
    results = simulator.run_random_tests(num_tests)
    
    table = simulator.generate_comparison_table(results)
    print(table)


def run_visualization():
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return
    
    print("\nGenerating visualization plots...")
    print("-" * 50)
    
    # 1. Membership functions
    print("1. Plotting membership functions...")
    from src.membership_functions import plot_membership_functions
    mf = MembershipFunctions()
    plot_membership_functions(mf, "membership_functions.png")
    
    # 2. Control surfaces
    print("2. Plotting Mamdani control surface...")
    from src.mamdani_controller import plot_control_surface as plot_mamdani
    mamdani = MamdaniController()
    plot_mamdani(mamdani, save_path="mamdani_control_surface.png")
    
    print("3. Plotting Sugeno control surface...")
    from src.sugeno_controller import plot_control_surface as plot_sugeno
    sugeno = SugenoController()
    plot_sugeno(sugeno, save_path="sugeno_control_surface.png")
    
    # 3. Simulation
    print("4. Running and plotting simulation...")
    from src.simulator import plot_simulation_results
    simulator = GreenhouseSimulator()
    result = simulator.run_simulation("Tomato", GrowthStage.VEGETATIVE, "oscillating")
    plot_simulation_results(result, "simulation_results.png")
    
    print("\nVisualization complete! Generated files:")
    print("  - membership_functions.png")
    print("  - mamdani_control_surface.png")
    print("  - sugeno_control_surface.png")
    print("  - simulation_results.png")


def run_reinforcement_learning(num_episodes: int = 300):
    """Run reinforcement learning training."""
    print(f"\nRunning Reinforcement Learning ({num_episodes} episodes)")
    print("-" * 50)
    
    from src.reinforcement import compare_with_without_rl, FuzzyRLAgent
    
    # Compare with and without RL
    results = compare_with_without_rl(num_episodes=num_episodes)
    
    # Try to plot if matplotlib is available
    try:
        from src.reinforcement import plot_training_history
        agent = FuzzyRLAgent(controller_type="mamdani")
        agent.train(num_episodes=num_episodes, verbose=False)
        plot_training_history(agent, "rl_training_history.png")
    except ImportError:
        print("\nNote: Install matplotlib to generate plots")
    
    return results


def run_gui():
    """Launch the GUI interface."""
    print("\nLaunching GUI...")
    print("-" * 50)
    
    try:
        from gui.dashboard import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error launching GUI: {e}")
        print("Make sure tkinter is installed.")
    except Exception as e:
        print(f"GUI Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Greenhouse Adaptive Fuzzy Climate Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run demo
  python main.py --mode simulate --plant Lettuce --stage vegetative
  python main.py --mode optimize --controller mamdani
  python main.py --mode compare --tests 30
  python main.py --mode visualize
  python main.py --mode gui                # Launch GUI
  python main.py --mode rl --episodes 500  # Run reinforcement learning
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['demo', 'simulate', 'optimize', 'compare', 'visualize', 'gui', 'rl'],
        default='demo',
        help='Operation mode (default: demo)'
    )
    
    parser.add_argument(
        '--plant', '-p',
        choices=['Tomato', 'Lettuce', 'Orchid'],
        default='Tomato',
        help='Plant type (default: Tomato)'
    )
    
    parser.add_argument(
        '--stage', '-s',
        choices=['seedling', 'vegetative', 'flowering'],
        default='vegetative',
        help='Growth stage (default: vegetative)'
    )
    
    parser.add_argument(
        '--weather', '-w',
        choices=['stable', 'warming', 'cooling', 'heat_wave', 
                 'cold_snap', 'oscillating', 'humid_storm', 'dry_spell', 'random'],
        default='oscillating',
        help='Weather pattern for simulation (default: oscillating)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=100,
        help='Simulation duration in time steps (default: 100)'
    )
    
    parser.add_argument(
        '--controller', '-c',
        choices=['mamdani', 'sugeno'],
        default='mamdani',
        help='Controller type for optimization (default: mamdani)'
    )
    
    parser.add_argument(
        '--tests', '-t',
        type=int,
        default=20,
        help='Number of tests for comparison (default: 20)'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=300,
        help='Number of RL training episodes (default: 300)'
    )
    
    args = parser.parse_args()
    
    # Convert stage string to enum
    stage_map = {
        'seedling': GrowthStage.SEEDLING,
        'vegetative': GrowthStage.VEGETATIVE,
        'flowering': GrowthStage.FLOWERING
    }
    stage = stage_map[args.stage]
    
    # Run selected mode
    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'simulate':
        run_simulation(args.plant, stage, args.weather, args.duration)
    elif args.mode == 'optimize':
        run_optimization(args.plant, stage, args.controller)
    elif args.mode == 'compare':
        run_comparison(args.tests)
    elif args.mode == 'visualize':
        run_visualization()
    elif args.mode == 'gui':
        run_gui()
    elif args.mode == 'rl':
        run_reinforcement_learning(args.episodes)


if __name__ == "__main__":
    main()
