# Smart Greenhouse Adaptive Fuzzy Climate Control System

## Course: Fuzzy Logic and Control Systems

This project implements an adaptive fuzzy control system for greenhouse climate management,
supporting multiple plant species with varying temperature and humidity requirements across
different growth stages.

## Project Structure

```
greenhouse_fuzzy_control/
├── src/
│   ├── __init__.py                # Package exports
│   ├── membership_functions.py    # Fuzzy membership function definitions
│   ├── mamdani_controller.py      # Mamdani fuzzy controller
│   ├── sugeno_controller.py       # Takagi-Sugeno fuzzy controller
│   ├── adaptive_system.py         # Dynamic adaptation mechanism
│   ├── plant_database.py          # Plant species and growth stage data
│   ├── simulator.py               # Simulation engine
│   ├── optimizer.py               # PSO optimization for membership functions
│   ├── reinforcement.py           # Reinforcement learning for rule evolution
│   └── utils.py                   # Utility functions
├── gui/
│   ├── __init__.py
│   └── dashboard.py               # Tkinter GUI interface
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
├── FINAL_REPORT.md                # Comprehensive project report
└── README.md                      # This file
```

## Features

- **Three Plant Species**: Tomato, Lettuce, Orchid
- **Three Growth Stages**: Seedling, Vegetative, Flowering
- **Dual Controllers**: Mamdani and Takagi-Sugeno implementations
- **Adaptive System**: Automatic adjustment based on plant type and growth stage
- **PSO Optimization**: Membership function parameter optimization
- **Reinforcement Learning**: Q-learning for fuzzy rule evolution (Bonus)
- **GUI Interface**: Interactive dashboard for real-time simulation (Bonus)
- **Performance Evaluation**: Comprehensive comparison metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Demo
```bash
python main.py
```

### Available Modes

```bash
# Run demonstration
python main.py --mode demo

# Run simulation with specific plant and weather
python main.py --mode simulate --plant Lettuce --stage vegetative --weather heat_wave

# Run PSO optimization
python main.py --mode optimize --controller mamdani

# Compare controllers (20 random tests)
python main.py --mode compare --tests 20

# Generate visualization plots
python main.py --mode visualize

# Launch GUI interface (Bonus)
python main.py --mode gui

# Run reinforcement learning (Bonus)
python main.py --mode rl --episodes 500
```

### Command Line Options

| Option | Short | Values | Default | Description |
|--------|-------|--------|---------|-------------|
| --mode | -m | demo, simulate, optimize, compare, visualize, gui, rl | demo | Operation mode |
| --plant | -p | Tomato, Lettuce, Orchid | Tomato | Plant type |
| --stage | -s | seedling, vegetative, flowering | vegetative | Growth stage |
| --weather | -w | stable, warming, cooling, heat_wave, cold_snap, oscillating, humid_storm, dry_spell | oscillating | Weather pattern |
| --duration | -d | integer | 100 | Simulation duration |
| --controller | -c | mamdani, sugeno | mamdani | Controller type |
| --tests | -t | integer | 20 | Number of comparison tests |
| --episodes | -e | integer | 300 | RL training episodes |

## Controllers

### Mamdani Controller
- Uses min-max inference
- Centroid defuzzification
- 30 fuzzy rules
- Better for intuitive rule interpretation

### Takagi-Sugeno Controller
- Linear output functions
- Weighted average defuzzification
- 30 fuzzy rules
- More computationally efficient

## Performance Metrics

- **Average Response Time**: How quickly the controller responds to changes
- **Average Error**: Deviation from optimal setpoints
- **Energy Usage**: Total control effort
- **Smoothness Score**: Output variation (lower = smoother)

## Bonus Features

### GUI Interface
Interactive Tkinter dashboard with:
- Real-time temperature and humidity sliders
- Plant and growth stage selection
- Live controller output comparison
- Automatic simulation with weather patterns
- Performance status indicators

### Reinforcement Learning
Q-learning implementation for rule optimization:
- Automatic rule weight adjustment
- State-based learning (75 discrete states)
- 5 action types for rule modification
- Achieves 10-20% performance improvement

## Assignment Requirements Checklist

| Part | Requirement | Status |
|------|-------------|--------|
| 1 | 3+ plant species | ✅ |
| 1 | Climate requirements table | ✅ |
| 1 | Fuzzy vs PID justification | ✅ |
| 2 | 5 fuzzy sets per input | ✅ |
| 2 | Mamdani controller | ✅ |
| 2 | Sugeno controller | ✅ |
| 2 | 25+ rules per controller | ✅ (30) |
| 3 | Python implementation | ✅ |
| 4 | Adaptive mechanism | ✅ |
| 4 | Adaptation logic explanation | ✅ |
| 5 | 20+ random simulations | ✅ |
| 5 | Performance comparison table | ✅ |
| 6 | PSO optimization | ✅ |
| 6 | Before/after results | ✅ |
| 7 | Final report | ✅ |
| Bonus | GUI interface | ✅ |
| Bonus | Reinforcement learning | ✅ |

## Files Description

- **FINAL_REPORT.md**: Comprehensive report covering all assignment requirements
- **main.py**: Entry point with CLI interface
- **src/**: Core implementation modules
- **gui/**: GUI dashboard implementation

## Author

Fuzzy Logic and Control Systems Course Assignment
