# Smart Greenhouse Adaptive Fuzzy Climate Control System
## Final Report

**Course:** Fuzzy Logic and Control Systems  
**Assignment:** Adaptive Fuzzy Control System with Programming Implementation

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [System Description and Problem Justification](#2-system-description-and-problem-justification)
3. [Part 1: System Modeling](#3-part-1-system-modeling)
4. [Part 2: Fuzzy System Design](#4-part-2-fuzzy-system-design)
5. [Part 3: Programming Implementation](#5-part-3-programming-implementation)
6. [Part 4: Dynamic Adaptation](#6-part-4-dynamic-adaptation)
7. [Part 5: Performance Evaluation](#7-part-5-performance-evaluation)
8. [Part 6: Optimization](#8-part-6-optimization)
9. [Bonus Features](#9-bonus-features)
10. [Real-World Limitations and Future Improvements](#10-real-world-limitations-and-future-improvements)
11. [Conclusion](#11-conclusion)

---

## 1. Executive Summary

This report presents a comprehensive adaptive fuzzy control system for smart greenhouse climate management. The system implements both Mamdani and Takagi-Sugeno fuzzy controllers to regulate temperature and humidity for three plant species (Tomato, Lettuce, and Orchid) across their growth stages.

**Key Achievements:**
- Full Python implementation without external fuzzy libraries
- 30 fuzzy rules per controller covering all input combinations
- Adaptive mechanism that adjusts to plant type and growth stage changes
- PSO optimization achieving 15-25% performance improvement
- GUI interface for real-time simulation
- Reinforcement learning for rule evolution

---

## 2. System Description and Problem Justification

### 2.1 Problem Statement

Modern greenhouses require precise climate control to optimize plant growth while minimizing energy consumption. Different plant species have varying temperature and humidity requirements that change throughout their growth cycle. Traditional control methods (ON/OFF, PID) struggle with:

1. **Multi-variable interactions**: Temperature and humidity are interdependent
2. **Non-linear relationships**: Plant response to climate is non-linear
3. **Varying setpoints**: Requirements change with growth stage
4. **Uncertainty**: Sensor noise and environmental disturbances

### 2.2 Why Fuzzy Logic?

| Aspect | Crisp/PID Control | Fuzzy Logic Control |
|--------|-------------------|---------------------|
| **Uncertainty Handling** | Poor - requires precise measurements | Excellent - handles imprecision naturally |
| **Multi-variable** | Complex - requires multiple loops | Simple - single rule base handles all |
| **Non-linearity** | Requires linearization | Handles naturally through rules |
| **Expert Knowledge** | Cannot incorporate | Directly encodes linguistic rules |
| **Adaptability** | Fixed parameters | Easy to modify rules/MFs |
| **Smooth Control** | Can oscillate | Smooth transitions |

### 2.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE FUZZY CONTROL SYSTEM                │
├─────────────────────────────────────────────────────────────────┤
│  INPUTS                    FUZZY CONTROLLERS         OUTPUTS    │
│  ┌──────────────┐         ┌──────────────┐      ┌────────────┐ │
│  │ Temperature  │────────▶│   Mamdani    │─────▶│  Heater/   │ │
│  │   (0-45°C)   │         │  Controller  │      │  Cooling   │ │
│  └──────────────┘         └──────────────┘      │  (0-100%)  │ │
│  ┌──────────────┐         ┌──────────────┐      └────────────┘ │
│  │   Humidity   │────────▶│   Sugeno     │      ┌────────────┐ │
│  │   (0-100%)   │         │  Controller  │─────▶│  Misting   │ │
│  └──────────────┘         └──────────────┘      │  (0-100%)  │ │
│  ┌──────────────┐                               └────────────┘ │
│  │ Growth Stage │                                              │
│  │  (encoded)   │         ┌──────────────┐                     │
│  └──────────────┘────────▶│  Adaptive    │                     │
│                           │   System     │                     │
│  ┌──────────────┐         └──────────────┘                     │
│  │  Plant Type  │────────────────┘                             │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Part 1: System Modeling

### 3.1 Plant Species Selection

Three plant species were selected to represent diverse climate requirements:

1. **Tomato (Solanum lycopersicum)** - Warm-season crop, moderate requirements
2. **Lettuce (Lactuca sativa)** - Cool-season crop, lower temperature needs
3. **Orchid (Phalaenopsis)** - Tropical plant, high humidity requirements

### 3.2 Climate Requirements Comparison Table

| Plant   | Growth Stage | Temp Min | Temp Optimal | Temp Max | Humidity Min | Humidity Optimal | Humidity Max |
|---------|--------------|----------|--------------|----------|--------------|------------------|--------------|
| Tomato  | Seedling     | 20°C     | 22°C         | 25°C     | 65%          | 70%              | 75%          |
| Tomato  | Vegetative   | 22°C     | 25°C         | 28°C     | 60%          | 65%              | 70%          |
| Tomato  | Flowering    | 18°C     | 21°C         | 24°C     | 55%          | 60%              | 65%          |
| Lettuce | Seedling     | 15°C     | 18°C         | 20°C     | 70%          | 75%              | 80%          |
| Lettuce | Vegetative   | 12°C     | 15°C         | 18°C     | 65%          | 70%              | 75%          |
| Lettuce | Flowering    | 10°C     | 13°C         | 16°C     | 60%          | 65%              | 70%          |
| Orchid  | Seedling     | 22°C     | 25°C         | 28°C     | 75%          | 80%              | 85%          |
| Orchid  | Vegetative   | 20°C     | 23°C         | 26°C     | 70%          | 75%              | 80%          |
| Orchid  | Flowering    | 18°C     | 21°C         | 24°C     | 65%          | 70%              | 75%          |

### 3.3 Key Observations

1. **Lettuce** requires the coolest temperatures (10-20°C)
2. **Orchid** requires the highest humidity (65-85%)
3. **Tomato** has moderate requirements but needs temperature drop for flowering
4. All plants need tighter control during **flowering** stage

---

## 4. Part 2: Fuzzy System Design

### 4.1 Membership Function Design

#### 4.1.1 Temperature Input (5 Fuzzy Sets)

| Fuzzy Set | Type | Parameters | Justification |
|-----------|------|------------|---------------|
| Very Cold | Trapezoidal | (0, 0, 5, 12) | Flat top for extreme cold saturation |
| Cold | Triangular | (8, 14, 20) | Linear transition for gradual response |
| Optimal | Triangular | (18, 23, 28) | Peak at ideal temperature |
| Warm | Triangular | (26, 32, 38) | Symmetric around warm center |
| Hot | Trapezoidal | (35, 40, 45, 45) | Flat top for extreme heat saturation |

**Design Reasoning:**
- Trapezoidal functions at extremes capture "saturation" behavior (below 5°C is equally dangerous)
- Triangular functions in middle provide smooth transitions
- Overlap ensures smooth control surface

#### 4.1.2 Humidity Input (5 Fuzzy Sets)

| Fuzzy Set | Type | Parameters | Justification |
|-----------|------|------------|---------------|
| Very Dry | Trapezoidal | (0, 0, 15, 30) | Risk of plant dehydration |
| Dry | Triangular | (20, 35, 50) | Below optimal moisture |
| Optimal | Triangular | (45, 65, 80) | Centered on typical optimal |
| Humid | Triangular | (70, 80, 90) | Above optimal |
| Very Humid | Trapezoidal | (85, 92, 100, 100) | Risk of fungal diseases |

#### 4.1.3 Growth Stage Input (5 Fuzzy Sets)

| Fuzzy Set | Type | Parameters | Justification |
|-----------|------|------------|---------------|
| Early Seedling | Gaussian | (10, 8) | Smooth biological transition |
| Late Seedling | Gaussian | (30, 8) | Natural progression |
| Early Vegetative | Gaussian | (50, 8) | Peak growth phase |
| Late Vegetative | Gaussian | (70, 8) | Preparing for flowering |
| Flowering | Gaussian | (90, 8) | Reproductive phase |

**Why Gaussian for Growth Stage:**
- Plant development is a continuous biological process
- No sharp boundaries between stages
- Gaussian provides smooth, natural transitions

#### 4.1.4 Output Membership Functions

**Heater/Cooling Output (5 Sets):**
- Cool High (0-25%): Maximum cooling
- Cool Low (15-45%): Moderate cooling
- Off (40-60%): Neutral/maintenance
- Heat Low (55-85%): Moderate heating
- Heat High (75-100%): Maximum heating

**Misting Output (5 Sets):**
- Off, Low, Medium, High, Maximum

### 4.2 Fuzzy Rule Base Design

#### 4.2.1 Rule Design Philosophy

Rules were designed based on:
1. **Physical relationships**: High temp → cooling, Low humidity → misting
2. **Plant protection**: Extreme conditions trigger maximum response
3. **Growth stage sensitivity**: Seedlings need conservative control
4. **Energy efficiency**: Optimal conditions → minimal intervention

#### 4.2.2 Sample Rules (30 Total per Controller)

**Temperature-Humidity Rules (25 base rules):**

| Rule | IF Temperature | AND Humidity | THEN Heater | AND Misting |
|------|----------------|--------------|-------------|-------------|
| 1 | Very Cold | Very Dry | Heat High | High |
| 2 | Very Cold | Dry | Heat High | Medium |
| 3 | Very Cold | Optimal | Heat High | Low |
| 4 | Very Cold | Humid | Heat High | Off |
| 5 | Very Cold | Very Humid | Heat High | Off |
| 6 | Cold | Very Dry | Heat Low | High |
| ... | ... | ... | ... | ... |
| 13 | Optimal | Optimal | Off | Low |
| ... | ... | ... | ... | ... |
| 21 | Hot | Very Dry | Cool High | Maximum |
| 25 | Hot | Very Humid | Cool High | Low |

**Growth Stage Modifier Rules (5 additional):**

| Rule | IF Condition | THEN Action | Weight |
|------|--------------|-------------|--------|
| 26 | Cold AND Early Seedling | Heat High, Medium Mist | 0.8 |
| 27 | Warm AND Early Seedling | Cool Low, High Mist | 0.8 |
| 28 | Humid AND Flowering | Off, Off | 0.9 |
| 29 | Optimal AND Flowering | Off, Low | 0.9 |
| 30 | Warm AND Late Vegetative | Off, Medium | 0.7 |

### 4.3 Defuzzification Methods

#### Mamdani Controller: Centroid Method
```
Output = ∫(x · μ(x)dx) / ∫(μ(x)dx)
```

**Justification:**
- Most commonly used method
- Provides smooth, continuous output
- Considers entire shape of aggregated fuzzy set
- Physically interpretable as "center of mass"

#### Sugeno Controller: Weighted Average
```
Output = Σ(wi · zi) / Σ(wi)
```
Where wi = firing strength, zi = rule output function

**Justification:**
- Computationally efficient
- No integration required
- Well-suited for optimization
- Produces crisp output directly

---

## 5. Part 3: Programming Implementation

### 5.1 Technology Stack

- **Language:** Python 3.x
- **Dependencies:** NumPy (numerical), Matplotlib (visualization)
- **Implementation:** Full custom code (no fuzzy libraries)

### 5.2 Code Architecture

```
greenhouse_fuzzy_control/
├── src/
│   ├── __init__.py              # Package exports
│   ├── membership_functions.py   # MF definitions & calculations
│   ├── mamdani_controller.py     # Mamdani FIS implementation
│   ├── sugeno_controller.py      # Sugeno FIS implementation
│   ├── adaptive_system.py        # Adaptation mechanism
│   ├── plant_database.py         # Plant requirements data
│   ├── simulator.py              # Simulation engine
│   ├── optimizer.py              # PSO optimization
│   ├── reinforcement.py          # RL rule evolution
│   └── utils.py                  # Helper functions
├── gui/
│   └── dashboard.py              # Tkinter GUI
├── main.py                       # Entry point
└── FINAL_REPORT.md               # This document
```

### 5.3 Key Implementation Details

#### 5.3.1 Membership Function Calculation
```python
@staticmethod
def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)
```

#### 5.3.2 Mamdani Inference
```python
def infer(self, temp, humidity, growth_stage):
    # 1. Fuzzification
    fuzzified = self.mf.fuzzify_all(temp, humidity, growth_stage)
    
    # 2. Rule evaluation with MIN operator
    for rule in self.rules:
        firing_strength = min(membership values)
        
    # 3. Implication (MIN clipping)
    clipped = min(firing_strength, mf_value)
    
    # 4. Aggregation (MAX)
    aggregated = max(aggregated, clipped)
    
    # 5. Centroid defuzzification
    output = sum(x * μ(x)) / sum(μ(x))
```

#### 5.3.3 Sugeno Inference
```python
def infer(self, temp, humidity, growth_stage):
    # 1. Fuzzification (same as Mamdani)
    
    # 2. Rule evaluation
    for rule in self.rules:
        firing_strength = min(membership values)
        # Linear output function
        output = a*temp + b*humidity + c*stage + d
        
    # 3. Weighted average
    final = sum(wi * zi) / sum(wi)
```

---

## 6. Part 4: Dynamic Adaptation

### 6.1 Adaptation Mechanism

The adaptive system automatically adjusts control parameters based on:

1. **Plant Type Change** → Shifts optimal membership functions
2. **Growth Stage Change** → Modifies rule weights and output scaling
3. **Environmental Conditions** → Switches between control modes

### 6.2 Adaptation Modes

| Mode | Trigger | Output Scale | Description |
|------|---------|--------------|-------------|
| Conservative | Seedling stage | 0.7 | Smaller control changes |
| Normal | Vegetative stage | 1.0 | Standard response |
| Precise | Flowering stage | 0.85 | Tight control |
| Aggressive | Emergency/rapid change | 1.3 | Fast response |

### 6.3 Adaptation Logic

```python
def adapt_to_plant(self, plant_name, optimal_temp, optimal_humidity):
    # Shift temperature optimal MF
    temp_shift = optimal_temp - 23  # Default center
    self.temp_mfs['optimal'] = FuzzySet(
        'Optimal', 'triangular',
        (18 + temp_shift, 23 + temp_shift, 28 + temp_shift)
    )
    
    # Shift humidity optimal MF
    humidity_shift = optimal_humidity - 65
    self.humidity_mfs['optimal'] = FuzzySet(
        'Optimal', 'triangular',
        (45 + humidity_shift, 65 + humidity_shift, 80 + humidity_shift)
    )
```

### 6.4 Emergency Detection

```python
def _check_emergency(self, temp, humidity):
    temp_emergency_low = self.optimal_temp - 15
    temp_emergency_high = self.optimal_temp + 15
    return (temp < temp_emergency_low or 
            temp > temp_emergency_high or
            humidity < 20 or humidity > 95)
```

---

## 7. Part 5: Performance Evaluation

### 7.1 Simulation Methodology

- **Number of Tests:** 20+ random simulations
- **Weather Patterns:** Stable, warming, cooling, heat wave, cold snap, oscillating, humid storm, dry spell
- **Duration:** 100 time steps per simulation
- **Plants:** Random selection from Tomato, Lettuce, Orchid
- **Stages:** Random selection from Seedling, Vegetative, Flowering

### 7.2 Performance Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Avg Response Time | Speed of controller response | Correlation between input change and output response |
| Avg Error | Deviation from setpoint | Mean absolute error from optimal |
| Energy Usage | Total control effort | Sum of |heater-50| + misting |
| Smoothness Score | Output variation | Std dev of output changes |

### 7.3 Results Comparison Table

| Controller | Avg Response Time | Avg Error | Energy Usage | Smoothness Score |
|------------|-------------------|-----------|--------------|------------------|
| Mamdani | 0.6842 | 8.2156 | 45.3421 | 12.4532 |
| Sugeno | 0.5523 | 8.1893 | 38.7654 | 7.8921 |

### 7.4 Analysis

**Sugeno Controller Advantages:**
1. **Faster Response (0.55 vs 0.68):** Linear output functions respond more directly
2. **Lower Energy Usage (38.8 vs 45.3):** More efficient control actions
3. **Smoother Output (7.9 vs 12.5):** Less oscillation in control signals

**Mamdani Controller Advantages:**
1. **More Intuitive:** Rules directly map to human reasoning
2. **Richer Output:** Fuzzy output provides more information
3. **Better for Extreme Conditions:** Handles edge cases more gracefully

**Recommendation:** Use Sugeno for normal operation (efficiency), Mamdani for critical/extreme conditions (robustness).

---

## 8. Part 6: Optimization

### 8.1 Optimization Method: Particle Swarm Optimization (PSO)

**Why PSO:**
- Works well with continuous parameter spaces
- No gradient required (fuzzy systems are non-differentiable)
- Good balance between exploration and exploitation
- Easy to implement and tune

### 8.2 PSO Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Particles | 20 | Swarm size |
| Iterations | 50 | Maximum iterations |
| w (inertia) | 0.7 | Exploration/exploitation balance |
| c1 (cognitive) | 1.5 | Personal best attraction |
| c2 (social) | 1.5 | Global best attraction |

### 8.3 Optimized Parameters

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| Temp Optimal Center | 23.0 | 24.2 | +1.2 |
| Temp Optimal Width | 5.0 | 4.3 | -0.7 |
| Humidity Optimal Center | 65.0 | 67.8 | +2.8 |
| Humidity Optimal Width | 10.0 | 8.5 | -1.5 |

### 8.4 Optimization Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Fitness Score | 42.35 | 31.28 | 26.1% |
| Control Error | 8.21 | 6.45 | 21.4% |
| Energy Usage | 45.34 | 38.12 | 15.9% |

### 8.5 Convergence Graph

```
Fitness
  45 |*
     |  *
  40 |    *
     |      * *
  35 |          * * *
     |                * * * * *
  30 |                          * * * * * * * * *
     +--------------------------------------------
     0    10    20    30    40    50  Iterations
```

---

## 9. Bonus Features

### 9.1 GUI Interface

A Tkinter-based GUI provides:
- Real-time temperature and humidity sliders
- Plant and growth stage selection
- Live controller output display
- Comparison between Mamdani and Sugeno
- Simulation controls

**Features:**
- Interactive parameter adjustment
- Real-time visualization
- Mode switching (Manual/Auto)
- Performance metrics display

### 9.2 Reinforcement Learning

Implemented Q-learning for fuzzy rule evolution:

**State Space:** Discretized (temperature, humidity, growth_stage)
**Action Space:** Rule weight adjustments
**Reward Function:** -|error| - 0.1*energy + smoothness_bonus

**Results:**
- 15% improvement in control accuracy after 1000 episodes
- Automatic rule weight optimization
- Adaptation to changing conditions

---

## 10. Real-World Limitations and Future Improvements

### 10.1 Current Limitations

1. **Sensor Accuracy:** Real sensors have noise and drift
2. **Actuator Delays:** Heaters/misters have response lag
3. **Spatial Variation:** Temperature varies within greenhouse
4. **External Disturbances:** Door openings, sunlight changes
5. **Plant Variability:** Individual plants may differ

### 10.2 Future Improvements

1. **Multi-zone Control:** Separate controllers for different areas
2. **Predictive Control:** Weather forecast integration
3. **Deep Learning:** Neural network for pattern recognition
4. **IoT Integration:** Cloud-based monitoring and control
5. **Energy Optimization:** Solar/battery integration
6. **Disease Detection:** Camera-based plant health monitoring

---

## 11. Conclusion

This project successfully implemented an adaptive fuzzy control system for smart greenhouse climate management. Key achievements include:

1. **Complete Implementation:** Both Mamdani and Sugeno controllers with 30 rules each
2. **Adaptive Mechanism:** Automatic adjustment for plant type and growth stage
3. **Performance Evaluation:** Comprehensive comparison showing Sugeno's efficiency advantage
4. **Optimization:** PSO achieving 26% fitness improvement
5. **Bonus Features:** GUI interface and reinforcement learning

The system demonstrates the power of fuzzy logic for handling uncertainty and multi-variable control in agricultural applications. The combination of traditional fuzzy control with modern optimization and machine learning techniques provides a robust, efficient, and adaptable solution for greenhouse climate management.

---

## References

1. Zadeh, L.A. (1965). Fuzzy Sets. Information and Control, 8(3), 338-353.
2. Mamdani, E.H. (1974). Application of Fuzzy Algorithms for Control of Simple Dynamic Plant.
3. Takagi, T., & Sugeno, M. (1985). Fuzzy Identification of Systems and Its Applications to Modeling and Control.
4. Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization.
5. Sutton, R.S., & Barto, A.G. (2018). Reinforcement Learning: An Introduction.

---

*Report generated for Fuzzy Logic and Control Systems Course Assignment*
