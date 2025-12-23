"""
Utility Functions Module
========================
Helper functions for the greenhouse fuzzy control system.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
import os


def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to: {filename}")


def load_results(filename: str) -> Dict:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def normalize(data: np.ndarray, min_val: float = None, 
              max_val: float = None) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    if max_val - min_val == 0:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)


def denormalize(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Denormalize data from [0, 1] range."""
    return data * (max_val - min_val) + min_val


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Calculate moving average."""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def calculate_settling_time(response: np.ndarray, setpoint: float, 
                           tolerance: float = 0.05) -> int:
    """
    Calculate settling time (time to reach and stay within tolerance of setpoint).
    
    Args:
        response: System response array
        setpoint: Target value
        tolerance: Acceptable deviation (fraction of setpoint)
    
    Returns:
        Settling time in samples, or -1 if not settled
    """
    threshold = setpoint * tolerance
    
    for i in range(len(response) - 1, -1, -1):
        if abs(response[i] - setpoint) > threshold:
            if i < len(response) - 1:
                return i + 1
            return -1
    
    return 0


def calculate_overshoot(response: np.ndarray, setpoint: float) -> float:
    """
    Calculate maximum overshoot percentage.
    
    Args:
        response: System response array
        setpoint: Target value
    
    Returns:
        Overshoot as percentage
    """
    if setpoint == 0:
        return 0.0
    
    max_val = np.max(response)
    if max_val > setpoint:
        return (max_val - setpoint) / setpoint * 100
    return 0.0


def generate_step_input(duration: int, step_time: int, 
                       initial: float, final: float) -> np.ndarray:
    """Generate step input signal."""
    signal = np.ones(duration) * initial
    signal[step_time:] = final
    return signal


def generate_ramp_input(duration: int, start_time: int, 
                       initial: float, slope: float) -> np.ndarray:
    """Generate ramp input signal."""
    signal = np.ones(duration) * initial
    for i in range(start_time, duration):
        signal[i] = initial + slope * (i - start_time)
    return signal


def print_table(headers: List[str], rows: List[List], 
                col_widths: List[int] = None):
    """Print formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                      for i, h in enumerate(headers)]
    
    # Header
    header_str = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(f"|{header_str}|")
    print("|" + "|".join("-" * w for w in col_widths) + "|")
    
    # Rows
    for row in rows:
        row_str = "|".join(str(r).center(w) for r, w in zip(row, col_widths))
        print(f"|{row_str}|")


class PerformanceLogger:
    """Logger for tracking controller performance over time."""
    
    def __init__(self):
        self.logs = []
    
    def log(self, timestamp: float, temp: float, humidity: float,
            heater: float, misting: float, controller: str):
        """Log a single data point."""
        self.logs.append({
            'timestamp': timestamp,
            'temperature': temp,
            'humidity': humidity,
            'heater': heater,
            'misting': misting,
            'controller': controller
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.logs:
            return {}
        
        temps = [l['temperature'] for l in self.logs]
        humidities = [l['humidity'] for l in self.logs]
        heaters = [l['heater'] for l in self.logs]
        mistings = [l['misting'] for l in self.logs]
        
        return {
            'num_samples': len(self.logs),
            'temp_mean': np.mean(temps),
            'temp_std': np.std(temps),
            'humidity_mean': np.mean(humidities),
            'humidity_std': np.std(humidities),
            'heater_mean': np.mean(heaters),
            'misting_mean': np.mean(mistings),
        }
    
    def save(self, filename: str):
        """Save logs to file."""
        save_results({'logs': self.logs, 'summary': self.get_summary()}, filename)
    
    def clear(self):
        """Clear all logs."""
        self.logs = []
