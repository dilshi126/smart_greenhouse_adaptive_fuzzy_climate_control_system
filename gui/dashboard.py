"""
Greenhouse Fuzzy Control System - GUI Dashboard
================================================
Interactive GUI for real-time simulation and control visualization.

Features:
- Real-time temperature and humidity control
- Plant and growth stage selection
- Live controller output comparison (Mamdani vs Sugeno)
- Simulation mode with weather patterns
- Performance metrics display
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    PlantDatabase, GrowthStage, MembershipFunctions,
    MamdaniController, SugenoController, AdaptiveFuzzySystem
)


class GreenhouseGUI:
    """Main GUI application for greenhouse fuzzy control system."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Greenhouse Adaptive Fuzzy Control System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Set window icon
        self._set_window_icon()
        
        # Initialize controllers
        self.adaptive_system = None
        self.current_plant = "Tomato"
        self.current_stage = GrowthStage.VEGETATIVE
        
        # Simulation state
        self.simulation_running = False
        self.sim_step = 0
        
        # History for graphs
        self.temp_history = []
        self.humidity_history = []
        self.heater_history = []
        self.misting_history = []
        self.max_history = 100
        
        # Create GUI components
        self._create_styles()
        self._create_menu()
        self._create_main_layout()
        self._initialize_system()
        
        # Start update loop
        self._update_display()
    
    def _set_window_icon(self):
        """Set the window icon."""
        try:
            # Get the directory where this script is located
            gui_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(gui_dir, "icon.ico")
            
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
            else:
                # Try PNG as fallback
                logo_path = os.path.join(gui_dir, "logo.png")
                if os.path.exists(logo_path):
                    from PIL import Image, ImageTk
                    img = Image.open(logo_path)
                    photo = ImageTk.PhotoImage(img)
                    self.root.iconphoto(True, photo)
        except Exception as e:
            print(f"Could not set icon: {e}")
    
    def _create_styles(self):
        """Create custom styles for widgets."""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))
        style.configure('Value.TLabel', font=('Helvetica', 12))
        style.configure('Good.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Danger.TLabel', foreground='red')
    
    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reset", command=self._reset_system)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start Auto Simulation", command=self._start_simulation)
        sim_menu.add_command(label="Stop Simulation", command=self._stop_simulation)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title with logo
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Try to load and display logo
        try:
            from PIL import Image, ImageTk
            gui_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(gui_dir, "logo.png")
            if os.path.exists(logo_path):
                img = Image.open(logo_path)
                img = img.resize((48, 48), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(title_frame, image=self.logo_photo)
                logo_label.pack(side="left", padx=(0, 10))
        except Exception as e:
            print(f"Could not load logo: {e}")
        
        title_label = ttk.Label(title_frame, text="Smart Greenhouse Fuzzy Control System",
                               style='Title.TLabel')
        title_label.pack(side="left")
        
        # Left panel - Inputs
        self._create_input_panel(main_frame)
        
        # Center panel - Outputs
        self._create_output_panel(main_frame)
        
        # Right panel - Status
        self._create_status_panel(main_frame)
        
        # Bottom panel - Controls
        self._create_control_panel(main_frame)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def _create_input_panel(self, parent):
        """Create input controls panel."""
        frame = ttk.LabelFrame(parent, text="📊 Input Controls", padding="10")
        frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Plant selection
        ttk.Label(frame, text="Plant Type:", style='Header.TLabel').grid(
            row=0, column=0, sticky="w", pady=5)
        self.plant_var = tk.StringVar(value="Tomato")
        plant_combo = ttk.Combobox(frame, textvariable=self.plant_var,
                                   values=["Tomato", "Lettuce", "Orchid"],
                                   state="readonly", width=15)
        plant_combo.grid(row=0, column=1, pady=5, padx=5)
        plant_combo.bind("<<ComboboxSelected>>", self._on_plant_change)

        # Growth stage selection
        ttk.Label(frame, text="Growth Stage:", style='Header.TLabel').grid(
            row=1, column=0, sticky="w", pady=5)
        self.stage_var = tk.StringVar(value="Vegetative")
        stage_combo = ttk.Combobox(frame, textvariable=self.stage_var,
                                   values=["Seedling", "Vegetative", "Flowering"],
                                   state="readonly", width=15)
        stage_combo.grid(row=1, column=1, pady=5, padx=5)
        stage_combo.bind("<<ComboboxSelected>>", self._on_stage_change)
        
        # Temperature slider
        ttk.Label(frame, text="Temperature (°C):", style='Header.TLabel').grid(
            row=2, column=0, sticky="w", pady=(15, 5))
        self.temp_var = tk.DoubleVar(value=25.0)
        self.temp_slider = ttk.Scale(frame, from_=0, to=45, variable=self.temp_var,
                                     orient="horizontal", length=200,
                                     command=self._on_input_change)
        self.temp_slider.grid(row=3, column=0, columnspan=2, pady=5)
        self.temp_label = ttk.Label(frame, text="25.0°C", style='Value.TLabel')
        self.temp_label.grid(row=4, column=0, columnspan=2)
        
        # Humidity slider
        ttk.Label(frame, text="Humidity (%):", style='Header.TLabel').grid(
            row=5, column=0, sticky="w", pady=(15, 5))
        self.humidity_var = tk.DoubleVar(value=65.0)
        self.humidity_slider = ttk.Scale(frame, from_=0, to=100, variable=self.humidity_var,
                                         orient="horizontal", length=200,
                                         command=self._on_input_change)
        self.humidity_slider.grid(row=6, column=0, columnspan=2, pady=5)
        self.humidity_label = ttk.Label(frame, text="65.0%", style='Value.TLabel')
        self.humidity_label.grid(row=7, column=0, columnspan=2)
        
        # Optimal values display
        ttk.Separator(frame, orient='horizontal').grid(
            row=8, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(frame, text="Optimal Values:", style='Header.TLabel').grid(
            row=9, column=0, sticky="w")
        self.optimal_label = ttk.Label(frame, text="Temp: 25°C, Humidity: 65%")
        self.optimal_label.grid(row=10, column=0, columnspan=2, sticky="w")

    def _create_output_panel(self, parent):
        """Create output display panel."""
        frame = ttk.LabelFrame(parent, text="🎛️ Controller Outputs", padding="10")
        frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # Mamdani outputs
        ttk.Label(frame, text="Mamdani Controller", style='Header.TLabel').grid(
            row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(frame, text="Heater/Cooling:").grid(row=1, column=0, sticky="w")
        self.mamdani_heater_var = tk.StringVar(value="50.0%")
        ttk.Label(frame, textvariable=self.mamdani_heater_var, style='Value.TLabel').grid(
            row=1, column=1, sticky="e")
        
        self.mamdani_heater_bar = ttk.Progressbar(frame, length=200, mode='determinate')
        self.mamdani_heater_bar.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Label(frame, text="Misting:").grid(row=3, column=0, sticky="w")
        self.mamdani_misting_var = tk.StringVar(value="25.0%")
        ttk.Label(frame, textvariable=self.mamdani_misting_var, style='Value.TLabel').grid(
            row=3, column=1, sticky="e")
        
        self.mamdani_misting_bar = ttk.Progressbar(frame, length=200, mode='determinate')
        self.mamdani_misting_bar.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Separator
        ttk.Separator(frame, orient='horizontal').grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=15)
        
        # Sugeno outputs
        ttk.Label(frame, text="Sugeno Controller", style='Header.TLabel').grid(
            row=6, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(frame, text="Heater/Cooling:").grid(row=7, column=0, sticky="w")
        self.sugeno_heater_var = tk.StringVar(value="50.0%")
        ttk.Label(frame, textvariable=self.sugeno_heater_var, style='Value.TLabel').grid(
            row=7, column=1, sticky="e")
        
        self.sugeno_heater_bar = ttk.Progressbar(frame, length=200, mode='determinate')
        self.sugeno_heater_bar.grid(row=8, column=0, columnspan=2, pady=5)
        
        ttk.Label(frame, text="Misting:").grid(row=9, column=0, sticky="w")
        self.sugeno_misting_var = tk.StringVar(value="25.0%")
        ttk.Label(frame, textvariable=self.sugeno_misting_var, style='Value.TLabel').grid(
            row=9, column=1, sticky="e")
        
        self.sugeno_misting_bar = ttk.Progressbar(frame, length=200, mode='determinate')
        self.sugeno_misting_bar.grid(row=10, column=0, columnspan=2, pady=5)

    def _create_status_panel(self, parent):
        """Create status and metrics panel."""
        frame = ttk.LabelFrame(parent, text="📈 System Status", padding="10")
        frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        
        # Adaptation mode
        ttk.Label(frame, text="Adaptation Mode:", style='Header.TLabel').grid(
            row=0, column=0, sticky="w", pady=5)
        self.mode_var = tk.StringVar(value="Normal")
        self.mode_label = ttk.Label(frame, textvariable=self.mode_var, style='Value.TLabel')
        self.mode_label.grid(row=0, column=1, sticky="e", pady=5)
        
        # Output scale
        ttk.Label(frame, text="Output Scale:", style='Header.TLabel').grid(
            row=1, column=0, sticky="w", pady=5)
        self.scale_var = tk.StringVar(value="1.0")
        ttk.Label(frame, textvariable=self.scale_var, style='Value.TLabel').grid(
            row=1, column=1, sticky="e", pady=5)
        
        # Temperature status
        ttk.Separator(frame, orient='horizontal').grid(
            row=2, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(frame, text="Temperature Status:", style='Header.TLabel').grid(
            row=3, column=0, sticky="w", pady=5)
        self.temp_status_var = tk.StringVar(value="Optimal")
        self.temp_status_label = ttk.Label(frame, textvariable=self.temp_status_var)
        self.temp_status_label.grid(row=3, column=1, sticky="e", pady=5)
        
        # Humidity status
        ttk.Label(frame, text="Humidity Status:", style='Header.TLabel').grid(
            row=4, column=0, sticky="w", pady=5)
        self.humidity_status_var = tk.StringVar(value="Optimal")
        self.humidity_status_label = ttk.Label(frame, textvariable=self.humidity_status_var)
        self.humidity_status_label.grid(row=4, column=1, sticky="e", pady=5)
        
        # Simulation info
        ttk.Separator(frame, orient='horizontal').grid(
            row=5, column=0, columnspan=2, sticky='ew', pady=10)
        ttk.Label(frame, text="Simulation:", style='Header.TLabel').grid(
            row=6, column=0, sticky="w", pady=5)
        self.sim_status_var = tk.StringVar(value="Stopped")
        ttk.Label(frame, textvariable=self.sim_status_var, style='Value.TLabel').grid(
            row=6, column=1, sticky="e", pady=5)
        
        ttk.Label(frame, text="Step:", style='Header.TLabel').grid(
            row=7, column=0, sticky="w", pady=5)
        self.sim_step_var = tk.StringVar(value="0")
        ttk.Label(frame, textvariable=self.sim_step_var, style='Value.TLabel').grid(
            row=7, column=1, sticky="e", pady=5)

    def _create_control_panel(self, parent):
        """Create bottom control panel."""
        frame = ttk.Frame(parent, padding="10")
        frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Simulation controls
        self.start_btn = ttk.Button(frame, text="▶ Start Simulation",
                                    command=self._start_simulation)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(frame, text="⏹ Stop Simulation",
                                   command=self._stop_simulation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        ttk.Button(frame, text="🔄 Reset", command=self._reset_system).grid(
            row=0, column=2, padx=5)
        
        # Weather pattern selection
        ttk.Label(frame, text="Weather Pattern:").grid(row=0, column=3, padx=(20, 5))
        self.weather_var = tk.StringVar(value="oscillating")
        weather_combo = ttk.Combobox(frame, textvariable=self.weather_var,
                                     values=["stable", "warming", "cooling", "heat_wave",
                                            "cold_snap", "oscillating", "humid_storm"],
                                     state="readonly", width=12)
        weather_combo.grid(row=0, column=4, padx=5)
        
        # Speed control
        ttk.Label(frame, text="Speed:").grid(row=0, column=5, padx=(20, 5))
        self.speed_var = tk.IntVar(value=500)
        speed_scale = ttk.Scale(frame, from_=100, to=1000, variable=self.speed_var,
                               orient="horizontal", length=100)
        speed_scale.grid(row=0, column=6, padx=5)
    
    def _initialize_system(self):
        """Initialize the adaptive fuzzy system."""
        self.adaptive_system = AdaptiveFuzzySystem(
            self.current_plant, self.current_stage, controller_type="both"
        )
        self._update_optimal_display()
    
    def _update_optimal_display(self):
        """Update the optimal values display."""
        if self.adaptive_system:
            setpoints = self.adaptive_system.get_setpoints()
            self.optimal_label.config(
                text=f"Temp: {setpoints['temperature']}°C, Humidity: {setpoints['humidity']}%"
            )

    def _on_plant_change(self, event=None):
        """Handle plant type change."""
        plant = self.plant_var.get()
        if plant != self.current_plant:
            self.current_plant = plant
            if self.adaptive_system:
                self.adaptive_system.change_plant(plant)
            self._update_optimal_display()
            self._compute_outputs()
    
    def _on_stage_change(self, event=None):
        """Handle growth stage change."""
        stage_map = {
            "Seedling": GrowthStage.SEEDLING,
            "Vegetative": GrowthStage.VEGETATIVE,
            "Flowering": GrowthStage.FLOWERING
        }
        stage = stage_map[self.stage_var.get()]
        if stage != self.current_stage:
            self.current_stage = stage
            if self.adaptive_system:
                self.adaptive_system.change_growth_stage(stage)
            self._update_optimal_display()
            self._compute_outputs()
    
    def _on_input_change(self, event=None):
        """Handle input slider changes."""
        temp = self.temp_var.get()
        humidity = self.humidity_var.get()
        self.temp_label.config(text=f"{temp:.1f}°C")
        self.humidity_label.config(text=f"{humidity:.1f}%")
        self._compute_outputs()
    
    def _compute_outputs(self):
        """Compute and display controller outputs."""
        if not self.adaptive_system:
            return
        
        temp = self.temp_var.get()
        humidity = self.humidity_var.get()
        
        # Get control outputs
        result = self.adaptive_system.control(temp, humidity)
        
        # Update Mamdani outputs
        m_heater = result['mamdani']['heater']
        m_misting = result['mamdani']['misting']
        self.mamdani_heater_var.set(f"{m_heater:.1f}%")
        self.mamdani_misting_var.set(f"{m_misting:.1f}%")
        self.mamdani_heater_bar['value'] = m_heater
        self.mamdani_misting_bar['value'] = m_misting
        
        # Update Sugeno outputs
        s_heater = result['sugeno']['heater']
        s_misting = result['sugeno']['misting']
        self.sugeno_heater_var.set(f"{s_heater:.1f}%")
        self.sugeno_misting_var.set(f"{s_misting:.1f}%")
        self.sugeno_heater_bar['value'] = s_heater
        self.sugeno_misting_bar['value'] = s_misting
        
        # Update status
        self.mode_var.set(result['state']['mode'].title())
        self.scale_var.set(f"{result['state']['scale']:.2f}")
        
        # Update temperature status
        optimal_temp = result['state']['optimal_temp']
        temp_diff = abs(temp - optimal_temp)
        if temp_diff < 3:
            self.temp_status_var.set("✓ Optimal")
            self.temp_status_label.config(foreground='green')
        elif temp_diff < 8:
            self.temp_status_var.set("⚠ Warning")
            self.temp_status_label.config(foreground='orange')
        else:
            self.temp_status_var.set("✗ Critical")
            self.temp_status_label.config(foreground='red')
        
        # Update humidity status
        optimal_humidity = result['state']['optimal_humidity']
        humidity_diff = abs(humidity - optimal_humidity)
        if humidity_diff < 10:
            self.humidity_status_var.set("✓ Optimal")
            self.humidity_status_label.config(foreground='green')
        elif humidity_diff < 20:
            self.humidity_status_var.set("⚠ Warning")
            self.humidity_status_label.config(foreground='orange')
        else:
            self.humidity_status_var.set("✗ Critical")
            self.humidity_status_label.config(foreground='red')

    def _start_simulation(self):
        """Start automatic simulation."""
        self.simulation_running = True
        self.sim_step = 0
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.sim_status_var.set("Running")
        self._run_simulation_step()
    
    def _stop_simulation(self):
        """Stop automatic simulation."""
        self.simulation_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.sim_status_var.set("Stopped")
    
    def _run_simulation_step(self):
        """Execute one simulation step."""
        if not self.simulation_running:
            return
        
        import numpy as np
        
        # Generate weather-based changes
        pattern = self.weather_var.get()
        t = self.sim_step
        
        # Get current values
        base_temp = 25
        base_humidity = 65
        
        if pattern == "stable":
            temp = base_temp + np.random.normal(0, 1)
            humidity = base_humidity + np.random.normal(0, 2)
        elif pattern == "warming":
            temp = base_temp + 0.1 * t + np.random.normal(0, 0.5)
            humidity = base_humidity - 0.05 * t + np.random.normal(0, 1)
        elif pattern == "cooling":
            temp = base_temp - 0.1 * t + np.random.normal(0, 0.5)
            humidity = base_humidity + 0.05 * t + np.random.normal(0, 1)
        elif pattern == "heat_wave":
            temp = base_temp + 10 * np.sin(t * 0.1) * (1 if t < 50 else -1)
            humidity = base_humidity - 5 * np.sin(t * 0.1)
        elif pattern == "cold_snap":
            temp = base_temp - 10 * np.sin(t * 0.1) * (1 if t < 50 else -1)
            humidity = base_humidity + 5 * np.sin(t * 0.1)
        elif pattern == "oscillating":
            temp = base_temp + 8 * np.sin(t * 0.15)
            humidity = base_humidity - 10 * np.sin(t * 0.15)
        elif pattern == "humid_storm":
            temp = base_temp - 3 + np.random.normal(0, 1)
            humidity = base_humidity + 20 * np.exp(-((t - 50)**2) / 500)
        else:
            temp = base_temp + np.random.normal(0, 2)
            humidity = base_humidity + np.random.normal(0, 3)
        
        # Clip to valid ranges
        temp = np.clip(temp, 0, 45)
        humidity = np.clip(humidity, 0, 100)
        
        # Update sliders
        self.temp_var.set(temp)
        self.humidity_var.set(humidity)
        self.temp_label.config(text=f"{temp:.1f}°C")
        self.humidity_label.config(text=f"{humidity:.1f}%")
        
        # Compute outputs
        self._compute_outputs()
        
        # Update step counter
        self.sim_step += 1
        self.sim_step_var.set(str(self.sim_step))
        
        # Schedule next step
        delay = 1100 - self.speed_var.get()  # Invert so higher = faster
        self.root.after(delay, self._run_simulation_step)
    
    def _update_display(self):
        """Periodic display update."""
        self._compute_outputs()
        self.root.after(100, self._update_display)
    
    def _reset_system(self):
        """Reset the system to defaults."""
        self._stop_simulation()
        self.plant_var.set("Tomato")
        self.stage_var.set("Vegetative")
        self.temp_var.set(25.0)
        self.humidity_var.set(65.0)
        self.current_plant = "Tomato"
        self.current_stage = GrowthStage.VEGETATIVE
        self._initialize_system()
        self.sim_step = 0
        self.sim_step_var.set("0")
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Smart Greenhouse Adaptive Fuzzy Control System\n\n"
            "Course: Fuzzy Logic and Control Systems\n\n"
            "Features:\n"
            "• Mamdani & Sugeno Controllers\n"
            "• Adaptive System\n"
            "• Real-time Simulation\n"
            "• Multiple Plant Species\n\n"
            "© 2024"
        )


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    app = GreenhouseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
