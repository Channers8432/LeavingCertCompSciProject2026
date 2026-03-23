import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import math

# Grid cell constants
EMPTY   = 0         # No tree in cell
TREE    = 1         # Healthy tree in cell 
BURNING = 2         # Currently on fire
BURNED  = 3         # Burnt out, can no longer burn

drought = {
    "temp": 38,
    "moisture": 0.08,
    "density": 0.80,
    "wind_speed": 1.5,
    "wind_dir": 0,
    "spread_type": "closed_source"
}

high_winds = {
    "temp": 42,
    "moisture": 0.05,
    "density": 0.55,
    "wind_speed": 9.0,
    "wind_dir": 45,
    "spread_type": "open_source"
}

# User can set wind direction using "compass"
# Allows user to click and drag to set wind direction in degrees (0-360)
class WindCompass(tk.Canvas):
    def __init__(self, master, size = 80, initial_dir = 0, callback = None, **kwargs):
        # Arguments:
        #   master:         Parent GUI
        #   size:           Diameter of the compass in pixels
        #   initial_dir:    Starting wind direction in degrees (0 = East, 90 = North)
        #   callback:       Function to call when wind direction changes

        super().__init__(master, width=size, height=size, bg="#323232", highlightthickness=0, **kwargs)
        self.size = size
        self.center = size // 2
        self.radius = (size // 2) - 5
        self.callback = callback
        self.angle = initial_dir
        
        # Bind mouse events for direction selection
        self.bind("<B1-Motion>", self._on_click)    # Drag to update
        self.bind("<Button-1>", self._on_click)     # Click to update
        self.draw()

    # Set wind direction angle
    def set_angle(self, angle):
        self.angle = angle
        self.draw()

    # Handle muse click/drag to update wind direction
    # Uses arctan to calcualte angle from centre
    def _on_click(self, event):
        dx = event.x - self.center
        dy = event.y - self.center
        rad = math.atan2(-dy, dx)       # dy negative because canvas y increases downward
        deg = math.degrees(rad) % 360   # Convert from radians to degrees
        self.angle = deg
        self.draw()
        if self.callback:
            self.callback(self.angle)   # Notify GUI of change

    # Draw the arrow inside the compass
    def draw(self):
        self.delete("all")                                                                                      # CLear previous drawing
        self.create_oval(5, 5, self.size-5, self.size-5, fill="#c8c8c8", outline="#999999", width=2)        # Draw background
        rad = math.radians(self.angle)                                                                          # Calculate arrow endpoint based on angle
        tx  = self.center + self.radius * math.cos(rad)
        ty  = self.center - self.radius * math.sin(rad)
        self.create_line(self.center, self.center, tx, ty, fill="#323232", width=3, arrow=tk.LAST)            # Draw arrow from centre to endpoint

# Agent based model class implementing wildfire spread simulation
# Uses cellular autmation and probailitic spread
class AgentBasedModel:
    def __init__(self, width=60, height=40, tree_density=0.65, temp_c=30, soil_moisture=0.3, wind_speed=0, wind_dir=0, seed=None, spread_type="closed_source"):
        # Arguments:
        #   width, height:  Grid dimensions in cells
        #   tree_density:   Probability of cell containing a tree (0.0 = low, 1.0 = high)
        #   temp_c:         Ambient temperature in degrees Celsius
        #   soil_moisture:  Soil moisture content (0.0 = dry, 1.0 = saturated)
        #   wind_speed:     Wind speed in km/h
        #   wind_dir:       Wind direction in degrees (0 = East, 90 = North)
        #   seed:           Random number generator
        #   spread_type:    The probability function to use

        self.spread_type = spread_type

        self.w                  = width
        self.h                  = height

        # User inputs 
        self.temp_c             = temp_c
        self.soil_moisture      = soil_moisture
        self.tree_density       = tree_density
        self.wind_speed         = wind_speed
        self.wind_dir           = wind_dir

        self.time_of_day        = 0
        self.day_count          = 1

        self.rng                = np.random.default_rng(seed)

        # Generate grid, randomly place trees based on tree_density
        self.grid               = np.where(self.rng.random((height, width))< tree_density, TREE, EMPTY)   
        self.initial_tree_count = int(np.sum(self.grid == TREE))

        self.burn_time          = 3
        self.burn_counter       = np.zeros((height, width))

        # Save data for plotting line graph
        self.history            = {"burning": [], "burned": [], "trees": []}

        # List of [y, x, lifetime]
        self.embers             = []
        
    # Calculate temperature based on time of day using sin function
    # Peak = 1800hrs, minimum = 0600hrs
    # Returns temperature with +/- 5C variation
    def diurnal_temp(self):
        return self.temp_c + 5 * np.sin(2 * np.pi * (self.time_of_day - 6)/ 24)

    # Calculate base fire spread probability based on user inputs
    # Returns probility (0.0 = low, 1.0 = high) for fire spreading to adjecent cell
    def get_p_base(self):
        if self.spread_type == "open_source":
            return self.spread_open_source()
        return self.spread_close_source()

    # Function using data obtained from potted plate
    def spread_close_source(self):
        SLOPE     = -0.2795
        INTERCEPT = 21.3733
        MEAN_T    = 9.92
        STD_T     = 2.05

        current_temp = self.diurnal_temp()

        predicted_moisture = np.clip((SLOPE * current_temp + INTERCEPT) / 100.0, 0.0, 1.0)
        effective_moisture = 0.5 * predicted_moisture + 0.5 * self.soil_moisture

        dryness_effect = (1.0 - effective_moisture) ** 2
        temp_effect    = 1.0 / (1.0 + np.exp(-(current_temp - MEAN_T) / STD_T))

        return 0.30 * temp_effect * dryness_effect

    # Function using data obtained from Kaggle
    def spread_open_source(self):
        SLOPE      = 1.6530
        INTERCEPT  = -0.8037
        SIG_CENTRE = 6.35
        SIG_SCALE  = 3.71

        current_temp = self.diurnal_temp()
        fuel_dryness = np.clip((1.0 - self.soil_moisture) * 95 + (current_temp * 0.3), 0, 101)

        f_wind = np.exp(0.05039 * self.wind_speed)
        f_fuel = 0.9110 * np.exp(0.0337 * fuel_dryness) * 0.1
        isi_value = f_wind * f_fuel

        fwi_score = SLOPE * isi_value + INTERCEPT

        noise = self.rng.uniform(0.8, 1.2)
        return float(np.clip((1 / (1 + np.exp(-(fwi_score - SIG_CENTRE) / SIG_SCALE))) * noise, 0.01, 0.9))
    
    # Execute a simulation time step (0.5 hours)
    # Updates fire spread, ember movement, and cell states
    def step(self):

        # Increase time
        self.time_of_day += 0.5
        if self.time_of_day >= 24:
            self.time_of_day = 0
            self.day_count += 1
            
        p_base =    self.get_p_base()                                   # Spread probaility
        new_grid =  self.grid.copy()                                    # Create copy to avoid conflicts
        
        # Convert wind direction to vector 
        wind_rad = np.radians(self.wind_dir)
        wind_vec = np.array([np.cos(wind_rad), -np.sin(wind_rad)])

        # Update ember positions
        # Embers drift with wind + random "turbulance"
        new_embers = []
        for ey, ex, life in self.embers:
            if life > 0:
                # Small random drifts to model turbulence
                drift_x = self.rng.uniform(-0.1, 0.1)
                drift_y = self.rng.uniform(-0.1, 0.1)

                # Move ember in wind direction + drift
                nx = ex + (wind_vec[0] * (self.wind_speed * 0.4)) + drift_x
                ny = ey + (wind_vec[1] * (self.wind_speed * 0.4)) + drift_y
                
                new_embers.append([ny, nx, life - 1])                              # Decrease lifetime
        self.embers = new_embers

        # Main fire spread loop, iterate through all grid cells
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y, x] == BURNING:
                    # High winds (> 7 km/h) will randomly generate embers to create spot fires
                    if self.wind_speed > 7.0 and self.rng.random() < 0.05:
                        self.embers.append([y, x, self.rng.integers(3, 8)])

                    # Check all 8 adjecent cells for potential spread
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0: continue                        # Skip self
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.h and 0 <= nx < self.w:
                                if self.grid[ny, nx] == TREE:
                                    # Calculate wind alignemnt for this direction
                                    dir_vec = np.array([dx, dy])
                                    norm = np.linalg.norm(dir_vec)
                                    if norm > 0: dir_vec = dir_vec / norm
                                    alignment = np.dot(dir_vec, wind_vec)           # Dot product gives alignment

                                    # Wind increases spread in aligned direction, decreases against wind
                                    wind_multiplier = 1 + (self.wind_speed * 0.2 * alignment)
                                    p = p_base * max(0.1, wind_multiplier)
                                    if self.rng.random()< p:
                                        new_grid[ny, nx] = BURNING
                    
                    # Stronger winds = longer distance spottting
                    if self.wind_speed > 5.0 and self.rng.random()< (self.wind_speed * 0.02):
                        dist = self.rng.integers(3, 9)                              # Random spotting distance
                        tx = int(x + wind_vec[0] * dist)
                        ty = int(y + wind_vec[1] * dist)
                        if 0 <= tx < self.w and 0 <= ty < self.h:
                            if self.grid[ty, tx] == TREE:
                                # Direr conditions increases spotting ignition
                                ember_ignite_p = (1 - self.soil_moisture)* 0.2
                                if self.rng.random()< ember_ignite_p:
                                    new_grid[ty, tx] = BURNING

                    # Update burn counter and turn to BURNED state 
                    self.burn_counter[y, x] += 1
                    if self.burn_counter[y, x] >= self.burn_time:
                        new_grid[y, x] = BURNED

        # Apply all state chnages simultaenously
        self.grid = new_grid

        # Record current state counts for line graph plotting
        for key, val in [("burning", BURNING), ("burned", BURNED), ("trees", TREE)]:
            self.history[key].append(np.sum(self.grid == val))

# Main GUI class
# Contrls UI and simulation controls/ user inputs
class WildfireDashboard:
    def __init__(self, root):
        # Create left control panel (sidebar) and right graph view panel
        self.root = root
        self.root.title("[REDACTED] - Disaster Risk Modelling Dashboard")

        # Scale window to 95% of screen size so it fits on any monitor
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w    = int(screen_w * 0.95)
        win_h    = int(screen_h * 0.92)
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.configure(bg="#323232")

        # Define colour map (white (EMPTY), green (TREE), orange (BURNING), black (BURNED))
        self.fire_cmap = ListedColormap(['#ffffff', '#27ae60', '#eaa400', '#000000']) 

        # Sidebar (left)
        sidebar_w = max(300, min(350, int(screen_w * 0.22)))
        self.left_panel = tk.Frame(root, bg="#323232", width=sidebar_w, padx=10, pady=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)                                                   # Prevent automatic sizing

        # Tabs for scenario selection
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background="#323232", borderwidth=0)
        style.configure("TNotebook.Tab", background="#444444", foreground="white", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#4a69bd")])

        # Create UI for scenarios
        self.notebook = ttk.Notebook(self.left_panel)
        self.notebook.pack(fill=tk.X, pady=(0, 20))

        self.tab_custom = tk.Frame(self.notebook, bg="#323232")
        self.tab_one = tk.Frame(self.notebook, bg="#323232")
        self.tab_two = tk.Frame(self.notebook, bg="#323232")

        self.notebook.add(self.tab_custom, text="CUSTOM")
        self.notebook.add(self.tab_one, text="DROUGHT")
        self.notebook.add(self.tab_two, text="HIGH WINDS")

        # Create dictionary to store slider references for each tab
        self.vars = {}
        self.setup_tab(self.tab_custom, "CUSTOM")
        self.setup_tab(self.tab_one, "ONE")
        self.setup_tab(self.tab_two, "TWO")

        # Control buttons (line 184 → 205)

        # INITIALISE:        Creates new simulation enviroemnt using user inputs
        self.btn_init = tk.Button(self.left_panel, text="INITIALISE", command=self.initialise_engine, bg="#4a69bd", fg="#ffffff", font=("Arial", 11, "bold"), height=1, bd=0)
        self.btn_init.pack(fill=tk.X, pady=5)

        # RANDOM IGNITION:   Starts fire at random tree cell
        self.btn_random = tk.Button(self.left_panel, text="RANDOM IGNITION", command=self.ignite_random, bg="#444444", fg="#ffffff", font=("Arial", 10, "bold"), height=1, state=tk.DISABLED, bd=0)
        self.btn_random.pack(fill=tk.X, pady=5)
        
        # Playback controls frame
        playback_frame = tk.Frame(self.left_panel, bg="#323232")
        playback_frame.pack(fill=tk.X, pady=10)

        # PAUSE/RESUME:     Toggles simulation
        self.btn_pause = tk.Button(playback_frame, text="PAUSE", command=self.toggle_pause, bg="#444444", fg="#ffffff", font=("Arial", 10, "bold"), height=1, state=tk.DISABLED, bd=0)
        self.btn_pause.grid(row=0, column=0, columnspan=2, sticky="ew", pady=2)

        # SLOWER:           Decreases simulation speed
        self.btn_slower = tk.Button(playback_frame, text="SLOWER", command=self.slower, bg="#4a69bd", fg="#ffffff", height=1, bd=0)
        self.btn_slower.grid(row=1, column=0, sticky="ew", padx=2, pady=2)

        # FATSER:           Increases simulation speed
        self.btn_faster = tk.Button(playback_frame, text="FASTER", command=self.faster, bg="#4a69bd", fg="#ffffff", height=1, bd=0)
        self.btn_faster.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        playback_frame.columnconfigure((0,1), weight=1)

        # STOP/RESET:       Clears grid and all data
        self.btn_stop = tk.Button(self.left_panel, text="STOP / RESET", command=self.reset_sim, bg="#c71e1e", fg="#ffffff", font=("Arial", 10, "bold"), height=1, bd=0)
        self.btn_stop.pack(fill=tk.X, pady=5)

        # Live Analytics
        # Shows current paramaters and counts
        self.param_frame = tk.LabelFrame(self.left_panel, text="Live Analytics", fg="#bdc3c7", bg="#323232", font=("Arial", 10, "italic"), padx=10, pady=10)
        self.param_frame.pack(fill=tk.X, pady=20)
        self.sim_text = tk.StringVar(value="Waiting for input...")
        tk.Label(self.param_frame, textvariable=self.sim_text, justify=tk.LEFT, fg="#ecf0f1", bg="#323232", font=("Courier", 9)).pack(anchor="w")

        # Risk level indicator
        # Changes colour and text based on amount of BURNED cells
        self.risk_label = tk.Label(self.left_panel, text="RISK LEVEL: STABLE", bg="#27ae60", fg="white", font=("Arial", 10, "bold"), pady=15)
        self.risk_label.pack(side=tk.BOTTOM, fill=tk.X)
        self.current_risk_state = "STABLE"

        # Area containing grid and line graph
        self.right_panel = tk.Frame(root, bg="white")
        self.right_panel.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Top banner showing day, time and simulation status
        banner_container = tk.Frame(self.right_panel, bg="#ecf0f1")
        banner_container.pack(fill=tk.X)
        
        self.status_banner = tk.Label(banner_container, text="STEP 1: INITIALISE ENVIRONMENT", font=("Arial", 12, "bold"), bg="#ecf0f1", fg="#2c3e50", pady=10)
        self.status_banner.pack(side=tk.LEFT, padx=10)
        
        self.time_label = tk.Label(banner_container, text="Day 0: 00:00", font=("Courier", 12, "bold"), bg="#ecf0f1", fg="#2c3e50")
        self.time_label.pack(side=tk.RIGHT, padx=20)

        # Create two matplotlib graphs:
        #   Top:    Grid/colourmap showing live cell states
        #   Bottom: Line graph of cell count v. time of cells 
        fig_w = (win_w - sidebar_w - 20) / 100   # Convert sidebar-adjusted pixel width to inches
        fig_h = (win_h - 60)  / 100               # Convert pixel height (minus banner) to inches
        self.fig, (self.ax_grid, self.ax_line) = plt.subplots(2, 1, figsize=(fig_w, fig_h), gridspec_kw={'height_ratios': [2, 1]})
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)            # Click on a tree cell to ignite
        
        # Simulation state variables
        self.model, self.ani = None, None
        self.is_initialised = self.is_running = self.is_paused = False
        self.interval = 100         # Animation frame intervals (milliseconds)
        self.draw_blank()           # Display empty grid on startup

    # Button colours 
    def set_spread_func(self, mode, choice):
        # Update stored variable
        self.vars[mode]["Spread Function"].set(choice)
        
        # Update button colors to show selection
        cs_btn = self.vars[mode]["btn_cs"]
        os_btn = self.vars[mode]["btn_os"]
        
        if choice == "closed_source":
            cs_btn.config(bg="#4a69bd")
            os_btn.config(bg="#444444")
        else:
            cs_btn.config(bg="#444444")
            os_btn.config(bg="#4a69bd")

    # Draw sliders 
    def setup_tab(self, frame, mode):
        # Arguments:
        #   frame:  Tab frame
        #   mode:   The current selected tab (CUSTOM, ONE, or TWO)

        self.vars[mode] = {}

        tk.Label(frame, text="Spread Algorithm", fg="white", bg="#323232", pady=2).pack(anchor="w")
        func_frame = tk.Frame(frame, bg="#323232")
        func_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Store choice
        func_var = tk.StringVar(value="closed_source")
        self.vars[mode]["Spread Function"] = func_var

        # Closed source button
        btn_cs = tk.Button(func_frame, text="CLOSED SOURCE", command=lambda: self.set_spread_func(mode, "closed_source"), bg="#4a69bd", fg="white", font=("Arial", 9, "bold"), bd=0, height=1)
        btn_cs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.vars[mode]["btn_cs"] = btn_cs

        # Open source button
        btn_os = tk.Button(func_frame, text="OPEN SOURCE", command=lambda: self.set_spread_func(mode, "open_source"), bg="#444444", fg="white", font=("Arial", 9, "bold"), bd=0, height=1)
        btn_os.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        self.vars[mode]["btn_os"] = btn_os

        # Disable buttons if not in CUSTOM mode
        if mode != "CUSTOM":
            btn_cs.config(state=tk.DISABLED)
            btn_os.config(state=tk.DISABLED)
        
        # Define sliders (label, min, max, default_value)
        params = [("Tree Density", 0.1, 1.0, 0.65), ("Temperature (°C)", 10, 50, 35), ("Soil Moisture", 0.0, 1.0, 0.2)]
        for label, start, end, d_val in params:
            tk.Label(frame, text=label, fg="white", bg="#323232", pady=2).pack(anchor="w")
            target_val = d_val

            # Preset "What if" scenario tab setup
            if mode == "ONE":
                if "Density" in label: target_val = drought["density"]
                elif "Temp" in label: target_val = drought["temp"]
                elif "Moisture" in label: target_val = drought["moisture"]
            elif mode == "TWO":
                if "Density" in label: target_val = high_winds["density"]
                elif "Temp" in label: target_val = high_winds["temp"]
                elif "Moisture" in label: target_val = high_winds["moisture"]

            s = tk.Scale(frame, from_=start, to=end, resolution=0.01 if start < 2 else 1, orient="horizontal", bg="#323232", fg="white", highlightthickness=0)
            s.set(target_val)

            # Disbale sliders if tab is not CUSTOM
            if mode != "CUSTOM": s.config(state=tk.DISABLED)
            s.pack(fill=tk.X, pady=(0, 5))
            self.vars[mode][label] = s

        # Wind controls: speed slider + direction compass
        tk.Label(frame, text="Wind", fg="white", bg="#323232", pady=2).pack(anchor="w")
        wind_frame = tk.Frame(frame, bg="#323232")
        wind_frame.pack(fill=tk.X, pady=(0, 10))

        wind_default = 2.0
        wind_dir_default = 0
        if mode == "ONE":
            wind_default     = drought["wind_speed"]
            wind_dir_default = drought["wind_dir"]
            func_var.set(drought["spread_type"])
        elif mode == "TWO":
            wind_default     = high_winds["wind_speed"]
            wind_dir_default = high_winds["wind_dir"]
            func_var.set(high_winds["spread_type"])

        # Sync button highlight colours to whichever spread type was set above
        if func_var.get() == "open_source":
            btn_cs.config(bg="#444444")
            btn_os.config(bg="#4a69bd")

        speed_slider = tk.Scale(wind_frame, from_=0, to=10, resolution=0.1, orient="horizontal", bg="#323232", fg="white", highlightthickness=0)
        speed_slider.set(wind_default)
        if mode != "CUSTOM": speed_slider.config(state=tk.DISABLED)
        speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.vars[mode]["Wind Speed"] = speed_slider
        dir_var = tk.DoubleVar(value=wind_dir_default)
        self.vars[mode]["Wind Direction"] = dir_var 
        compass = WindCompass(wind_frame, size=60, initial_dir=wind_dir_default, callback=lambda a: dir_var.set(a))
        
        # Disable compass interation if mode is not CUSTOM
        if mode != "CUSTOM": 
            compass.unbind("<B1-Motion>")
            compass.unbind("<Button-1>")
        compass.pack(side=tk.RIGHT, padx=10)
        self.vars[mode]["compass_widget"] = compass

    # Check which tab is currently selected
    def get_current_mode(self):
        idx = self.notebook.index(self.notebook.select())
        return ["CUSTOM", "ONE", "TWO"][idx]

    # Helper function to style buttons and enable/disable
    def update_button_ui(self, button, active):
        button.config(state=tk.NORMAL if active else tk.DISABLED, bg="#4a69bd" if active else "#444444")

    # Draw empty grid at startup/reset
    def draw_blank(self):
        self.ax_grid.clear()
        self.ax_grid.set_xlim(-0.5, 59.5)
        self.ax_grid.set_ylim(39.5, -0.5)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.set_aspect('equal')
        self.canvas.draw()

    # Create new simulation enviroment using user inputs
    # Resets all states and prepares enviroment for ignition
    def initialise_engine(self):
        mode = self.get_current_mode()
        self.model = AgentBasedModel(
            tree_density=self.vars[mode]["Tree Density"].get(), 
            temp_c=self.vars[mode]["Temperature (°C)"].get(), 
            soil_moisture=self.vars[mode]["Soil Moisture"].get(),
            wind_speed=self.vars[mode]["Wind Speed"].get(),
            wind_dir=self.vars[mode]["Wind Direction"].get(),
            spread_type=self.vars[mode]["Spread Function"].get()
        )
        
        self.is_initialised, self.is_running, self.is_paused = True, False, False
        if self.ani:
            self.ani.event_source.stop()                                               # Pause running animations
            self.ani = None
        self.status_banner.config(text="STEP 2: CLICK A GRID CELL OR PRESS RANDOM TO IGNITE")
        self.time_label.config(text="Day 1: 00:00")
        self.update_button_ui(self.btn_random, True)                                            # Enables RANDOM IGNITION buttom
        self.update_button_ui(self.btn_init, False)                                             # Disable re-initialisation
        self.update_button_ui(self.btn_pause, False)
        self.risk_label.config(text="RISK LEVEL: STABLE", bg="#27ae60")
        self.current_risk_state = "STABLE"
        self.refresh_display()

    # Handle mouse click on grid
    # If simulation is initialised and not running, ignite fire at mouse click position
    # Only ignites if clicked cell contains a tree
    def on_click(self, event):
        # Convert click coordinates to grid coordidinates
        if not self.is_initialised or self.is_running or event.inaxes != self.ax_grid: return
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.model.w and 0 <= iy < self.model.h:
            if self.model.grid[iy, ix] == TREE:
                self.ignite(ix, iy)

    # Select random tree cell and ignite
    def ignite_random(self):
        if not self.is_initialised or self.is_running: return
        coords = np.argwhere(self.model.grid == TREE)
        if len(coords)> 0:
            choice = coords[np.random.choice(len(coords))]
            self.ignite(choice[1], choice[0])

    # Start fire and selected grid cell and begin animation
    def ignite(self, x, y):
        # Arguments:
        #   x, y:   Grid coordinates for ignition point
        self.model.grid[y, x] = BURNING
        self.is_running = True
        self.update_button_ui(self.btn_random, False)               # Disable RANDOM IGNITION button
        self.update_button_ui(self.btn_pause, True)                 # Enable PAUSE/RESUME button
        self.status_banner.config(text="SIMULATION RUNNING")
        # Create matplotlib animation that repeatadley calls update()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.interval, cache_frame_data=False)
        self.canvas.draw()

    # Pause / resume animation
    def toggle_pause(self):
        if not self.is_running: return
        if self.is_paused:
            self.ani.event_source.start()
            self.is_paused = False
            self.btn_pause.config(text="PAUSE")
            self.status_banner.config(text="SIMULATION RUNNING")
        else:
            self.ani.event_source.stop()
            self.is_paused = True
            self.btn_pause.config(text="RESUME")
            self.status_banner.config(text="PAUSED")

    # Decrease animation interval to speed up playback (min 10ms)
    def faster(self): 
        self.interval = max(10, self.interval - 20)
        self.update_speed()

    # Oncrease animation interva; to slow down playback (max 500ms)
    def slower(self): 
        self.interval = min(500, self.interval + 20)
        self.update_speed()

    # Apply new speed to animation
    def update_speed(self):
        if self.ani: self.ani.event_source.interval = self.interval
        self.refresh_display()

    # Reset simulation
    # Returns to startup UI
    def reset_sim(self):
        self.is_initialised = self.is_running = self.is_paused = False
        self.interval = 100
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        self.update_button_ui(self.btn_init, True)
        self.update_button_ui(self.btn_random, False)
        self.update_button_ui(self.btn_pause, False)
        self.status_banner.config(text="STEP 1: INITIALISE ENVIRONMENT")
        self.time_label.config(text="Day 0: 00:00")
        self.sim_text.set("Waiting for input...")
        self.risk_label.config(text="RISK LEVEL: STABLE", bg="#27ae60")
        self.current_risk_state = "STABLE"
        self.draw_blank()
        self.ax_line.clear()
        self.canvas.draw()

    # Animation callback function called each frame
    # Advances model by one step
    # Updates displays
    # Checks risk thresholds
    def update(self, frame):
        if not self.is_running or self.is_paused: return
        self.model.step()                           # Execute one simulation time step
        self.refresh_display()
        burned = self.model.history["burned"][-1]
        new_risk = self.current_risk_state

        # Evaluation risk level based on percentage cells BURNED
        # >10% = WARNING, >40% = CRITICAL
        if burned > (self.model.initial_tree_count * 0.4):
            new_risk = "CRITICAL"
        elif burned > (self.model.initial_tree_count * 0.1):
            new_risk = "WARNING"
        else:
            new_risk = "STABLE"

        # Emergency alert popup windows
        if new_risk != self.current_risk_state:
            self.current_risk_state = new_risk
            # Critical Alert - immediate evacuation
            if new_risk == "CRITICAL":
                self.risk_label.config(text="RISK LEVEL: CRITICAL", bg="#c0392b")
                messagebox.showwarning("EMERGENCY ALERT", "EVACUATE NOW: An immediate evacuation has been ordered due to approaching wildfire. Life-threatening conditions are present in your area.\n\nAction:\nLeave the area immediately. Close all windows and doors before departing.\nRoute:\nFollow directions from emergency personnel and use main roads.\nDO NOT DELAY. This is an official order from local law enforcement.")
            # Warning Alert - prepare yo evacuate
            elif new_risk == "WARNING":
                self.risk_label.config(text="RISK LEVEL: WARNING", bg="#e67e22")
                messagebox.showwarning("EMERGENCY ALERT", "WILDFIRE WARNING: A wildfire has been reported in your vicinity. Local authorities advise all residents to prepare for potential evacuation.\n\nAction:\nPack an emergency kit, fuel vehicles, and monitor local news.\nStatus:\nBe ready to leave at a moment's notice. If you feel unsafe, do not wait for an order, leave now.")
        
        # Stop simulation when fire completely burns out
        if self.model.history["burning"][-1] == 0:
            self.is_running = False
            self.status_banner.config(text="SIMULATION FINISHED")
            self.update_button_ui(self.btn_pause, False)
            if self.ani: self.ani.event_source.stop()

    # Redraw grid and line graph with current model state
    # Called after each time step
    def refresh_display(self):
        self.ax_grid.clear()

        # Render grid with colours
        self.ax_grid.imshow(self.model.grid, cmap=self.fire_cmap, vmin=0, vmax=3,
                            extent=[-0.5, self.model.w - 0.5, self.model.h - 0.5, -0.5])
        self.ax_grid.set_xlim(-0.5, self.model.w - 0.5)
        self.ax_grid.set_ylim(self.model.h - 0.5, -0.5)
        self.ax_grid.set_aspect('equal')

        # Overlay orange embers
        if hasattr(self.model, 'embers') and self.model.embers:
            ey, ex, _ = zip(*self.model.embers)
            self.ax_grid.scatter(ex, ey, c='#e67e22', s=6.5, marker='*', alpha=0.8)

        # Upate live analytics panel
        self.ax_grid.set_axis_off()
        if self.model:
            self.time_label.config(text=f"Day {self.model.day_count}: {int(self.model.time_of_day):02d}:00")
            current_step = len(self.model.history["burned"])
            burned = self.model.history["burned"][-1] if current_step > 0 else 0
            burning = self.model.history["burning"][-1] if current_step > 0 else 0
            healthy = self.model.history["trees"][-1] if current_step > 0 else self.model.initial_tree_count
            
            current_temp = self.model.diurnal_temp()
            prob_spread = self.model.get_p_base()

            display_algo = self.model.spread_type.replace("_", " ").title()

            directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            dir_idx = int((((90 - self.model.wind_dir) % 360) + 22.5) / 45) % 8
            cardinal = directions[dir_idx]

            wind_str = f"{int(((90 - self.model.wind_dir) % 360))}° {cardinal}"
            
            stats_str = (
                f"ALGORITHM:       {display_algo}\n"
                f"WIND:            {self.model.wind_speed}km/h @ {wind_str}\n"
                f"TEMPERATURE:     {current_temp:.1f}°C\n"
                f"SOIL MOISTURE:   {self.model.soil_moisture:.2f}\n"
                f"P(SPREAD):       {prob_spread:.3f}\n"
                f"{'—'*36}\n"
                f"TOTAL:           {self.model.initial_tree_count}\n"
                f"HEALTHY:         {healthy}\n"
                f"BURNING:         {burning}\n"
                f"BURNED:          {burned}"
            )
            self.sim_text.set(stats_str)

            # Update line graph with historical data
            self.ax_line.clear()
            self.ax_line.plot(self.model.history["trees"],      color="#27ae60", label="Healthy")
            self.ax_line.plot(self.model.history["burning"],    color="#eaa400", label="Burning")
            self.ax_line.plot(self.model.history["burned"],     color="#000000", label="Burned")
            self.ax_line.legend(loc="upper right", fontsize='x-small')
            self.ax_line.set_xlabel("Time (Hours)")
            self.ax_line.set_ylabel("Cell Count")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = WildfireDashboard(root)
    root.mainloop()