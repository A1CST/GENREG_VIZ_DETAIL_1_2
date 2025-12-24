"""
GENREG Evolution Visualizer Configuration
Official GENREG architecture parameters
"""

# Display settings
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FPS = 60

# Panel dimensions (three equal panels)
PANEL_WIDTH = WINDOW_WIDTH // 3
PANEL_HEIGHT = WINDOW_HEIGHT - 100  # Leave room for controls
CONTROL_HEIGHT = 100

# Colors - Organic/biological theme
COLORS = {
    # Background colors
    'bg_dark': (15, 20, 25),
    'bg_panel': (20, 28, 35),
    'bg_control': (25, 35, 45),

    # Snake game colors
    'snake_head': (80, 200, 120),
    'snake_body': (60, 160, 90),
    'snake_tail': (40, 120, 70),
    'food': (255, 100, 100),
    'food_glow': (255, 150, 150),
    'wall': (100, 100, 120),
    'grid': (30, 40, 50),
    'energy_high': (100, 255, 150),
    'energy_low': (255, 100, 100),
    'energy_bar_bg': (40, 50, 60),

    # Protein cascade colors - 6 types
    'protein_inactive': (60, 60, 80),
    'protein_active': (100, 180, 255),
    'protein_glow': (150, 200, 255),

    # Protein type colors
    'sensor_protein': (255, 200, 100),      # Yellow/orange - receives signals
    'comparator_protein': (100, 200, 255),  # Light blue - compares inputs
    'trend_protein': (200, 150, 255),       # Purple - tracks changes
    'integrator_protein': (150, 255, 200),  # Cyan/green - accumulates
    'gate_protein': (255, 150, 150),        # Pink/red - conditional
    'trust_modifier_protein': (255, 215, 0), # Gold - outputs trust

    # Legacy protein colors (for backward compatibility)
    'motor_protein': (100, 255, 150),
    'hidden_protein': (180, 130, 255),

    # Connection colors
    'connection_weak': (50, 60, 70),
    'connection_strong': (100, 150, 200),
    'connection_inactive': (100, 180, 220),  # Neutral cyan at 18% opacity
    'connection_highlight': (0, 220, 255),   # Bright cyan for highlights
    'particle': (200, 220, 255),

    # Neural network colors
    'nn_input': (100, 200, 255),
    'nn_hidden': (180, 130, 255),
    'nn_output': (100, 255, 150),
    'nn_weight_positive': (100, 255, 150),
    'nn_weight_negative': (255, 100, 100),
    'nn_active': (255, 255, 100),

    # Trust display colors
    'trust_positive': (100, 255, 150),
    'trust_negative': (255, 100, 100),
    'trust_neutral': (150, 150, 150),
    'trust_bar_bg': (40, 50, 60),
    'trust_delta_positive': (0, 255, 100),
    'trust_delta_negative': (255, 80, 80),

    # Population genetics colors
    'organism_healthy': (100, 200, 150),
    'organism_weak': (200, 100, 100),
    'organism_elite': (255, 215, 0),
    'mutation': (255, 150, 50),
    'crossbreed': (200, 100, 255),
    'cull': (255, 80, 80),
    'trust_high': (100, 255, 150),
    'trust_low': (255, 100, 100),

    # UI colors
    'text_primary': (220, 230, 240),
    'text_secondary': (150, 160, 170),
    'text_highlight': (255, 255, 255),
    'button': (50, 70, 90),
    'button_hover': (70, 90, 110),
    'button_active': (90, 120, 150),
    'timeline_bg': (30, 40, 50),
    'timeline_fill': (80, 140, 200),
    'timeline_marker': (255, 200, 100),

    # Stage indicators
    'stage_early': (255, 100, 100),
    'stage_mid': (255, 200, 100),
    'stage_late': (100, 255, 150),
}

# Official GENREG simulation parameters
SIMULATION_CONFIG = {
    # Population settings
    'population_size': 50,
    'total_generations': 500,

    # Evolution rates (official GENREG values)
    'elite_rate': 0.10,           # 10% elite survive
    'crossover_rate': 0.40,       # 40% crossbreed
    'mutation_rate': 0.05,        # 5% mutation rate
    'mutation_scale': 0.20,       # 20% mutation scale
    'trust_inheritance': 0.05,    # 5% trust inherited by offspring

    # Fitness calculation
    'trust_weight': 0.70,         # 70% trust
    'stability_weight': 0.30,     # 30% stability

    # Snake game settings (official GENREG)
    'grid_size': 10,              # 10x10 grid (was 20)
    'games_per_evaluation': 3,    # Games per fitness eval
    'base_energy': 25,            # Starting energy
    'energy_per_food': 2,         # Energy gain per food

    # Neural network settings
    'num_signals': 11,            # 11 input signals
    'nn_hidden_size': 32,         # Hidden layer size
    'nn_output_size': 4,          # 4 actions (UP, DOWN, LEFT, RIGHT)

    # Protein network settings
    'num_sensor_proteins': 11,    # One per signal
    'num_processing_proteins': 6, # Trend, Comparator, Integrator, Gate, etc.
    'num_trust_proteins': 3,      # Trust modifier proteins

    # Visualization sampling
    'sample_interval': 10,        # Log every N generations in detail
    'exemplar_generations': [1, 50, 100, 250, 500],
}

# Signal names for display
SIGNAL_NAMES = [
    "steps_alive",
    "energy",
    "dist_to_food",
    "head_x",
    "head_y",
    "food_x",
    "food_y",
    "food_dx",
    "food_dy",
    "near_wall",
    "alive"
]

# Action names for display
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]

# Protein type display info
PROTEIN_TYPES = {
    'sensor': {
        'color': 'sensor_protein',
        'label': 'Sensor',
        'description': 'Normalizes raw input signals'
    },
    'comparator': {
        'color': 'comparator_protein',
        'label': 'Comparator',
        'description': 'Compares two inputs (diff/ratio/greater/less)'
    },
    'trend': {
        'color': 'trend_protein',
        'label': 'Trend',
        'description': 'Detects velocity/momentum of change'
    },
    'integrator': {
        'color': 'integrator_protein',
        'label': 'Integrator',
        'description': 'Rolling accumulation with decay'
    },
    'gate': {
        'color': 'gate_protein',
        'label': 'Gate',
        'description': 'Conditional activation with hysteresis'
    },
    'trust_modifier': {
        'color': 'trust_modifier_protein',
        'label': 'Trust',
        'description': 'Converts signal to trust delta'
    },
}

# Playback settings
PLAYBACK_CONFIG = {
    'default_speed': 1.0,
    'speed_options': [0.25, 0.5, 1.0, 2.0, 4.0],
    'frames_per_generation': 60,  # At 1x speed
}

# Animation timings (in seconds)
ANIMATION_TIMING = {
    'protein_activation': 0.3,
    'particle_travel': 0.5,
    'mutation_flash': 0.4,
    'crossbreed_animation': 0.6,
    'cull_fade': 0.5,
    'stage_transition': 1.0,
    'snake_move': 0.1,
    'trust_pulse': 0.2,
    'nn_propagation': 0.3,
}
