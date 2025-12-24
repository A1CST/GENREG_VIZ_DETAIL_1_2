# GENREG Evolution Visualizer

<div align="center">

![GENREG Visualizer](screenshot.png)

**An interactive visualization of evolutionary machine learning without backpropagation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pygame](https://img.shields.io/badge/pygame-2.5.0+-green.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#overview) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Controls](#controls)

</div>

---

## Overview

GENREG (Genetic Regulation) Evolution Visualizer is an educational tool that demonstrates how **trust-based evolutionary selection** can replace gradient descent in machine learning. Instead of training neural networks through backpropagation, GENREG evolves populations of organisms using biological principles: mutation, crossbreeding, and selection based on "trust scores."

This visualizer makes the abstract concrete through three synchronized panels:
- 🎮 **Snake Game Evolution**: Watch behavior progress from random wall-crashing to intelligent food-seeking
- 🧬 **Protein Cascade**: See real-time neural activation as organisms process sensory information
- 👥 **Population Genetics**: Observe mutations, crossbreeding, and natural selection in action

### Why GENREG?

Traditional neural networks use:
- **Gradient descent** to optimize loss functions
- **Backpropagation** to compute weight updates
- **Training data** with explicit labels

GENREG uses:
- **Trust scores** instead of loss (emergent fitness signal)
- **Evolutionary selection** instead of backpropagation
- **Self-play** without labeled training data
- **Protein networks** that compute trust deltas from gameplay

This approach mirrors biological evolution: organisms that survive and reproduce pass on their genes. No explicit "teaching signal" required—just selection pressure.

---

## Features

### 🎬 Pre-Computed Playback System
- Evolution runs offline, visualization replays saved state
- Timeline scrubbing with generation markers
- Variable playback speed (0.25x - 4x)
- Smooth interpolation between discrete states

###  Dual-Path Neural Architecture
**Protein Network (Trust Computation)**
- 6 specialized protein types:
  - **Sensors**: Normalize environmental signals
  - **Comparators**: Compare inputs (diff/ratio/greater/less)
  - **Trend Detectors**: Track velocity and momentum
  - **Integrators**: Rolling accumulation with decay
  - **Gates**: Conditional activation with hysteresis
  - **Trust Modifiers**: Convert signals to fitness deltas

**Neural Controller (Action Selection)**
- Standard feedforward network (11 → 32 → 4)
- Processes 11 sensory inputs (position, food direction, walls, energy)
- Outputs movement probabilities (UP/DOWN/LEFT/RIGHT)

###  Real-Time Visualizations

**Snake Game Panel**
- Three evolutionary stages shown simultaneously
- Stage 1 (Early): Random wall-crashing behavior
- Stage 2 (Mid): Exploratory wandering with accidental food discovery
- Stage 3 (Late): Intelligent pathfinding and strategic play

**Protein Cascade Panel** 
- Interactive network graph with 11 sensors → 12 hidden → 4 motors
- Click nodes to highlight specific pathways
- Real-time activation flow with particle effects
- Connection strength based on weights
- Trust value visualization via node colors

**Population Genetics Panel**
- Track 3 example organisms (elite/average/struggling)
- Live mutation/crossbreed/cull animations
- Trust score evolution over generations
- Genetic lineage tracing

### 🔬 Educational Features
- Stage transition markers showing behavioral milestones
- Trust score breakdowns (fitness + stability components)
- Protein activation timelines
- Generation-by-generation population statistics

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/genreg-visualizer.git
cd genreg-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the visualizer
python main.py
```

### Dependencies
```
pygame>=2.5.0
numpy>=1.24.0
```

---

## Usage

### Interactive Menu Mode
```bash
python main.py
```
Launches an interactive menu with options to:
1. Load existing checkpoint files (from official GENREG training)
2. Use pre-generated simulation data
3. Generate new evolution from scratch

### Command-Line Modes

**Generate new evolution data:**
```bash
python main.py --simulate
```

**Visualize existing data:**
```bash
python main.py --visualize
```

**Load specific checkpoint:**
```bash
python main.py --checkpoint path/to/checkpoint_gen_00500.pkl
```

**Custom data directory:**
```bash
python main.py --data-dir my_simulation_data
```

### Configuration

Edit `config.py` to customize evolution parameters:

```python
SIMULATION_CONFIG = {
    'population_size': 50,          # Number of organisms
    'total_generations': 500,       # Evolution duration
    'mutation_rate': 0.05,          # 5% mutation probability
    'elite_rate': 0.10,             # 10% elite preservation
    'crossover_rate': 0.40,         # 40% sexual reproduction
    'grid_size': 10,                # Snake game grid (10x10)
    'games_per_evaluation': 3,      # Games per fitness eval
}
```

---

## Controls

### Playback Controls
| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `LEFT` / `RIGHT` | Seek backward / forward |
| `UP` / `DOWN` | Increase / decrease speed |
| `R` | Reset to beginning |
| `1` / `2` / `3` | Jump to Early / Mid / Late stage |
| `Mouse Click` | Scrub timeline |

### Visualization Controls
| Key | Action |
|-----|--------|
| `P` | Open detailed protein cascade view |
| `Click Node` | Highlight neural pathway |
| `Click Background` | Clear highlights |
| `ESC` | Exit |

### Protein Cascade Interactions
- **Click Sensor**: Highlights pathway from that sensor through network to motors
- **Click Motor**: Traces back to contributing sensors
- **Click Hidden Node**: Shows incoming and outgoing connections separately
- **Click Empty Space**: Restores default view (all connections visible)

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                   GENREG VISUALIZER                      │
├─────────────────────────────────────────────────────────┤
│  Phase 1: Pre-Computation (Offline)                     │
│  ┌────────────────────────────────────────────┐        │
│  │ - Run 500 generations of evolution         │        │
│  │ - Each genome plays Snake 3x               │        │
│  │ - Log protein activations, trust scores    │        │
│  │ - Record genetic events (mutations, etc)   │        │
│  │ - Identify behavioral stage exemplars      │        │
│  └────────────────────────────────────────────┘        │
│                        ↓                                 │
│  Phase 2: Playback (Interactive)                        │
│  ┌────────────────────────────────────────────┐        │
│  │ - Load pre-computed JSON/NumPy logs        │        │
│  │ - Interpolate between discrete snapshots   │        │
│  │ - Render synchronized 3-panel view         │        │
│  │ - User controls timeline, highlights       │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
genreg-visualizer/
├── main.py                 # Entry point, menu system
├── config.py               # All configuration parameters
├── data_structures.py      # Genome, proteins, neural network classes
├── simulation.py           # Evolution engine, Snake game logic
├── visualizer.py           # Pygame rendering, UI, animations
├── requirements.txt        # Python dependencies
└── simulation_data/        # Generated during pre-computation
    ├── evolution_log.json          # Generation snapshots
    ├── genetic_events.json         # Mutations, crossbreeds, culls
    ├── stage_exemplars.json        # Early/Mid/Late examples
    ├── protein_activations.npz     # Neural activation timeseries
    └── config.json                 # Simulation parameters
```

### Data Structures

**Genome** (Organism DNA)
- List of Protein objects with parameters
- NeuralController with weight matrices
- Trust score history
- Genetic lineage information

**Protein Types**
1. `SensorProtein`: Normalizes raw signals (min/max scaling, running stats)
2. `ComparatorProtein`: Compares two inputs (difference, ratio, boolean)
3. `TrendProtein`: Tracks delta between timesteps (velocity/momentum)
4. `IntegratorProtein`: Exponentially-weighted accumulation
5. `GateProtein`: Threshold-based switching with hysteresis
6. `TrustModifierProtein`: Maps signal to trust delta with saturation

**NeuralController**
- Simple 2-layer MLP: Input → Hidden (ReLU) → Output (Softmax)
- Weights evolved, not trained via gradient descent
- Completely independent from protein trust computation

---

## How GENREG Works

### Evolution Loop (Simplified)

```python
for generation in range(500):
    # 1. Evaluate fitness: each genome plays Snake
    for genome in population:
        trust_score = play_snake_games(genome)  # Protein network computes trust
        genome.fitness = trust_score
    
    # 2. Selection: cull bottom 40%
    population.sort(by=lambda g: g.fitness)
    survivors = population[top_60%]
    
    # 3. Reproduction: crossbreed + mutate to refill population
    while len(population) < population_size:
        parent1, parent2 = select_weighted_by_trust(survivors)
        offspring = crossbreed(parent1, parent2)
        
        if random() < mutation_rate:
            mutate_random_proteins(offspring)
        
        population.append(offspring)
```

### Trust Score Computation

During gameplay, at each timestep:

```python
# Get environmental signals
signals = {
    'steps_alive': 42,
    'energy': 0.8,
    'dist_to_food': 0.3,
    'head_x': 5, 'head_y': 5,
    'food_x': 7, 'food_y': 8,
    ...
}

# Protein network processes signals
protein_outputs = {}
for protein in genome.proteins:
    protein_outputs[protein.name] = protein.forward(signals, protein_outputs)

# Trust modifier proteins produce fitness deltas
trust_delta = sum(protein_outputs[p] for p in trust_proteins)

# Accumulate over episode
cumulative_trust += trust_delta
```

Trust serves as the **selection pressure**: genomes with higher cumulative trust are more likely to survive and reproduce.

### Why "Trust" Instead of "Loss"?

Traditional ML minimizes a loss function (lower is better). GENREG maximizes trust (higher is better). The key difference:

- **Loss**: Requires labeled data, differentiable objective, gradient computation
- **Trust**: Emergent from gameplay, no labels needed, evolved not trained

Trust represents "how much can we trust this genome to achieve goals?" It's computed bottom-up from protein activations, not imposed top-down by a human-designed loss function.

---

## Educational Use Cases

### For ML Researchers
- Explore alternatives to gradient descent
- Study emergent behavior from evolutionary pressure
- Compare evolved vs. trained neural networks
- Investigate neuroevolution and genetic algorithms

### For Educators
- Teach evolutionary algorithms visually
- Demonstrate biological inspiration in AI
- Show genetic operators (mutation, crossover, selection)
- Illustrate fitness landscapes and local optima

### For Students
- Understand how evolution produces intelligent behavior
- See the connection between genotype (genome) and phenotype (behavior)
- Learn about neural networks without calculus
- Explore reinforcement learning concepts (reward signals, policies)

---

## Advanced Features

### Checkpoint Loading

Load genomes from official GENREG training runs:

```bash
python main.py --checkpoint checkpoint_gen_00500.pkl
```

The visualizer automatically converts external genome formats, runs demo games, and generates visualization data.

### Protein Cascade Deep Dive

Press `P` during playback to open the detailed protein network view:
- Full network topology with all connections
- Weight strength visualization (thicker = stronger)
- Activation flow particles
- Interactive pathway highlighting

**Click interactions:**
- Click any sensor (S0-S10) to trace its influence on motor outputs
- Click any motor (M0-M3) to see which sensors contribute
- Click hidden layer neurons (H0-H11) to isolate inputs and outputs

### Stage Exemplar Analysis

The visualizer identifies three behavioral milestones:
1. **Early Stage (Gen 0-150)**: Random, reactive behavior. Snake dies quickly from wall collisions.
2. **Mid Stage (Gen 150-350)**: Exploratory behavior. Snake wanders and occasionally finds food by chance.
3. **Late Stage (Gen 350-500)**: Strategic behavior. Snake actively pursues food and avoids obstacles.

Each stage loops its exemplar gameplay continuously in the left panel, while the protein and population panels show the current timeline position.

---

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame numpy
```

### "Data directory not found"
Run simulation first:
```bash
python main.py --simulate
```

### Low FPS / Performance Issues
- Reduce `PARTICLE_COUNT` in `config.py`
- Lower screen resolution in `config.py` (WINDOW_WIDTH, WINDOW_HEIGHT)
- Disable particle effects in visualizer settings

### Checkpoint Format Errors
Ensure checkpoint files are from compatible GENREG versions. The visualizer attempts automatic conversion but may fail on drastically different architectures.

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional protein types (e.g., oscillators, memory banks)
- [ ] Multi-agent Snake gameplay (competitive/cooperative)
- [ ] Export trained genomes to standard formats (ONNX, PyTorch)
- [ ] Performance optimizations for larger populations
- [ ] Web-based visualization (convert to JavaScript/WebGL)
- [ ] Statistical analysis tools (convergence plots, diversity metrics)

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/genreg-visualizer.git
cd genreg-visualizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests (if available)
pytest
```

---

## Citation

If you use this visualizer in your research or educational materials, please cite:

```bibtex
@software{genreg_visualizer,
  author = {Your Name},
  title = {GENREG Evolution Visualizer},
  year = {2024},
  url = {https://github.com/yourusername/genreg-visualizer}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by biological evolution and neuroevolution research
- Built with [Pygame](https://www.pygame.org/) for rendering
- Snake game mechanics adapted from classic implementations
- Protein concept inspired by gene regulatory networks in systems biology

---

## FAQ

**Q: Is this a real training system or just a visualization?**  
A: Both! The evolution engine is a simplified but functional implementation of GENREG principles. It genuinely evolves organisms through selection pressure, but it's optimized for educational clarity rather than performance.

**Q: Can I use this to train agents for other games?**  
A: Yes, with modifications. Replace the Snake game with your environment, adjust the sensory signals, and the evolution loop will work similarly. However, GENREG is primarily a research/educational tool, not a production RL framework.

**Q: How does this compare to NEAT (NeuroEvolution of Augmenting Topologies)?**  
A: NEAT evolves network topology (adding nodes and connections). GENREG uses a fixed protein template but evolves the parameters within each protein type. Both are neuroevolution approaches but with different evolutionary operators.

**Q: What's the difference between proteins and neurons?**  
A: Proteins are **algorithmic processing units** (comparators, integrators, gates) that compute trust deltas. Neurons are simple weighted sums with activations that compute actions. Proteins = fitness signal, neurons = policy. They work in parallel on the same sensory input.

**Q: Why is the snake game so simple (10x10 grid)?**  
A: Educational focus! Smaller grids make evolution faster and behavior changes more visible. The principles scale to larger, more complex environments.

**Q: Can I train this with gradient descent instead?**  
A: Technically yes—you could freeze the protein network and train the neural controller with policy gradients. But that defeats the purpose! GENREG demonstrates that evolution alone, without gradients, can produce intelligent behavior.

---

<div align="center">

**Built by AsyncVibes**

[⬆ Back to Top](#genreg-evolution-visualizer)

</div>
