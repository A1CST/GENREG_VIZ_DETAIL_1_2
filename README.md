# GENREG Evolution Visualizer

<div align="center">

![GENREG Visualizer](/assets/Screenshot 2025-12-23 185934.png)

**Interactive visualization of real evolutionary machine learning**  
*Demonstration of 500 generations evolved with official GENREG proteins*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pygame](https://img.shields.io/badge/pygame-2.5.0+-green.svg)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Overview](#overview) • [Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Controls](#controls)

</div>

---

> ###  Important: Demonstration Repository
> This is a **visualization-only repository**. The evolution you'll see was pre-computed using the **official GENREG protein system** (not included). This demonstrates real evolutionary learning results, but you cannot run new evolution or modify the training system. Perfect for education, exploration, and understanding GENREG principles!

---

## Overview

GENREG (Genetic Regulation) Evolution Visualizer is an **educational demonstration tool** that shows how **trust-based evolutionary selection** can replace gradient descent in machine learning. This visualizer replays pre-computed evolution data from the official GENREG system, making the abstract principles of evolutionary learning concrete and observable.

> ** Important Note:** This repository contains a **visualization-only demonstration**. The evolution was pre-computed using the **official GENREG protein system** (not included in this repository). The visualizer replays this saved evolutionary history to demonstrate the concepts. This is an educational tool to understand GENREG principles, not a standalone training system.

This visualizer makes evolutionary learning concrete through three synchronized panels:
-  **Snake Game Evolution**: Watch behavior progress from random wall-crashing to intelligent food-seeking (real evolved agents!)
-  **Protein Cascade**: See real-time neural activation from actual GENREG protein networks
-  **Population Genetics**: Observe mutations, crossbreeding, and natural selection that actually occurred

### What is GENREG?

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

### What's Included vs. What's Not

**✅ Included in this repository:**
- Complete Pygame visualization system
- Pre-computed evolution data (500 generations)
- Interactive playback with timeline control
- Protein network visualization
- Educational documentation

**❌ NOT included (official GENREG system):**
- Full protein system implementation
- Evolution engine (pre-computation code)
- Training/evolution capabilities
- Official GENREG checkpoint integration

The data you see was generated using the real GENREG protein architecture, but this visualizer is **playback-only** for demonstration purposes.

---

## Features

###  Pre-Computed Playback System
- **Real evolution data** from official GENREG protein system (500 generations)
- Timeline scrubbing with generation markers
- Variable playback speed (0.25x - 4x)
- Smooth interpolation between discrete evolutionary snapshots

###  Authentic GENREG Architecture Visualization
**Protein Network (Trust Computation)** - *As used in actual evolution*
- 6 specialized protein types evolved through natural selection:
  - **Sensors**: Normalize environmental signals
  - **Comparators**: Compare inputs (diff/ratio/greater/less)
  - **Trend Detectors**: Track velocity and momentum
  - **Integrators**: Rolling accumulation with decay
  - **Gates**: Conditional activation with hysteresis
  - **Trust Modifiers**: Convert signals to fitness deltas

**Neural Controller (Action Selection)** - *Evolved alongside proteins*
- Standard feedforward network (11 → 32 → 4)
- Processes 11 sensory inputs (position, food direction, walls, energy)
- Outputs movement probabilities (UP/DOWN/LEFT/RIGHT)
- Weights evolved (not trained!) through genetic operators

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

### Quick Start (Demonstration Mode)

```bash
# Clone the repository
git clone https://github.com/yourusername/genreg-visualizer.git
cd genreg-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the visualizer (uses included pre-computed data)
python main.py
```

The visualizer will automatically load the pre-computed evolution data and start playback.

### What You'll See

The included dataset contains:
- **500 generations** of evolution using official GENREG proteins
- **50 organisms** per generation competing through natural selection
- **Real behavioral progression** from random to intelligent play
- **Authentic protein activations** from the actual GENREG system

> **Note:** This is a **demonstration visualizer only**. You cannot run new evolution or modify the protein system. The data you're viewing was generated using the proprietary GENREG implementation, which is not included in this repository.

### Command-Line Options

**Run visualizer with pre-computed data:**
```bash
python main.py --visualize
```

**Specify custom data directory:**
```bash
python main.py --data-dir my_simulation_data
```

> **Removed features:** `--simulate` and `--checkpoint` options are not available in the demonstration version. These required the full GENREG system.

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

### System Design (Demonstration Version)

```
┌─────────────────────────────────────────────────────────┐
│              GENREG VISUALIZER (DEMO)                    │
├─────────────────────────────────────────────────────────┤
│  Phase 1: Pre-Computation (COMPLETED - Not Included)    │
│  ┌────────────────────────────────────────────┐        │
│  │ ✅ Run 500 generations (OFFICIAL GENREG)   │        │
│  │ ✅ Evolved proteins via natural selection  │        │
│  │ ✅ Logged all activations & genetic events │        │
│  │ ✅ Identified behavioral milestones        │        │
│  │ ✅ Saved to JSON/NumPy format              │        │
│  └────────────────────────────────────────────┘        │
│                        ↓                                 │
│  Phase 2: Playback (THIS REPOSITORY)                    │
│  ┌────────────────────────────────────────────┐        │
│  │  Included: Pre-computed data files       │        │
│  │  Pygame visualization engine             │        │
│  │  Timeline control & scrubbing            │        │
│  │  Interactive protein network explorer    │        │
│  │  Population genetics animations          │        │
│  └────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### What's Included in This Repository

```
genreg-visualizer/
├── main.py                 # Entry point (playback only)
├── config.py               # Visualization parameters
├── visualizer.py           # Pygame rendering engine
├── requirements.txt        # Dependencies (pygame, numpy)
│
├── simulation_data/        # 📦 PRE-COMPUTED (from official GENREG)
│   ├── evolution_log.json          # 500 generation snapshots
│   ├── genetic_events.json         # Mutations, crossbreeds, culls
│   ├── stage_exemplars.json        # Early/Mid/Late behavioral examples
│   ├── protein_activations.npz     # Real protein activation timeseries
│   └── config.json                 # Original evolution parameters
│
└── [EXCLUDED - Official GENREG system]
    ├── Full protein implementation
    ├── Evolution engine
    ├── Training/simulation code
    └── Checkpoint generation
```

### Data Authenticity

The included `simulation_data/` was generated using:
- **Official GENREG protein system** (6 protein types with evolved parameters)
- **Natural selection** over 500 generations (no human tuning)
- **Real trust-based fitness** (not hand-crafted reward functions)
- **Genetic operators** (mutation, crossover, elite preservation)

This is **genuine evolutionary data**, not a simplified demo or simulation.

### Data Structures (For Reference)

The pre-computed data uses these structures (simplified for visualization):

**Evolution Log** (`evolution_log.json`)
- Generation-by-generation snapshots (500 total)
- Population statistics (avg trust, best trust, diversity)
- Organism summaries (trust scores, survival status)
- Stage classifications (early/mid/late)

**Protein Activations** (`protein_activations.npz`)
- Timestep-by-timestep protein outputs
- 11 sensor activations (environmental signals)
- 12 hidden protein activations (processing layer)
- 4 motor activations (movement decisions)
- Trust delta per timestep

**Genetic Events** (`genetic_events.json`)
- Mutation events (which protein, parameter change)
- Crossbreed events (parent IDs, offspring ID)
- Cull events (eliminated organisms)
- Elite preservation records

**Stage Exemplars** (`stage_exemplars.json`)
- Representative gameplay for Early/Mid/Late stages
- Complete frame-by-frame recordings
- Snake positions, food locations, scores
- Associated protein activations

> **Note:** The actual protein classes and neural network implementation are not included in this repository. The data structures here are for visualization purposes only.

---

## Understanding What You're Seeing

### The Evolution That Happened (Pre-Computed)

The data you're viewing represents **500 generations of genuine evolutionary learning** using the official GENREG system. Here's what occurred:

**Generation 0-150 (Early Stage)**
- Random protein parameters
- Chaotic, reactive behavior
- Snake dies from wall collisions within seconds
- Trust scores: 0.1 - 0.3 range

**Generation 150-350 (Mid Stage)**  
- Emerging patterns in protein activations
- Exploratory wandering behavior
- Accidental food discovery
- Trust scores: 0.4 - 0.6 range

**Generation 350-500 (Late Stage)**
- Optimized protein parameters through selection
- Strategic food-seeking behavior
- Long survival times (100+ moves)
- Trust scores: 0.7 - 0.95 range

### Trust Scores: The Selection Pressure

Unlike traditional ML loss functions, GENREG uses **trust scores** as evolutionary fitness:

```
Trust = accumulated signal from protein network over entire episode
```

During gameplay:
1. Environmental signals enter protein network (distance to food, energy, position, etc.)
2. Proteins process signals through comparators, integrators, gates
3. Trust modifier proteins output fitness deltas based on game state
4. Cumulative trust determines survival probability

**High trust organisms** → survive and reproduce  
**Low trust organisms** → culled from population

No gradients, no backpropagation—just selection pressure on trust scores.

### Dual-Path Architecture

Each organism has TWO parallel systems:

**Path 1: Protein Network** (Trust Computation)
- Processes 11 environmental signals
- 6 protein types with evolved parameters
- Outputs trust deltas each timestep
- **Purpose:** Fitness signal for evolution

**Path 2: Neural Controller** (Action Selection)
- Same 11 inputs as proteins
- 2-layer feedforward network (11→32→4)
- Outputs movement probabilities
- **Purpose:** Behavioral policy

Both systems evolve together. Proteins don't control movement—they compute fitness. The neural network controls movement but has no concept of "good" or "bad" moves except through evolutionary selection.

### Why This Matters

This demonstrates that **intelligent behavior can emerge from evolution alone**, without:
- Labeled training data
- Hand-crafted reward functions  
- Gradient descent
- Human supervision

The agents you see in the Late Stage genuinely learned to play Snake through survival pressure.

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

## Exploring the Visualization

### Interactive Timeline

The bottom control bar lets you explore 500 generations of evolution:
- **Scrub** through time by clicking/dragging the timeline
- **Jump to milestones**: Generation 0 (random), Gen 150 (emerging), Gen 500 (mastery)
- **Speed control**: Watch evolution at 0.25x (slow-motion) to 4x (fast-forward)
- **Stage markers**: Visual indicators show Early → Mid → Late transitions

### Protein Cascade Deep Dive

Press `P` during playback to open the detailed protein network view:
- **Full network topology** showing all 11 sensors → 12 hidden proteins → 4 motors
- **Weight visualization**: Thicker connections = stronger weights (evolved parameters)
- **Real-time activation**: Watch signals flow through the actual protein network
- **Interactive pathway tracing**: Click to highlight specific information flows

**Click interactions:**
- **Click Sensor (S0-S10)**: Trace how that environmental signal influences motor outputs
- **Click Motor (M0-M3)**: See which sensors contribute to that movement decision  
- **Click Hidden Protein (H0-H11)**: Isolate inputs and outputs to understand processing
- **Click Background**: Restore full network view

This lets you reverse-engineer the evolved computation: "How does the snake decide to turn left when food is northwest?"

### Stage Analysis

The visualizer automatically identifies three behavioral milestones:

**Early Stage (Gen 0-150)**
- **Behavior**: Random, reactive wall-crashing
- **Trust**: 0.1 - 0.3
- **Protein patterns**: Chaotic activations with no stable logic
- **Example**: Snake dies within 5-10 moves

**Mid Stage (Gen 150-350)**
- **Behavior**: Exploratory wandering, accidental food discovery
- **Trust**: 0.4 - 0.6  
- **Protein patterns**: Emerging consistency, basic obstacle avoidance
- **Example**: Snake collects 2-5 food items before dying

**Late Stage (Gen 350-500)**
- **Behavior**: Strategic food-seeking, long survival
- **Trust**: 0.7 - 0.95
- **Protein patterns**: Stable, purposeful activations
- **Example**: Snake collects 10-25 food items with efficient pathfinding

Each stage continuously loops its exemplar game in the left panel, letting you compare all three behavioral levels simultaneously.

---

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame numpy
```

### "Data directory not found" or "Missing simulation_data/"
Ensure you cloned the full repository including the `simulation_data/` directory with pre-computed evolution files. The repository should include:
- `evolution_log.json` (~50-100 MB)
- `protein_activations.npz` (~200-300 MB)
- `genetic_events.json` (~20-30 MB)
- Other data files

If these are missing, the repository may not have been cloned completely.

### Low FPS / Performance Issues
- Close other applications to free up resources
- Reduce particle effects by editing `config.py`:
  ```python
  PARTICLE_COUNT = 100  # Reduce from default
  ```
- Lower screen resolution:
  ```python
  WINDOW_WIDTH = 1280   # Down from 1600
  WINDOW_HEIGHT = 720   # Down from 900
  ```

### Visualization looks wrong / protein network shows no connections
Ensure you're using Python 3.8+ and the correct pygame version:
```bash
python --version  # Should be 3.8 or higher
pip show pygame   # Should be 2.5.0 or higher
```

### "Can't run simulation" errors
This is expected! The simulation engine is not included in this demonstration repository. You can only view the pre-computed evolution data.

---

## Contributing

This is a **demonstration repository** showcasing GENREG principles. Contributions welcome for improving the visualization and educational value!

### Areas for Enhancement

**Visualization Improvements:**
- [ ] Enhanced particle effects for protein activation flow
- [ ] Improved timeline scrubbing UI/UX
- [ ] Additional color schemes and themes
- [ ] Performance optimizations for low-end hardware
- [ ] Protein network layout algorithms (force-directed graphs)

**Educational Features:**
- [ ] Tooltips explaining protein types and their functions
- [ ] Guided tutorial mode with annotations
- [ ] Export visualizations to video/GIF
- [ ] Interactive quizzes about evolution concepts
- [ ] Comparison view (side-by-side generations)

**Documentation:**
- [ ] More detailed protein type explanations
- [ ] Video walkthrough of the visualizer
- [ ] Educational lesson plans for teachers
- [ ] Translation to other languages

**Technical:**
- [ ] Web-based version (convert to JavaScript/WebGL)
- [ ] Mobile-friendly responsive design
- [ ] Docker containerization
- [ ] Automated testing for UI components

### What NOT to Contribute

Please **do not** submit:
- ❌ Alternative evolution engines or training code
- ❌ Modified protein implementations
- ❌ New simulation data (this repo uses official GENREG results)
- ❌ Attempts to reverse-engineer the full GENREG system

This repository is intentionally limited to visualization and education.

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
A: This is a **visualization of real training results**. The evolution you're seeing is authentic—it was generated using the official GENREG protein system over 500 generations. However, this repository only contains the visualizer and pre-computed data, not the training system itself.

**Q: Can I run my own evolution or train new agents?**  
A: No, this is a **demonstration-only repository**. The full GENREG system (protein implementation, evolution engine) is not included. You can only view the pre-computed evolutionary history.

**Q: Is the data genuine or synthetic?**  
A: 100% genuine! The behaviors, protein activations, and trust scores you see are from actual evolution using the official GENREG system. Nothing was hand-tuned or faked for demonstration purposes.

**Q: Can I use this to train agents for other games?**  
A: Not with this repository. This is a playback visualizer only. To train on other environments, you would need the full GENREG implementation (not included here).

**Q: Why share just the visualizer without the training code?**  
A: Educational purposes! The visualizer demonstrates GENREG principles and evolutionary learning concepts without requiring users to run computationally expensive evolution themselves. The pre-computed data lets anyone explore the results immediately.

**Q: How does this compare to NEAT (NeuroEvolution of Augmenting Topologies)?**  
A: NEAT evolves network topology (adding nodes and connections). GENREG uses a fixed protein template but evolves the parameters within each protein type. Both are neuroevolution approaches but with different evolutionary operators. The key GENREG innovation is the **dual-path system**: proteins compute trust (fitness), neural network computes actions.

**Q: What's the difference between proteins and neurons?**  
A: **Proteins** are algorithmic processing units (comparators, integrators, gates) that compute trust deltas from environmental signals. **Neurons** are simple weighted sums with activations that compute movement actions. They work in parallel:
- Proteins → Trust score → Evolutionary fitness
- Neurons → Action probabilities → Behavior

**Q: Why is the snake game so simple (10x10 grid)?**  
A: Smaller grids make evolution faster and behavioral changes more visible for educational purposes. The principles demonstrated here scale to larger, more complex environments.

**Q: Can I see the protein implementation code?**  
A: The official GENREG protein code is not included in this repository. However, the `config.py` file documents the 6 protein types and their roles, and the visualizer shows their activations in real-time.

**Q: What license is the pre-computed data under?**  
A: The visualization code is MIT licensed. The pre-computed evolution data is provided for educational/demonstration purposes. Check the LICENSE file for details.

---

<div align="center">

**Built by AsyncVibes**

[⬆ Back to Top](#genreg-evolution-visualizer)

</div>
