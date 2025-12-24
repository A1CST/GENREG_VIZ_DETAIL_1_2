"""
GENREG Data Structures
Official GENREG architecture with dual-path system:
- Proteins compute TRUST (fitness signal)
- Neural Controller computes ACTION (movement)
"""

import numpy as np
import json
import copy
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GeneticEventType(Enum):
    MUTATION = "mutation"
    CROSSBREED = "crossbreed"
    CULL = "cull"
    BIRTH = "birth"
    ELITE_PRESERVE = "elite_preserve"


# ================================================================
# Base Protein Class
# ================================================================
class Protein:
    """Base class for all protein types in the regulatory genome."""

    def __init__(self, name: str, protein_type: str):
        self.name = name
        self.type = protein_type

        # Internal state (biological memory)
        self.state: Dict = {}

        # Cached output during one forward pass
        self.output: float = 0.0

        # Inputs - can be environment signals or other protein outputs
        self.inputs: List[str] = []

        # Hyperparameters
        self.params: Dict = {}

    def bind_inputs(self, inputs: List[str]):
        """Bind list of input names."""
        self.inputs = inputs

    def mutate_param(self, key: str, scale: float = 0.1):
        """Gaussian mutation of a parameter."""
        if key in self.params:
            val = self.params[key]

            if isinstance(val, (int, float)):
                delta = random.gauss(0, scale * (abs(val) + 1e-9))
                self.params[key] = val + delta
            elif isinstance(val, str):
                # Categorical mutation for mode parameters
                if random.random() < 0.1 and "mode" in key:
                    options = ["diff", "ratio", "greater", "less"]
                    self.params[key] = random.choice(options)

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        """Override in subclasses."""
        raise NotImplementedError

    def reset_state(self):
        """Reset internal state to defaults."""
        for key in self.state:
            if isinstance(self.state[key], (int, float)):
                if key == "running_max":
                    self.state[key] = 1.0
                elif key == "count":
                    self.state[key] = 0
                else:
                    self.state[key] = 0.0
            elif isinstance(self.state[key], bool):
                self.state[key] = False
            elif self.state[key] is None:
                self.state[key] = None

    def to_dict(self) -> dict:
        """Serialize protein to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'params': dict(self.params),
            'inputs': list(self.inputs)
        }


# ================================================================
# 1. SENSOR PROTEIN
# Reads a single environment signal with normalization.
# ================================================================
class SensorProtein(Protein):
    def __init__(self, signal_name: str):
        super().__init__(signal_name, "sensor")
        self.params["decay"] = 0.999

        self.state["running_max"] = 1.0
        self.state["count"] = 0

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        raw = signals.get(self.name, 0.0)

        # Distance signals need absolute values preserved
        if "dist" in self.name.lower() or self.name == "dist_to_food":
            max_distance = 18.0  # Maximum Manhattan distance on 10x10 grid
            self.output = raw / max_distance
            self.output = max(min(self.output, 2.0), 0.0)
            return self.output

        # Adaptive normalization for other signals
        self.state["running_max"] = max(
            self.params["decay"] * self.state["running_max"],
            abs(raw),
            1.0
        )
        self.state["count"] += 1

        self.output = raw / self.state["running_max"]
        self.output = max(min(self.output, 5.0), -5.0)

        return self.output


# ================================================================
# 2. COMPARATOR PROTEIN
# Compares two inputs: difference, ratio, or threshold test.
# ================================================================
class ComparatorProtein(Protein):
    def __init__(self, name: str = "comparator"):
        super().__init__(name, "comparator")
        self.params.update({
            "mode": "diff",  # diff | ratio | greater | less
            "threshold": 0.0,
        })

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        def resolve(x):
            return signals.get(x, protein_outputs.get(x, 0.0))

        a = resolve(self.inputs[0])
        b = resolve(self.inputs[1])

        mode = self.params["mode"]

        if mode == "diff":
            self.output = a - b
        elif mode == "ratio":
            self.output = a / (b + 1e-6)
        elif mode == "greater":
            self.output = 1.0 if a > b else -1.0
        elif mode == "less":
            self.output = 1.0 if a < b else -1.0
        else:
            self.output = a - b

        return self.output


# ================================================================
# 3. TREND PROTEIN
# Detects direction and momentum of change.
# ================================================================
class TrendProtein(Protein):
    def __init__(self, name: str):
        super().__init__(name, "trend")
        self.params["momentum"] = 0.9

        self.state["last"] = None
        self.state["velocity"] = 0.0

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 1:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        last = self.state["last"]
        if last is None:
            self.state["last"] = x
            self.output = 0.0
            return 0.0

        delta = x - last
        self.state["last"] = x

        # Momentum-smoothed delta (EMA)
        self.state["velocity"] = (
            self.params["momentum"] * self.state["velocity"]
            + (1 - self.params["momentum"]) * delta
        )

        self.output = self.state["velocity"]
        return self.output


# ================================================================
# 4. INTEGRATOR PROTEIN
# Rolling accumulation with decay.
# ================================================================
class IntegratorProtein(Protein):
    def __init__(self, name: str):
        super().__init__(name, "integrator")
        self.params["decay"] = 0.05

        self.state["accum"] = 0.0

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 1:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        self.state["accum"] = self.state["accum"] * (1 - self.params["decay"]) + x
        self.output = max(min(self.state["accum"], 10.0), -10.0)

        return self.output


# ================================================================
# 5. GATE PROTEIN
# Conditional activation based on threshold with hysteresis.
# ================================================================
class GateProtein(Protein):
    def __init__(self, name: str):
        super().__init__(name, "gate")
        self.params["threshold"] = 0.0
        self.params["hysteresis"] = 0.1

        self.state["active"] = False

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        condition = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))
        value = signals.get(self.inputs[1], protein_outputs.get(self.inputs[1], 0.0))

        if not self.state["active"] and condition > (self.params["threshold"] + self.params["hysteresis"]):
            self.state["active"] = True
        elif self.state["active"] and condition < (self.params["threshold"] - self.params["hysteresis"]):
            self.state["active"] = False

        if self.state["active"]:
            self.output = 0.5 * (condition + value)
        else:
            self.output = 0.0

        return self.output


# ================================================================
# 6. TRUST MODIFIER PROTEIN
# Converts protein output to trust delta.
# ================================================================
class TrustModifierProtein(Protein):
    def __init__(self, name: str = "trust_mod"):
        super().__init__(name, "trust_modifier")
        self.params["gain"] = 1.0
        self.params["scale"] = 1.0
        self.params["decay"] = 0.999

        self.state["running"] = 0.0
        self.trust_output = 0.0

    def forward(self, signals: Dict[str, float], protein_outputs: Dict[str, float]) -> float:
        if len(self.inputs) < 1:
            self.trust_output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        # Rolling influence
        self.state["running"] = (
            self.params["decay"] * self.state["running"]
            + (1 - self.params["decay"]) * x
        )

        # Compute trust output
        raw_trust = self.params["gain"] * self.params["scale"] * self.state["running"]
        self.trust_output = raw_trust
        self.output = self.trust_output

        return self.trust_output


# ================================================================
# PROTEIN CASCADE HELPER
# ================================================================
def run_protein_cascade(proteins: List[Protein], signals: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Run forward pass through all proteins in order.
    Returns (outputs_dict, trust_delta).
    """
    outputs = {}

    for p in proteins:
        signal = p.forward(signals, outputs)
        outputs[p.name] = signal
        p.output = signal

    # Collect trust contributions from TrustModifierProteins
    trust_delta = sum(
        p.trust_output
        for p in proteins
        if isinstance(p, TrustModifierProtein)
    )

    # Clamp final trust delta
    trust_delta = max(min(trust_delta, 5.0), -5.0)

    return outputs, trust_delta


# ================================================================
# NEURAL CONTROLLER
# 2-layer feedforward network for action selection.
# ================================================================
class NeuralController:
    """Forward-pass-only neural network for action selection."""

    def __init__(self, input_size: int = 11, hidden_size: int = 32, output_size: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights randomly
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.b2 = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

        # Store activations for visualization
        self.hidden_activations: List[float] = []
        self.output_activations: List[float] = []

    def clone(self) -> 'NeuralController':
        """Deep copy controller."""
        new = NeuralController(self.input_size, self.hidden_size, self.output_size)
        new.w1 = copy.deepcopy(self.w1)
        new.w2 = copy.deepcopy(self.w2)
        new.b1 = copy.deepcopy(self.b1)
        new.b2 = copy.deepcopy(self.b2)
        return new

    def mutate(self, rate: float = 0.05, scale: float = 0.1):
        """Gaussian mutation across all weights."""
        def mutate_matrix(mat):
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if random.random() < rate:
                        mat[i][j] += random.gauss(0, scale)

        def mutate_vector(vec):
            for i in range(len(vec)):
                if random.random() < rate:
                    vec[i] += random.gauss(0, scale)

        mutate_matrix(self.w1)
        mutate_matrix(self.w2)
        mutate_vector(self.b1)
        mutate_vector(self.b2)

    def forward(self, inputs: List[float]) -> int:
        """
        Forward pass through network.
        Returns action index (0-3).
        """
        # Hidden layer with tanh activation
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(self.input_size):
                if j < len(inputs):
                    s += self.w1[i][j] * inputs[j]
            hidden.append(math.tanh(s))

        self.hidden_activations = hidden

        # Output layer (linear)
        outputs = []
        for i in range(self.output_size):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        self.output_activations = outputs

        # Return action with max activation
        return max(range(self.output_size), key=lambda i: outputs[i])

    def to_dict(self) -> dict:
        """Serialize controller to dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NeuralController':
        """Deserialize controller from dictionary."""
        controller = cls(data['input_size'], data['hidden_size'], data['output_size'])
        controller.w1 = data['w1']
        controller.b1 = data['b1']
        controller.w2 = data['w2']
        controller.b2 = data['b2']
        return controller


# ================================================================
# DEFAULT PROTEIN TEMPLATE
# ================================================================
def build_protein_template() -> List[Protein]:
    """
    Build the default protein cascade for a new genome.
    This creates a reasonable starting configuration.
    """
    proteins = []

    # Signal names for 11 inputs
    signal_names = [
        "steps_alive", "energy", "dist_to_food",
        "head_x", "head_y", "food_x", "food_y",
        "food_dx", "food_dy", "near_wall", "alive"
    ]

    # 1. Sensor proteins for each input signal
    for name in signal_names:
        proteins.append(SensorProtein(name))

    # 2. Trend protein for distance changes
    trend_dist = TrendProtein("trend_dist")
    trend_dist.bind_inputs(["dist_to_food"])
    proteins.append(trend_dist)

    # 3. Comparator for food direction
    comp_food = ComparatorProtein("comp_food_dir")
    comp_food.bind_inputs(["food_dx", "food_dy"])
    comp_food.params["mode"] = "diff"
    proteins.append(comp_food)

    # 4. Integrator for survival time
    integ_alive = IntegratorProtein("integ_alive")
    integ_alive.bind_inputs(["steps_alive"])
    proteins.append(integ_alive)

    # 5. Gate for danger avoidance
    gate_danger = GateProtein("gate_danger")
    gate_danger.bind_inputs(["near_wall", "dist_to_food"])
    gate_danger.params["threshold"] = 0.5
    proteins.append(gate_danger)

    # 6. Trust modifiers
    trust_dist = TrustModifierProtein("trust_dist")
    trust_dist.bind_inputs(["trend_dist"])
    trust_dist.params["gain"] = 2.0
    trust_dist.params["scale"] = -1.0  # Negative = reward getting closer
    proteins.append(trust_dist)

    trust_alive = TrustModifierProtein("trust_alive")
    trust_alive.bind_inputs(["alive"])
    trust_alive.params["gain"] = 0.5
    trust_alive.params["scale"] = 1.0
    proteins.append(trust_alive)

    trust_energy = TrustModifierProtein("trust_energy")
    trust_energy.bind_inputs(["energy"])
    trust_energy.params["gain"] = 0.1
    trust_energy.params["scale"] = 1.0
    proteins.append(trust_energy)

    return proteins


# ================================================================
# GENOME CLASS - DUAL ARCHITECTURE
# ================================================================
@dataclass
class Genome:
    """
    Complete genome with dual-path architecture:
    - proteins: Regulatory cascade for TRUST computation
    - controller: Neural network for ACTION selection
    """
    genome_id: str
    proteins: List[Protein] = field(default_factory=list)
    controller: Optional[NeuralController] = None
    generation_born: int = 0
    parent_ids: List[str] = field(default_factory=list)

    @classmethod
    def create_random(cls, generation: int = 0) -> 'Genome':
        """Create a new genome with random initialization."""
        genome = cls(
            genome_id=str(uuid.uuid4())[:8],
            generation_born=generation
        )

        # Build protein cascade from template
        genome.proteins = build_protein_template()

        # Create neural controller (11 inputs, 32 hidden, 4 outputs)
        genome.controller = NeuralController(input_size=11, hidden_size=32, output_size=4)

        return genome

    def clone(self) -> 'Genome':
        """Create a deep copy of this genome."""
        new_genome = Genome(
            genome_id=str(uuid.uuid4())[:8],
            generation_born=self.generation_born + 1,
            parent_ids=[self.genome_id]
        )

        # Deep copy proteins
        new_genome.proteins = [copy.deepcopy(p) for p in self.proteins]

        # Reset protein states
        for p in new_genome.proteins:
            p.reset_state()

        # Clone controller
        new_genome.controller = self.controller.clone() if self.controller else None

        return new_genome

    def mutate(self, rate: float = 0.05, scale: float = 0.2) -> 'Genome':
        """Apply mutations to proteins and controller."""
        # Mutate protein parameters
        for p in self.proteins:
            for k in p.params:
                if random.random() < rate:
                    p.mutate_param(k, scale=scale)

                    # Apply bounds
                    if k == "scale":
                        p.params[k] = max(min(p.params[k], 5.0), -5.0)
                    elif k == "gain":
                        p.params[k] = max(min(p.params[k], 10.0), 0.1)
                    elif k == "decay":
                        p.params[k] = max(min(p.params[k], 0.999), 0.0)
                    elif k == "threshold":
                        p.params[k] = max(min(p.params[k], 10.0), -10.0)
                    elif k == "momentum":
                        p.params[k] = max(min(p.params[k], 0.99), 0.0)

        # Mutate controller weights
        if self.controller:
            self.controller.mutate(rate=rate, scale=scale)

        return self

    @classmethod
    def crossbreed(cls, parent1: 'Genome', parent2: 'Genome', generation: int) -> 'Genome':
        """Create offspring by combining two parent genomes."""
        child = cls(
            genome_id=str(uuid.uuid4())[:8],
            generation_born=generation,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )

        # Deep copy proteins from parent1 as base
        child.proteins = [copy.deepcopy(p) for p in parent1.proteins]

        # Average protein parameters from both parents
        for i, protein in enumerate(child.proteins):
            if i < len(parent2.proteins):
                for param_name in protein.params:
                    if param_name in parent2.proteins[i].params:
                        p1_val = parent1.proteins[i].params[param_name]
                        p2_val = parent2.proteins[i].params[param_name]
                        if isinstance(p1_val, (int, float)) and isinstance(p2_val, (int, float)):
                            protein.params[param_name] = (p1_val + p2_val) / 2.0

        # Reset protein states
        for p in child.proteins:
            p.reset_state()

        # Mix controller weights
        if parent1.controller and parent2.controller:
            child.controller = parent1.controller.clone()

            # Randomly swap weights from parent2
            for i in range(len(child.controller.w1)):
                for j in range(len(child.controller.w1[i])):
                    if random.random() < 0.5:
                        child.controller.w1[i][j] = parent2.controller.w1[i][j]

            for i in range(len(child.controller.b1)):
                if random.random() < 0.5:
                    child.controller.b1[i] = parent2.controller.b1[i]

            for i in range(len(child.controller.w2)):
                for j in range(len(child.controller.w2[i])):
                    if random.random() < 0.5:
                        child.controller.w2[i][j] = parent2.controller.w2[i][j]

            for i in range(len(child.controller.b2)):
                if random.random() < 0.5:
                    child.controller.b2[i] = parent2.controller.b2[i]
        else:
            child.controller = parent1.controller.clone() if parent1.controller else NeuralController()

        return child

    def forward(self, signals: Dict[str, float]) -> Tuple[Dict[str, float], float, int]:
        """
        Run full forward pass:
        1. Proteins compute trust delta
        2. Controller computes action
        Returns (protein_outputs, trust_delta, action)
        """
        # Run protein cascade for trust
        outputs, trust_delta = run_protein_cascade(self.proteins, signals)

        # Run controller for action
        signal_list = [
            signals.get("steps_alive", 0.0),
            signals.get("energy", 0.0),
            signals.get("dist_to_food", 0.0),
            signals.get("head_x", 0.0),
            signals.get("head_y", 0.0),
            signals.get("food_x", 0.0),
            signals.get("food_y", 0.0),
            signals.get("food_dx", 0.0),
            signals.get("food_dy", 0.0),
            signals.get("near_wall", 0.0),
            signals.get("alive", 1.0),
        ]

        action = self.controller.forward(signal_list) if self.controller else 0

        return outputs, trust_delta, action

    def to_dict(self) -> dict:
        """Convert genome to dictionary for JSON serialization."""
        return {
            'genome_id': self.genome_id,
            'generation_born': self.generation_born,
            'parent_ids': self.parent_ids,
            'proteins': [p.to_dict() for p in self.proteins],
            'controller': self.controller.to_dict() if self.controller else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        """Deserialize genome from dictionary."""
        genome = cls(
            genome_id=data['genome_id'],
            generation_born=data.get('generation_born', 0),
            parent_ids=data.get('parent_ids', [])
        )

        # Rebuild proteins from data
        genome.proteins = []
        for p_data in data.get('proteins', []):
            p_type = p_data['type']
            p_name = p_data['name']

            if p_type == 'sensor':
                p = SensorProtein(p_name)
            elif p_type == 'comparator':
                p = ComparatorProtein(p_name)
            elif p_type == 'trend':
                p = TrendProtein(p_name)
            elif p_type == 'integrator':
                p = IntegratorProtein(p_name)
            elif p_type == 'gate':
                p = GateProtein(p_name)
            elif p_type == 'trust_modifier':
                p = TrustModifierProtein(p_name)
            else:
                continue

            p.params = p_data.get('params', {})
            p.inputs = p_data.get('inputs', [])
            genome.proteins.append(p)

        # Rebuild controller
        if data.get('controller'):
            genome.controller = NeuralController.from_dict(data['controller'])

        return genome


# ================================================================
# ORGANISM CLASS
# ================================================================
@dataclass
class Organism:
    """An organism with genome and fitness tracking."""
    organism_id: str
    genome: Genome
    trust_score: float = 0.0
    cumulative_trust: float = 0.0
    games_played: int = 0
    total_food_eaten: int = 0
    total_moves: int = 0
    best_game_score: int = 0
    alive: bool = True
    trust_history: List[float] = field(default_factory=list)
    stability: float = 0.0

    def update_trust(self, delta: float):
        """Add trust delta to cumulative trust."""
        self.cumulative_trust += delta
        self.cumulative_trust = max(min(self.cumulative_trust, 100000.0), -100000.0)

    def reset_for_game(self):
        """Reset state for a new game."""
        # Reset protein states
        for p in self.genome.proteins:
            p.reset_state()

    def update_stability(self, window_size: int = 5):
        """Calculate stability as inverse of trust variance."""
        self.trust_history.append(self.cumulative_trust)

        if len(self.trust_history) > window_size:
            self.trust_history.pop(0)

        if len(self.trust_history) < 2:
            self.stability = 0.0
            return

        mean_trust = sum(self.trust_history) / len(self.trust_history)
        variance = sum((t - mean_trust) ** 2 for t in self.trust_history) / len(self.trust_history)
        self.stability = 1.0 / (variance + 1.0)

    def calculate_fitness(self) -> float:
        """
        Calculate fitness combining trust (70%) and stability (30%).
        """
        if self.games_played == 0:
            return 0.0

        # Combine trust and stability
        self.trust_score = 0.7 * self.cumulative_trust + 0.3 * self.stability
        return self.trust_score


# ================================================================
# EVENT AND SNAPSHOT STRUCTURES
# ================================================================
@dataclass
class GeneticEvent:
    """Record of a genetic event during evolution."""
    event_id: str
    event_type: GeneticEventType
    generation: int
    timestamp: float
    organism_ids: List[str]
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'generation': self.generation,
            'timestamp': self.timestamp,
            'organism_ids': self.organism_ids,
            'details': self.details
        }


@dataclass
class GenerationSnapshot:
    """Snapshot of population state at a generation."""
    generation: int
    organisms: List[Dict]
    best_trust: float
    avg_trust: float
    worst_trust: float
    genetic_events: List[Dict]
    stage: str  # 'early', 'mid', 'late'

    def to_dict(self) -> dict:
        return {
            'generation': self.generation,
            'organisms': self.organisms,
            'best_trust': self.best_trust,
            'avg_trust': self.avg_trust,
            'worst_trust': self.worst_trust,
            'genetic_events': self.genetic_events,
            'stage': self.stage
        }


@dataclass
class GameFrame:
    """Single frame of snake game state with dual-path data."""
    frame_num: int
    snake_positions: List[Tuple[int, int]]
    food_position: Tuple[int, int]
    direction: Direction
    score: int
    alive: bool
    energy: float

    # 11 input signals
    signals: Dict[str, float]

    # Protein cascade outputs
    protein_outputs: Dict[str, float]
    trust_delta: float
    cumulative_trust: float

    # Neural controller data
    nn_hidden_activations: List[float]
    nn_output_activations: List[float]
    selected_action: int

    def to_dict(self) -> dict:
        return {
            'frame_num': self.frame_num,
            'snake_positions': self.snake_positions,
            'food_position': self.food_position,
            'direction': self.direction.value,
            'score': self.score,
            'alive': self.alive,
            'energy': self.energy,
            'signals': self.signals,
            'protein_outputs': self.protein_outputs,
            'trust_delta': self.trust_delta,
            'cumulative_trust': self.cumulative_trust,
            'nn_hidden_activations': self.nn_hidden_activations,
            'nn_output_activations': self.nn_output_activations,
            'selected_action': self.selected_action
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GameFrame':
        return cls(
            frame_num=data['frame_num'],
            snake_positions=[tuple(p) for p in data['snake_positions']],
            food_position=tuple(data['food_position']),
            direction=Direction(data['direction']),
            score=data['score'],
            alive=data['alive'],
            energy=data.get('energy', 25.0),
            signals=data.get('signals', {}),
            protein_outputs=data.get('protein_outputs', {}),
            trust_delta=data.get('trust_delta', 0.0),
            cumulative_trust=data.get('cumulative_trust', 0.0),
            nn_hidden_activations=data.get('nn_hidden_activations', []),
            nn_output_activations=data.get('nn_output_activations', []),
            selected_action=data.get('selected_action', 0)
        )


@dataclass
class StageExemplar:
    """Pre-selected exemplar game for a specific evolution stage."""
    stage: str
    generation: int
    organism_id: str
    genome_snapshot: Dict
    game_frames: List[Dict]
    final_score: int
    description: str

    def to_dict(self) -> dict:
        return {
            'stage': self.stage,
            'generation': self.generation,
            'organism_id': self.organism_id,
            'genome_snapshot': self.genome_snapshot,
            'game_frames': self.game_frames,
            'final_score': self.final_score,
            'description': self.description
        }


# ================================================================
# EVOLUTION LOGGER
# ================================================================
class EvolutionLogger:
    """Logs all evolution data for visualization playback."""

    def __init__(self):
        self.generation_snapshots: List[GenerationSnapshot] = []
        self.genetic_events: List[GeneticEvent] = []
        self.stage_exemplars: Dict[str, StageExemplar] = {}
        self.protein_activation_log: List[Dict] = []
        self.config: Dict = {}

    def log_generation(self, snapshot: GenerationSnapshot):
        self.generation_snapshots.append(snapshot)

    def log_event(self, event: GeneticEvent):
        self.genetic_events.append(event)

    def log_exemplar(self, exemplar: StageExemplar):
        self.stage_exemplars[exemplar.stage] = exemplar

    def log_protein_activations(self, generation: int, frame: int,
                                organism_id: str, activations: Dict[str, float]):
        self.protein_activation_log.append({
            'generation': generation,
            'frame': frame,
            'organism_id': organism_id,
            'activations': activations
        })

    def save_all(self, output_dir: str):
        """Save all logs to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save evolution log
        with open(os.path.join(output_dir, 'evolution_log.json'), 'w') as f:
            json.dump({
                'generations': [s.to_dict() for s in self.generation_snapshots],
                'total_generations': len(self.generation_snapshots)
            }, f, indent=2)

        # Save genetic events
        with open(os.path.join(output_dir, 'genetic_events.json'), 'w') as f:
            json.dump({
                'events': [e.to_dict() for e in self.genetic_events]
            }, f, indent=2)

        # Save stage exemplars
        with open(os.path.join(output_dir, 'stage_exemplars.json'), 'w') as f:
            json.dump({
                stage: ex.to_dict() for stage, ex in self.stage_exemplars.items()
            }, f, indent=2)

        # Save protein activations (compressed)
        np.savez_compressed(
            os.path.join(output_dir, 'protein_activations.npz'),
            activations=self.protein_activation_log
        )

        # Save config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load_all(cls, input_dir: str) -> 'EvolutionLogger':
        """Load all logs from files."""
        import os
        logger = cls()

        # Load evolution log
        with open(os.path.join(input_dir, 'evolution_log.json'), 'r') as f:
            data = json.load(f)
            for gen_data in data['generations']:
                logger.generation_snapshots.append(GenerationSnapshot(
                    generation=gen_data['generation'],
                    organisms=gen_data['organisms'],
                    best_trust=gen_data['best_trust'],
                    avg_trust=gen_data['avg_trust'],
                    worst_trust=gen_data['worst_trust'],
                    genetic_events=gen_data['genetic_events'],
                    stage=gen_data['stage']
                ))

        # Load genetic events
        with open(os.path.join(input_dir, 'genetic_events.json'), 'r') as f:
            data = json.load(f)
            for event_data in data['events']:
                logger.genetic_events.append(GeneticEvent(
                    event_id=event_data['event_id'],
                    event_type=GeneticEventType(event_data['event_type']),
                    generation=event_data['generation'],
                    timestamp=event_data['timestamp'],
                    organism_ids=event_data['organism_ids'],
                    details=event_data['details']
                ))

        # Load stage exemplars
        with open(os.path.join(input_dir, 'stage_exemplars.json'), 'r') as f:
            data = json.load(f)
            for stage, ex_data in data.items():
                logger.stage_exemplars[stage] = StageExemplar(
                    stage=ex_data['stage'],
                    generation=ex_data['generation'],
                    organism_id=ex_data['organism_id'],
                    genome_snapshot=ex_data['genome_snapshot'],
                    game_frames=ex_data['game_frames'],
                    final_score=ex_data['final_score'],
                    description=ex_data['description']
                )

        # Load config
        with open(os.path.join(input_dir, 'config.json'), 'r') as f:
            logger.config = json.load(f)

        return logger
