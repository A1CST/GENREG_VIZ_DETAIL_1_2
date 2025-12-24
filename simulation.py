"""
GENREG Simulation Engine
Official architecture: Proteins compute TRUST, Neural Controller computes ACTION
10x10 grid with energy system and 11 signal inputs
"""

import numpy as np
import random
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data_structures import (
    Genome, Organism, Direction, GeneticEvent,
    GeneticEventType, GenerationSnapshot, GameFrame, StageExemplar,
    EvolutionLogger, run_protein_cascade
)
from config import SIMULATION_CONFIG


class SnakeGame:
    """Snake game environment - Official GENREG version (10x10 grid with energy)"""

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> Dict[str, float]:
        """Reset game to initial state, returns initial signals"""
        # Random starting position
        self.head_x = random.randint(0, self.grid_size - 1)
        self.head_y = random.randint(0, self.grid_size - 1)

        # Snake body (for visualization, not collision in minimal version)
        self.snake = [(self.head_x, self.head_y)]

        # Direction vector
        self.direction = Direction.UP
        self.direction_vec = (0, -1)

        # Food
        self._spawn_food()

        # Game state
        self.score = 0
        self.steps_alive = 0
        self.alive = True
        self.food_eaten = 0

        # Energy system
        self.max_energy = 25
        self.energy = self.max_energy

        self.last_death_reason = None

        return self.get_signals()

    def _spawn_food(self) -> None:
        """Spawn food at random empty location"""
        while True:
            fx = random.randint(0, self.grid_size - 1)
            fy = random.randint(0, self.grid_size - 1)
            # Don't spawn on head
            if fx != self.head_x or fy != self.head_y:
                break
        self.food_x = fx
        self.food_y = fy
        self.food = (fx, fy)

    def get_signals(self) -> Dict[str, float]:
        """Get the 11 official GENREG signals"""
        # Distance to food (manhattan)
        food_dx = self.food_x - self.head_x
        food_dy = self.food_y - self.head_y
        dist = abs(food_dx) + abs(food_dy)

        # Wall proximity (1.0 if at edge)
        near_wall_x = 1.0 if (self.head_x <= 0 or self.head_x >= self.grid_size - 1) else 0.0
        near_wall_y = 1.0 if (self.head_y <= 0 or self.head_y >= self.grid_size - 1) else 0.0
        near_wall = max(near_wall_x, near_wall_y)

        return {
            "steps_alive": float(self.steps_alive),
            "energy": float(self.energy),
            "dist_to_food": float(dist),
            "head_x": float(self.head_x),
            "head_y": float(self.head_y),
            "food_x": float(self.food_x),
            "food_y": float(self.food_y),
            "food_dx": float(food_dx),
            "food_dy": float(food_dy),
            "near_wall": near_wall,
            "alive": 1.0 if self.alive else 0.0,
        }

    def step(self, action: int) -> Tuple[Dict[str, float], bool]:
        """
        Execute one game step.
        action: 0=up, 1=down, 2=left, 3=right
        Returns (signals, done)
        """
        if not self.alive:
            return self.get_signals(), True

        # Update direction and move
        if action == 0:  # up
            self.direction = Direction.UP
            self.direction_vec = (0, -1)
            self.head_y -= 1
        elif action == 1:  # down
            self.direction = Direction.DOWN
            self.direction_vec = (0, 1)
            self.head_y += 1
        elif action == 2:  # left
            self.direction = Direction.LEFT
            self.direction_vec = (-1, 0)
            self.head_x -= 1
        elif action == 3:  # right
            self.direction = Direction.RIGHT
            self.direction_vec = (1, 0)
            self.head_x += 1

        self.steps_alive += 1
        self.energy -= 1

        # Update snake body for visualization
        self.snake.insert(0, (self.head_x, self.head_y))

        # Check food collision
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.score += 1
            self.food_eaten += 1
            self.max_energy += 2
            self.energy = self.max_energy
            self._spawn_food()
            # Don't pop tail - snake grows
        else:
            if len(self.snake) > 1:
                self.snake.pop()

        # Check wall collision
        if (self.head_x < 0 or self.head_x >= self.grid_size or
                self.head_y < 0 or self.head_y >= self.grid_size):
            self.alive = False
            self.last_death_reason = "wall collision"
            return self.get_signals(), True

        # Check energy depletion
        if self.energy <= 0:
            self.alive = False
            self.last_death_reason = "energy depletion"
            return self.get_signals(), True

        self.last_death_reason = None
        return self.get_signals(), False

    def get_state(self) -> Dict:
        """Get current game state for visualization"""
        return {
            'snake': list(self.snake),
            'food': self.food,
            'direction': self.direction,
            'score': self.score,
            'alive': self.alive,
            'energy': self.energy,
            'steps_alive': self.steps_alive
        }


class GENREGSimulation:
    """Main simulation runner with official GENREG evolutionary algorithm"""

    def __init__(self, config: Dict = None):
        self.config = config or SIMULATION_CONFIG
        self.population: List[Organism] = []
        self.generation = 0
        self.logger = EvolutionLogger()
        self.logger.config = dict(self.config)

    def initialize_population(self) -> None:
        """Create initial random population"""
        self.population = []
        for _ in range(self.config['population_size']):
            genome = Genome.create_random(generation=0)
            organism = Organism(
                organism_id=str(uuid.uuid4())[:8],
                genome=genome
            )
            self.population.append(organism)

            # Log birth event
            self.logger.log_event(GeneticEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type=GeneticEventType.BIRTH,
                generation=0,
                timestamp=0.0,
                organism_ids=[organism.organism_id],
                details={'genome_id': genome.genome_id}
            ))

    def evaluate_organism(self, organism: Organism,
                          record_frames: bool = False) -> Tuple[float, List[GameFrame]]:
        """
        Evaluate organism fitness through Snake gameplay.
        Uses dual-path: proteins for trust, controller for action.
        """
        total_score = 0
        total_moves = 0
        best_score = 0
        all_frames = []

        # Reset organism for evaluation
        organism.cumulative_trust = 0.0
        organism.reset_for_game()

        for game_num in range(self.config['games_per_evaluation']):
            game = SnakeGame(self.config['grid_size'])
            signals = game.reset()
            frame_num = 0
            cumulative_trust = 0.0

            while game.alive:
                # Run genome forward pass (dual-path)
                protein_outputs, trust_delta, action = organism.genome.forward(signals)

                # Update trust
                cumulative_trust += trust_delta
                organism.update_trust(trust_delta)

                # Record frame if needed
                if record_frames:
                    nn_hidden = []
                    nn_output = []
                    if organism.genome.controller:
                        nn_hidden = list(organism.genome.controller.hidden_activations)
                        nn_output = list(organism.genome.controller.output_activations)

                    frame = GameFrame(
                        frame_num=frame_num,
                        snake_positions=list(game.snake),
                        food_position=game.food,
                        direction=game.direction,
                        score=game.score,
                        alive=game.alive,
                        energy=game.energy,
                        signals=dict(signals),
                        protein_outputs=dict(protein_outputs),
                        trust_delta=trust_delta,
                        cumulative_trust=cumulative_trust,
                        nn_hidden_activations=nn_hidden,
                        nn_output_activations=nn_output,
                        selected_action=action
                    )
                    all_frames.append(frame)

                # Execute action
                signals, done = game.step(action)
                frame_num += 1

                # Safety limit
                if frame_num > 500:
                    break

            total_score += game.score
            total_moves += game.steps_alive
            best_score = max(best_score, game.score)

        # Update organism stats
        organism.games_played += self.config['games_per_evaluation']
        organism.total_food_eaten += total_score
        organism.total_moves += total_moves
        organism.best_game_score = max(organism.best_game_score, best_score)

        fitness = organism.calculate_fitness()
        return fitness, all_frames

    def select_and_reproduce(self) -> List[GeneticEvent]:
        """
        Selection and reproduction using official GENREG algorithm:
        - 10% elite survive
        - 40% crossover
        - 50% asexual with mutation
        - 5% trust inheritance
        """
        events = []

        # Update stability for all organisms
        for organism in self.population:
            organism.update_stability()

        # Sort by trust score (highest first)
        self.population.sort(key=lambda o: o.trust_score, reverse=True)

        # Calculate survival cutoff (top 20% survive for breeding)
        elite_rate = self.config.get('elite_rate', 0.10)
        crossover_rate = self.config.get('crossover_rate', 0.40)

        elite_count = max(1, int(len(self.population) * elite_rate))
        survival_count = max(1, int(len(self.population) * 0.20))

        survivors = self.population[:survival_count]
        elite = self.population[:elite_count]

        # Log cull events for non-survivors
        culled = self.population[survival_count:]
        for i, organism in enumerate(culled):
            organism.alive = False
            events.append(GeneticEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type=GeneticEventType.CULL,
                generation=self.generation,
                timestamp=0.3 + i * 0.01,
                organism_ids=[organism.organism_id],
                details={'trust_score': organism.trust_score}
            ))

        # Log elite preservation
        for organism in elite:
            events.append(GeneticEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type=GeneticEventType.ELITE_PRESERVE,
                generation=self.generation,
                timestamp=0.4,
                organism_ids=[organism.organism_id],
                details={'trust_score': organism.trust_score}
            ))

        # Calculate fitness weights combining trust (70%) and stability (30%)
        min_trust = min(o.trust_score for o in survivors)
        max_stability = max(o.stability for o in survivors) + 1e-6

        fitness_weights = [
            0.7 * (o.trust_score - min_trust + 1.0) + 0.3 * (o.stability / max_stability)
            for o in survivors
        ]

        # Create new population
        new_population = []

        for i in range(self.config['population_size']):
            # Determine reproduction method
            rand = random.random()

            if rand < crossover_rate:
                # Crossbreed (40%)
                parent1, parent2 = random.choices(survivors, weights=fitness_weights, k=2)

                child_genome = Genome.crossbreed(
                    parent1.genome, parent2.genome, self.generation + 1
                )

                # Apply mutation
                mutation_rate = self.config.get('mutation_rate', 0.05)
                mutation_scale = self.config.get('mutation_scale', 0.2)
                child_genome.mutate(rate=mutation_rate, scale=mutation_scale)

                child = Organism(
                    organism_id=str(uuid.uuid4())[:8],
                    genome=child_genome
                )

                # Inherit 5% of average parent trust
                trust_inheritance = self.config.get('trust_inheritance', 0.05)
                child.cumulative_trust = (parent1.cumulative_trust + parent2.cumulative_trust) * trust_inheritance

                new_population.append(child)

                events.append(GeneticEvent(
                    event_id=str(uuid.uuid4())[:8],
                    event_type=GeneticEventType.CROSSBREED,
                    generation=self.generation,
                    timestamp=0.5 + len(new_population) * 0.01,
                    organism_ids=[parent1.organism_id, parent2.organism_id, child.organism_id],
                    details={'child_genome_id': child_genome.genome_id}
                ))
            else:
                # Asexual reproduction with mutation (50%)
                parent = random.choices(survivors, weights=fitness_weights, k=1)[0]

                child_genome = parent.genome.clone()
                mutation_rate = self.config.get('mutation_rate', 0.05)
                mutation_scale = self.config.get('mutation_scale', 0.2)
                child_genome.mutate(rate=mutation_rate, scale=mutation_scale)

                child = Organism(
                    organism_id=str(uuid.uuid4())[:8],
                    genome=child_genome
                )

                # Inherit 5% of parent trust
                trust_inheritance = self.config.get('trust_inheritance', 0.05)
                child.cumulative_trust = parent.cumulative_trust * trust_inheritance

                new_population.append(child)

                events.append(GeneticEvent(
                    event_id=str(uuid.uuid4())[:8],
                    event_type=GeneticEventType.MUTATION,
                    generation=self.generation,
                    timestamp=0.5 + len(new_population) * 0.01,
                    organism_ids=[parent.organism_id, child.organism_id],
                    details={'child_genome_id': child_genome.genome_id}
                ))

        self.population = new_population
        return events

    def get_stage(self, generation: int) -> str:
        """Determine evolution stage based on generation"""
        total = self.config['total_generations']
        if generation < total * 0.2:
            return 'early'
        elif generation < total * 0.6:
            return 'mid'
        else:
            return 'late'

    def run_generation(self) -> GenerationSnapshot:
        """Run one complete generation"""
        # Evaluate all organisms
        for organism in self.population:
            self.evaluate_organism(organism, record_frames=False)

        # Create snapshot before selection
        trust_scores = [o.trust_score for o in self.population]
        stage = self.get_stage(self.generation)

        # Selection and reproduction
        events = self.select_and_reproduce()

        # Log events
        for event in events:
            self.logger.log_event(event)

        # Create generation snapshot
        snapshot = GenerationSnapshot(
            generation=self.generation,
            organisms=[{
                'organism_id': o.organism_id,
                'genome_id': o.genome.genome_id,
                'trust_score': o.trust_score,
                'cumulative_trust': o.cumulative_trust,
                'best_score': o.best_game_score,
                'stability': o.stability,
                'alive': o.alive
            } for o in self.population[:10]],  # Top 10 for display
            best_trust=max(trust_scores) if trust_scores else 0,
            avg_trust=np.mean(trust_scores) if trust_scores else 0,
            worst_trust=min(trust_scores) if trust_scores else 0,
            genetic_events=[e.to_dict() for e in events],
            stage=stage
        )

        self.logger.log_generation(snapshot)
        self.generation += 1

        return snapshot

    def capture_stage_exemplar(self, stage: str) -> StageExemplar:
        """Capture an exemplar game for a specific stage"""
        # Find best organism
        best_organism = max(self.population, key=lambda o: o.trust_score)

        # Run a recorded game
        _, frames = self.evaluate_organism(best_organism, record_frames=True)

        # Take subset of frames for the exemplar
        max_frames = min(200, len(frames))
        selected_frames = frames[:max_frames]

        descriptions = {
            'early': "Random exploration, frequent wall collisions, no food-seeking behavior",
            'mid': "Basic food-seeking behavior emerging, learning to avoid walls",
            'late': "Intelligent navigation, efficient food collection, stable behavior"
        }

        exemplar = StageExemplar(
            stage=stage,
            generation=self.generation,
            organism_id=best_organism.organism_id,
            genome_snapshot=best_organism.genome.to_dict(),
            game_frames=[f.to_dict() for f in selected_frames],
            final_score=best_organism.best_game_score,
            description=descriptions.get(stage, "Evolution in progress")
        )

        self.logger.log_exemplar(exemplar)
        return exemplar

    def run_full_simulation(self, progress_callback=None) -> str:
        """Run complete evolutionary simulation"""
        print("Initializing population...")
        self.initialize_population()

        # Capture early stage exemplar
        print("Capturing early stage exemplar...")
        for _ in range(5):  # Run a few generations first
            self.run_generation()
        self.capture_stage_exemplar('early')

        total_gens = self.config['total_generations']
        mid_point = int(total_gens * 0.4)
        late_point = int(total_gens * 0.8)

        print(f"Running {total_gens} generations...")
        for gen in range(self.generation, total_gens):
            snapshot = self.run_generation()

            if progress_callback:
                progress_callback(gen, total_gens, snapshot)

            if gen % 50 == 0:
                print(f"Generation {gen}: Best={snapshot.best_trust:.2f}, "
                      f"Avg={snapshot.avg_trust:.2f}, Stage={snapshot.stage}")

            # Capture mid and late exemplars
            if gen == mid_point:
                print("Capturing mid stage exemplar...")
                self.capture_stage_exemplar('mid')
            elif gen == late_point:
                print("Capturing late stage exemplar...")
                self.capture_stage_exemplar('late')

        # Save all data
        output_dir = "simulation_data"
        print(f"Saving simulation data to {output_dir}...")
        self.logger.save_all(output_dir)

        return output_dir


def main():
    """Run pre-computation"""
    print("=" * 60)
    print("GENREG Evolution Pre-Computation (Official Architecture)")
    print("=" * 60)

    sim = GENREGSimulation()
    output_dir = sim.run_full_simulation()

    print("=" * 60)
    print(f"Pre-computation complete! Data saved to: {output_dir}")
    print("Run the visualizer to view the evolution.")
    print("=" * 60)


if __name__ == "__main__":
    main()
