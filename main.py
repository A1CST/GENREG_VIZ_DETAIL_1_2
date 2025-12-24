#!/usr/bin/env python3
"""
GENREG Evolution Visualizer
Main entry point

Usage:
    python main.py              # Interactive menu
    python main.py --simulate   # Force re-run simulation
    python main.py --visualize  # Only run visualizer (requires existing data)
    python main.py --checkpoint # Load checkpoint directly (skips menu)
"""

import sys
import os
import glob
import pickle
import argparse


def find_checkpoints():
    """Find checkpoint files in the parent directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Look for checkpoint files
    checkpoint_pattern = os.path.join(parent_dir, "checkpoint_*.pkl")
    checkpoints = glob.glob(checkpoint_pattern)

    # Sort by generation number (extract from filename)
    def get_gen_number(path):
        filename = os.path.basename(path)
        try:
            # Format: checkpoint_gen_XXXXX.pkl
            gen_str = filename.replace("checkpoint_gen_", "").replace(".pkl", "")
            return int(gen_str)
        except:
            return 0

    checkpoints.sort(key=get_gen_number, reverse=True)
    return checkpoints


def load_best_genome_from_checkpoint(checkpoint_path: str):
    """
    Load the best genome from a checkpoint file.
    Returns the best genome or None if loading fails.
    """
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)

        # Handle different checkpoint formats
        if isinstance(data, dict):
            # Dict format - look for genomes or population
            if 'genomes' in data:
                genomes = data['genomes']
            elif 'population' in data:
                genomes = data['population']
            elif 'best_genome' in data:
                return data['best_genome']
            else:
                print(f"Unknown checkpoint dict format. Keys: {data.keys()}")
                return None
        elif isinstance(data, list):
            # List of genomes
            genomes = data
        elif hasattr(data, 'genomes'):
            # Population object
            genomes = data.genomes
        else:
            print(f"Unknown checkpoint format: {type(data)}")
            return None

        if not genomes:
            print("No genomes found in checkpoint")
            return None

        # Find best genome by trust score
        best_genome = max(genomes, key=lambda g: getattr(g, 'trust', 0))

        print(f"Loaded best genome with trust: {getattr(best_genome, 'trust', 'N/A')}")
        return best_genome

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def run_checkpoint_demo(genome):
    """Run a live demo with the loaded genome"""
    from simulation import SnakeGame
    from data_structures import GameFrame, Direction, StageExemplar, EvolutionLogger, Genome
    from config import SIMULATION_CONFIG
    import json

    print("\n" + "=" * 60)
    print("Running Demo with Loaded Genome")
    print("=" * 60)

    # Convert external genome to our format if needed
    our_genome = convert_external_genome(genome)

    if our_genome is None:
        print("Failed to convert genome. Running with random genome instead.")
        our_genome = Genome.create_random()

    # Run several games and record frames
    all_frames = []
    best_score = 0
    total_cumulative_trust = 0.0  # Track total trust across all games
    game_trusts = []  # Track trust per game

    print("Recording gameplay...")

    for game_num in range(3):
        game = SnakeGame(SIMULATION_CONFIG['grid_size'])
        signals = game.reset()
        frame_num = 0
        game_cumulative_trust = 0.0

        # Reset protein states
        for p in our_genome.proteins:
            p.reset_state()

        while game.alive and frame_num < 500:
            # Run forward pass
            protein_outputs, trust_delta, action = our_genome.forward(signals)
            game_cumulative_trust += trust_delta
            total_cumulative_trust += trust_delta

            # Get NN activations
            nn_hidden = list(our_genome.controller.hidden_activations) if our_genome.controller else []
            nn_output = list(our_genome.controller.output_activations) if our_genome.controller else []

            # Record frame
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
                cumulative_trust=total_cumulative_trust,  # Use total cumulative
                nn_hidden_activations=nn_hidden,
                nn_output_activations=nn_output,
                selected_action=action
            )
            all_frames.append(frame)

            # Step game
            signals, done = game.step(action)
            frame_num += 1

        best_score = max(best_score, game.score)
        game_trusts.append(game_cumulative_trust)
        print(f"  Game {game_num + 1}: Score = {game.score}, Frames = {frame_num}, Trust = {game_cumulative_trust:.2f}")

    # Use the original genome trust if available, otherwise use calculated trust
    original_trust = getattr(genome, 'trust', None)
    display_trust = original_trust if original_trust is not None and original_trust != 0 else total_cumulative_trust
    avg_game_trust = sum(game_trusts) / len(game_trusts) if game_trusts else 0

    print(f"\nBest score: {best_score}")
    print(f"Total frames recorded: {len(all_frames)}")
    print(f"Total trust accumulated: {total_cumulative_trust:.2f}")
    print(f"Average trust per game: {avg_game_trust:.2f}")
    if original_trust:
        print(f"Original checkpoint trust: {original_trust:.2f}")

    # Create stage exemplar for visualization
    exemplar = StageExemplar(
        stage='checkpoint',
        generation=0,
        organism_id='checkpoint_best',
        genome_snapshot=our_genome.to_dict(),
        game_frames=[f.to_dict() for f in all_frames[:200]],
        final_score=best_score,
        description=f"Best genome from checkpoint (Trust: {display_trust:.1f}, Best Score: {best_score})"
    )

    # Save to temporary visualization data
    output_dir = "simulation_data"
    os.makedirs(output_dir, exist_ok=True)

    # Create minimal logger data
    logger = EvolutionLogger()
    logger.stage_exemplars['checkpoint'] = exemplar
    logger.stage_exemplars['early'] = exemplar
    logger.stage_exemplars['mid'] = exemplar
    logger.stage_exemplars['late'] = exemplar

    # Create generation snapshots with proper trust values
    from data_structures import GenerationSnapshot
    for i in range(10):
        snapshot = GenerationSnapshot(
            generation=i,
            organisms=[{
                'organism_id': 'checkpoint_best',
                'genome_id': our_genome.genome_id,
                'trust_score': display_trust,
                'cumulative_trust': total_cumulative_trust,
                'best_score': best_score,
                'alive': True
            }],
            best_trust=display_trust,
            avg_trust=avg_game_trust,
            worst_trust=avg_game_trust,
            genetic_events=[],
            stage='checkpoint'
        )
        logger.generation_snapshots.append(snapshot)

    logger.config = dict(SIMULATION_CONFIG)
    logger.save_all(output_dir)

    print(f"\nDemo data saved to {output_dir}")
    return output_dir


def convert_external_genome(external_genome):
    """Convert an external genome (from official training) to our Genome format"""
    from data_structures import (
        Genome, NeuralController,
        SensorProtein, ComparatorProtein, TrendProtein,
        IntegratorProtein, GateProtein, TrustModifierProtein
    )

    try:
        # Create new genome
        our_genome = Genome(
            genome_id='checkpoint_imported',
            generation_born=0
        )

        # Convert proteins
        our_genome.proteins = []

        if hasattr(external_genome, 'proteins'):
            for ext_protein in external_genome.proteins:
                ptype = type(ext_protein).__name__
                pname = getattr(ext_protein, 'name', 'unknown')

                # Create matching protein type
                if 'Sensor' in ptype:
                    p = SensorProtein(pname)
                elif 'Trend' in ptype:
                    p = TrendProtein(pname)
                    if hasattr(ext_protein, 'inputs'):
                        p.bind_inputs(ext_protein.inputs)
                elif 'Comparator' in ptype:
                    p = ComparatorProtein(pname)
                    if hasattr(ext_protein, 'inputs'):
                        p.bind_inputs(ext_protein.inputs)
                elif 'Integrator' in ptype:
                    p = IntegratorProtein(pname)
                    if hasattr(ext_protein, 'inputs'):
                        p.bind_inputs(ext_protein.inputs)
                elif 'Gate' in ptype:
                    p = GateProtein(pname)
                    if hasattr(ext_protein, 'inputs'):
                        p.bind_inputs(ext_protein.inputs)
                elif 'Trust' in ptype:
                    p = TrustModifierProtein(pname)
                    if hasattr(ext_protein, 'inputs'):
                        p.bind_inputs(ext_protein.inputs)
                else:
                    # Default to sensor
                    p = SensorProtein(pname)

                # Copy parameters
                if hasattr(ext_protein, 'params'):
                    for k, v in ext_protein.params.items():
                        if k in p.params:
                            p.params[k] = v

                our_genome.proteins.append(p)

        # Convert neural controller
        if hasattr(external_genome, 'controller'):
            ext_ctrl = external_genome.controller

            # Get dimensions
            input_size = len(ext_ctrl.w1[0]) if ext_ctrl.w1 else 11
            hidden_size = len(ext_ctrl.w1) if ext_ctrl.w1 else 32
            output_size = len(ext_ctrl.w2) if ext_ctrl.w2 else 4

            our_genome.controller = NeuralController(input_size, hidden_size, output_size)

            # Copy weights
            our_genome.controller.w1 = [list(row) for row in ext_ctrl.w1]
            our_genome.controller.b1 = list(ext_ctrl.b1)
            our_genome.controller.w2 = [list(row) for row in ext_ctrl.w2]
            our_genome.controller.b2 = list(ext_ctrl.b2)
        else:
            our_genome.controller = NeuralController()

        # If no proteins were converted, create default template
        if not our_genome.proteins:
            from data_structures import build_protein_template
            our_genome.proteins = build_protein_template()

        return our_genome

    except Exception as e:
        print(f"Error converting genome: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simulation():
    """Run the pre-computation simulation"""
    from simulation import GENREGSimulation

    print("=" * 60)
    print("GENREG Evolution Pre-Computation")
    print("=" * 60)
    print()

    sim = GENREGSimulation()
    output_dir = sim.run_full_simulation()

    print()
    print("=" * 60)
    print(f"Pre-computation complete! Data saved to: {output_dir}")
    print("=" * 60)

    return output_dir


def run_visualizer(data_dir: str = "simulation_data"):
    """Run the Pygame visualizer"""
    from visualizer import VisualizerApp

    print("=" * 60)
    print("GENREG Evolution Visualizer")
    print("=" * 60)
    print()
    print("Controls:")
    print("  SPACE     - Play/Pause")
    print("  LEFT/RIGHT- Seek through timeline")
    print("  UP/DOWN   - Change playback speed")
    print("  1/2/3     - Jump to Early/Mid/Late stage")
    print("  R         - Reset to beginning")
    print("  P         - Open detailed protein view")
    print("  Click     - Scrub timeline")
    print()

    app = VisualizerApp(data_dir)
    app.run()


def interactive_menu():
    """Show interactive menu for user selection"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    data_dir = "simulation_data"

    print("\n" + "=" * 60)
    print("GENREG Evolution Visualizer")
    print("=" * 60)

    # Find checkpoints
    checkpoints = find_checkpoints()

    # Check for existing simulation data
    has_sim_data = os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, "evolution_log.json"))

    # Build menu options
    options = []

    if checkpoints:
        print("\nFound checkpoint files:")
        for i, cp in enumerate(checkpoints):
            filename = os.path.basename(cp)
            print(f"  [{i + 1}] {filename}")
            options.append(('checkpoint', cp))

    print("\nOptions:")
    next_num = len(checkpoints) + 1

    if has_sim_data:
        print(f"  [{next_num}] Load existing simulation data")
        options.append(('load', data_dir))
        next_num += 1

    print(f"  [{next_num}] Generate new simulation data")
    options.append(('generate', None))
    next_num += 1

    print(f"  [0] Exit")

    # Get user choice
    while True:
        try:
            choice = input("\nEnter choice: ").strip()
            if choice == '0':
                print("Exiting.")
                sys.exit(0)

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                break
            else:
                print(f"Please enter a number between 0 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")

    action, data = options[choice_idx]

    if action == 'checkpoint':
        print(f"\nLoading checkpoint: {os.path.basename(data)}")
        genome = load_best_genome_from_checkpoint(data)
        if genome:
            output_dir = run_checkpoint_demo(genome)
            run_visualizer(output_dir)
        else:
            print("Failed to load checkpoint. Exiting.")
            sys.exit(1)

    elif action == 'load':
        print(f"\nLoading existing simulation data from {data}")
        run_visualizer(data)

    elif action == 'generate':
        print("\nGenerating new simulation data...")
        run_simulation()
        run_visualizer(data_dir)


def main():
    parser = argparse.ArgumentParser(description="GENREG Evolution Visualizer")
    parser.add_argument('--simulate', action='store_true',
                        help='Force re-run the simulation')
    parser.add_argument('--visualize', action='store_true',
                        help='Only run visualizer (requires existing data)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load specific checkpoint file')
    parser.add_argument('--data-dir', type=str, default='simulation_data',
                        help='Directory for simulation data')

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    data_dir = args.data_dir

    # Handle specific checkpoint
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
            sys.exit(1)

        genome = load_best_genome_from_checkpoint(args.checkpoint)
        if genome:
            output_dir = run_checkpoint_demo(genome)
            run_visualizer(output_dir)
        else:
            print("Failed to load checkpoint.")
            sys.exit(1)
        return

    if args.simulate:
        # Force simulation
        run_simulation()
        if not args.visualize:
            return

    if args.visualize:
        # Only visualize
        if not os.path.exists(data_dir):
            print(f"Error: Data directory '{data_dir}' not found!")
            print("Run with --simulate first to generate data.")
            sys.exit(1)
        run_visualizer(data_dir)
        return

    # No specific flags - show interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()
