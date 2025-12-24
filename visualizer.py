"""
GENREG Evolution Visualizer
Main Pygame application with three synchronized panels
"""

import pygame
import math
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, PANEL_WIDTH, PANEL_HEIGHT,
    CONTROL_HEIGHT, COLORS, PLAYBACK_CONFIG, ANIMATION_TIMING,
    SIMULATION_CONFIG, SIGNAL_NAMES, ACTION_NAMES, PROTEIN_TYPES
)
from data_structures import Direction, GeneticEventType, EvolutionLogger, GameFrame


class PlaybackState(Enum):
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class Particle:
    """Animated particle for protein cascade visualization"""
    x: float
    y: float
    target_x: float
    target_y: float
    color: Tuple[int, int, int]
    progress: float = 0.0
    speed: float = 0.02
    size: float = 3.0

    def update(self) -> bool:
        """Update particle position, return True if still alive"""
        self.progress += self.speed
        t = self.progress
        self.x = self.x + (self.target_x - self.x) * t
        self.y = self.y + (self.target_y - self.y) * t
        return self.progress < 1.0


class DetailedProteinWindow:
    """Separate window showing detailed protein cascade and neural network.
    Updated for GENREG dual architecture with 6 protein types."""

    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 800

    # Visual constants
    NODE_RADIUS = 14

    # Protein type colors
    PROTEIN_COLORS = {
        'sensor': (255, 200, 100),
        'trend': (200, 150, 255),
        'comparator': (100, 200, 255),
        'integrator': (150, 255, 200),
        'gate': (255, 150, 150),
        'trust_modifier': (255, 215, 0),
    }

    def __init__(self, parent_app):
        self.parent = parent_app
        self.running = True

        # Create window
        self.screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("GENREG Detailed View - Proteins + Neural Network")

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 14)

        # Scroll offset
        self.scroll_y = 0
        self.max_scroll = 0

    def handle_events(self) -> bool:
        """Handle events, return False if window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.scroll_y = max(0, self.scroll_y - 30)
                elif event.key == pygame.K_DOWN:
                    self.scroll_y = min(self.max_scroll, self.scroll_y + 30)
                elif event.key == pygame.K_SPACE:
                    self.parent.toggle_playback()
                elif event.key == pygame.K_LEFT:
                    self.parent.seek(-10)
                elif event.key == pygame.K_RIGHT:
                    self.parent.seek(10)
            elif event.type == pygame.MOUSEWHEEL:
                self.scroll_y = max(0, min(self.max_scroll, self.scroll_y - event.y * 30))
        return True

    def draw(self) -> None:
        """Draw the detailed protein + NN view."""
        self.screen.fill(COLORS['bg_dark'])

        # Get current data
        stage_data = self.parent.get_current_stage_data()
        if not stage_data or not stage_data.game_frames:
            no_data = self.font_large.render("No data - waiting for playback...",
                                             True, COLORS['text_secondary'])
            self.screen.blit(no_data, (self.WINDOW_WIDTH // 2 - 150, self.WINDOW_HEIGHT // 2))
            pygame.display.flip()
            return

        frame_idx = int(self.parent.current_game_frame) % len(stage_data.game_frames)
        frame = stage_data.game_frames[frame_idx]

        # Get frame data
        protein_outputs = frame.get('protein_outputs', {})
        signals = frame.get('signals', {})
        nn_hidden = frame.get('nn_hidden_activations', [])
        nn_output = frame.get('nn_output_activations', [])
        trust_delta = frame.get('trust_delta', 0)
        cumulative_trust = frame.get('cumulative_trust', 0)
        action = frame.get('selected_action', 0)

        # Title
        title = self.font_large.render("GENREG Detailed View", True, COLORS['text_primary'])
        self.screen.blit(title, (20, 10))

        # Playback state
        is_playing = self.parent.playback_state == PlaybackState.PLAYING
        state_text = "PLAYING" if is_playing else "PAUSED"

        gen_text = self.font_medium.render(
            f"Generation: {self.parent.current_generation} | Frame: {frame_idx}",
            True, COLORS['text_secondary']
        )
        self.screen.blit(gen_text, (20, 45))

        # Playback badge
        badge_x = self.WINDOW_WIDTH - 100
        badge_color = COLORS['trust_high'] if is_playing else COLORS['button']
        pygame.draw.rect(self.screen, badge_color, (badge_x, 40, 80, 25), border_radius=12)
        badge_text = self.font_small.render(state_text, True, COLORS['text_primary'])
        badge_rect = badge_text.get_rect(center=(badge_x + 40, 52))
        self.screen.blit(badge_text, badge_rect)

        # Create scrollable content surface
        content_height = 900
        content_surface = pygame.Surface((self.WINDOW_WIDTH - 40, content_height))
        content_surface.fill(COLORS['bg_panel'])

        # ============================================================
        # SECTION 1: PROTEIN CASCADE (top section)
        # ============================================================
        section1_y = 10
        section1_title = self.font_medium.render("PROTEIN CASCADE -> TRUST",
                                                  True, COLORS['trust_modifier_protein'])
        content_surface.blit(section1_title, (20, section1_y))

        # Get proteins from genome snapshot
        proteins_by_type = {}
        if stage_data.genome_snapshot:
            proteins_list = stage_data.genome_snapshot.get('proteins', [])
            for p_data in proteins_list:
                ptype = p_data.get('type', 'sensor')
                if ptype not in proteins_by_type:
                    proteins_by_type[ptype] = []
                proteins_by_type[ptype].append(p_data)

        # Draw protein types in columns
        col_width = 150
        row_height = 45
        start_x = 30
        start_y = section1_y + 35

        type_order = ['sensor', 'trend', 'comparator', 'integrator', 'gate', 'trust_modifier']

        for col_idx, ptype in enumerate(type_order):
            col_x = start_x + col_idx * col_width

            # Type header
            pcolor = self.PROTEIN_COLORS.get(ptype, COLORS['protein_inactive'])
            header = self.font_small.render(ptype.upper()[:8], True, pcolor)
            content_surface.blit(header, (col_x, start_y))

            # Draw proteins of this type
            type_proteins = proteins_by_type.get(ptype, [])
            for row_idx, p_data in enumerate(type_proteins[:5]):  # Max 5 per type
                p_name = p_data.get('name', '?')
                output = protein_outputs.get(p_name, 0)

                node_x = col_x + 10
                node_y = start_y + 25 + row_idx * row_height

                # Draw node
                radius = self.NODE_RADIUS
                intensity = min(1.0, abs(output))
                inactive = COLORS['protein_inactive']
                node_color = tuple(int(inactive[i] * (1 - intensity) + pcolor[i] * intensity)
                                  for i in range(3))

                # Glow for high activation
                if intensity > 0.5:
                    pygame.draw.circle(content_surface, pcolor, (node_x, node_y), radius + 4)

                pygame.draw.circle(content_surface, node_color, (node_x, node_y), radius)
                pygame.draw.circle(content_surface, COLORS['text_secondary'],
                                 (node_x, node_y), radius, 1)

                # Label
                short_name = p_name[:10]
                label = self.font_tiny.render(short_name, True, COLORS['text_primary'])
                content_surface.blit(label, (node_x + radius + 5, node_y - 6))

                # Value
                val_text = self.font_tiny.render(f"{output:.2f}", True, COLORS['text_secondary'])
                content_surface.blit(val_text, (node_x + radius + 5, node_y + 6))

        # Trust output display
        trust_x = start_x + 6 * col_width
        trust_y = start_y + 20

        trust_label = self.font_medium.render("TRUST", True, COLORS['trust_modifier_protein'])
        content_surface.blit(trust_label, (trust_x, trust_y))

        # Trust delta bar
        bar_y = trust_y + 30
        bar_width = 80
        bar_height = 25

        pygame.draw.rect(content_surface, COLORS['trust_bar_bg'],
                        (trust_x, bar_y, bar_width, bar_height), border_radius=4)

        # Fill based on trust delta
        center_x = trust_x + bar_width // 2
        delta_normalized = max(-1, min(1, trust_delta / 2))

        if trust_delta > 0:
            fill_width = int(delta_normalized * bar_width / 2)
            pygame.draw.rect(content_surface, COLORS['trust_delta_positive'],
                           (center_x, bar_y, fill_width, bar_height), border_radius=4)
        elif trust_delta < 0:
            fill_width = int(-delta_normalized * bar_width / 2)
            pygame.draw.rect(content_surface, COLORS['trust_delta_negative'],
                           (center_x - fill_width, bar_y, fill_width, bar_height), border_radius=4)

        pygame.draw.line(content_surface, COLORS['text_primary'],
                        (center_x, bar_y), (center_x, bar_y + bar_height), 2)

        # Trust values
        delta_color = COLORS['trust_delta_positive'] if trust_delta >= 0 else COLORS['trust_delta_negative']
        delta_sign = "+" if trust_delta > 0 else ""
        delta_text = self.font_medium.render(f"{delta_sign}{trust_delta:.2f}", True, delta_color)
        content_surface.blit(delta_text, (trust_x, bar_y + 30))

        cum_text = self.font_small.render(f"Total: {cumulative_trust:.1f}", True, COLORS['text_secondary'])
        content_surface.blit(cum_text, (trust_x, bar_y + 55))

        # ============================================================
        # SECTION 2: INPUT SIGNALS
        # ============================================================
        section2_y = 280
        pygame.draw.line(content_surface, COLORS['text_secondary'],
                        (20, section2_y - 10), (self.WINDOW_WIDTH - 60, section2_y - 10), 1)

        signals_title = self.font_medium.render("INPUT SIGNALS (11)", True, COLORS['nn_input'])
        content_surface.blit(signals_title, (20, section2_y))

        # Draw all 11 signals as bars
        sig_start_y = section2_y + 30
        bar_width = 120
        bar_height = 18

        for i, sig_name in enumerate(SIGNAL_NAMES):
            col = i // 6
            row = i % 6

            x = 30 + col * 450
            y = sig_start_y + row * 28

            sig_val = signals.get(sig_name, 0)

            # Background
            pygame.draw.rect(content_surface, COLORS['bg_dark'],
                           (x, y, bar_width, bar_height), border_radius=3)

            # Normalize and fill
            max_val = 20.0 if 'dist' in sig_name or 'steps' in sig_name else 10.0
            normalized = min(1.0, abs(sig_val) / max_val)
            fill_width = int(normalized * bar_width)

            fill_color = COLORS['nn_input'] if sig_val >= 0 else COLORS['nn_weight_negative']
            if fill_width > 0:
                pygame.draw.rect(content_surface, fill_color,
                               (x, y, fill_width, bar_height), border_radius=3)

            # Label and value
            label = self.font_tiny.render(sig_name, True, COLORS['text_primary'])
            content_surface.blit(label, (x + bar_width + 10, y + 2))

            val = self.font_tiny.render(f"{sig_val:.1f}", True, COLORS['text_secondary'])
            content_surface.blit(val, (x + bar_width + 100, y + 2))

        # ============================================================
        # SECTION 3: NEURAL NETWORK WITH CONNECTIONS
        # ============================================================
        section3_y = 480
        pygame.draw.line(content_surface, COLORS['text_secondary'],
                        (20, section3_y - 10), (self.WINDOW_WIDTH - 60, section3_y - 10), 1)

        nn_title = self.font_medium.render("NEURAL CONTROLLER -> ACTION", True, COLORS['nn_output'])
        content_surface.blit(nn_title, (20, section3_y))

        # Layout positions
        input_x = 60
        hidden_x = 350
        output_x = 650

        input_start_y = section3_y + 50
        hidden_start_y = section3_y + 45
        output_start_y = section3_y + 80

        input_spacing = 28
        hidden_spacing_x = 40
        hidden_spacing_y = 38
        output_spacing = 50

        neuron_radius = 10
        input_radius = 8

        # Calculate all positions first
        input_positions = []
        for i in range(11):
            ix = input_x
            iy = input_start_y + i * input_spacing
            input_positions.append((ix, iy))

        hidden_positions = []
        for i in range(32):
            row = i // 8
            col = i % 8
            hx = hidden_x + col * hidden_spacing_x
            hy = hidden_start_y + row * hidden_spacing_y
            hidden_positions.append((hx, hy))

        output_positions = []
        for i in range(4):
            ox = output_x
            oy = output_start_y + i * output_spacing
            output_positions.append((ox, oy))

        # Get controller weights from genome
        controller_data = None
        if stage_data.genome_snapshot:
            controller_data = stage_data.genome_snapshot.get('controller', None)

        # --- LAYER 1: Draw connections (behind nodes) ---
        connection_alpha = 0.12  # Low opacity for connections

        # Input to Hidden connections (sample to avoid clutter)
        if controller_data and 'w1' in controller_data:
            w1 = controller_data['w1']
            for h_idx in range(min(32, len(w1))):
                hx, hy = hidden_positions[h_idx]
                h_activation = nn_hidden[h_idx] if h_idx < len(nn_hidden) else 0

                # Only draw connections for active hidden neurons or sample
                if abs(h_activation) > 0.3 or h_idx % 4 == 0:
                    for i_idx in range(min(11, len(w1[h_idx]))):
                        weight = w1[h_idx][i_idx]
                        ix, iy = input_positions[i_idx]

                        # Color based on weight sign
                        if weight > 0:
                            base_color = COLORS['nn_weight_positive']
                        else:
                            base_color = COLORS['nn_weight_negative']

                        # Intensity based on weight magnitude and activation
                        intensity = min(1.0, abs(weight) * abs(h_activation) * 2)
                        alpha = connection_alpha + intensity * 0.3

                        bg = COLORS['bg_panel']
                        line_color = tuple(int(bg[j] * (1 - alpha) + base_color[j] * alpha) for j in range(3))

                        line_width = 1 if intensity < 0.3 else 2
                        pygame.draw.line(content_surface, line_color,
                                       (ix + input_radius, iy), (hx - neuron_radius, hy), line_width)

        # Hidden to Output connections
        if controller_data and 'w2' in controller_data:
            w2 = controller_data['w2']
            for o_idx in range(min(4, len(w2))):
                ox, oy = output_positions[o_idx]
                o_activation = nn_output[o_idx] if o_idx < len(nn_output) else 0
                is_selected = (o_idx == action)

                for h_idx in range(min(32, len(w2[o_idx]))):
                    weight = w2[o_idx][h_idx]
                    hx, hy = hidden_positions[h_idx]
                    h_activation = nn_hidden[h_idx] if h_idx < len(nn_hidden) else 0

                    # Color based on weight sign
                    if weight > 0:
                        base_color = COLORS['nn_weight_positive']
                    else:
                        base_color = COLORS['nn_weight_negative']

                    # Highlight connections to selected action
                    if is_selected:
                        intensity = min(1.0, abs(weight) * abs(h_activation) * 3)
                        alpha = 0.2 + intensity * 0.6
                        line_width = 2 if intensity > 0.2 else 1
                    else:
                        intensity = min(1.0, abs(weight) * abs(h_activation) * 2)
                        alpha = connection_alpha + intensity * 0.2
                        line_width = 1

                    bg = COLORS['bg_panel']
                    line_color = tuple(int(bg[j] * (1 - alpha) + base_color[j] * alpha) for j in range(3))

                    pygame.draw.line(content_surface, line_color,
                                   (hx + neuron_radius, hy), (ox - 15, oy), line_width)

        # --- LAYER 2: Draw labels ---
        input_label = self.font_small.render("Inputs (11)", True, COLORS['nn_input'])
        content_surface.blit(input_label, (input_x - 20, section3_y + 25))

        hidden_label = self.font_small.render("Hidden (32)", True, COLORS['nn_hidden'])
        content_surface.blit(hidden_label, (hidden_x + 60, section3_y + 25))

        output_label = self.font_small.render("Actions", True, COLORS['nn_output'])
        content_surface.blit(output_label, (output_x - 10, section3_y + 25))

        # --- LAYER 3: Draw input nodes ---
        for i, sig_name in enumerate(SIGNAL_NAMES):
            ix, iy = input_positions[i]
            sig_val = signals.get(sig_name, 0)

            # Normalize for color
            intensity = min(1.0, abs(sig_val) / 10.0)
            color = tuple(int(COLORS['protein_inactive'][j] * (1 - intensity) +
                            COLORS['nn_input'][j] * intensity) for j in range(3))

            pygame.draw.circle(content_surface, color, (ix, iy), input_radius)
            pygame.draw.circle(content_surface, COLORS['text_secondary'], (ix, iy), input_radius, 1)

            # Short label
            short_name = sig_name[:6]
            label = self.font_tiny.render(short_name, True, COLORS['text_secondary'])
            content_surface.blit(label, (ix - 50, iy - 5))

        # --- LAYER 4: Draw hidden nodes ---
        for i in range(min(32, len(nn_hidden) if nn_hidden else 32)):
            hx, hy = hidden_positions[i]
            activation = nn_hidden[i] if i < len(nn_hidden) else 0
            intensity = (activation + 1) / 2  # tanh [-1, 1] -> [0, 1]

            color = tuple(int(COLORS['protein_inactive'][j] * (1 - intensity) +
                            COLORS['nn_hidden'][j] * intensity) for j in range(3))

            # Glow for highly active neurons
            if intensity > 0.7:
                pygame.draw.circle(content_surface, COLORS['nn_hidden'], (hx, hy), neuron_radius + 4)

            pygame.draw.circle(content_surface, color, (hx, hy), neuron_radius)
            pygame.draw.circle(content_surface, COLORS['text_secondary'], (hx, hy), neuron_radius, 1)

        # --- LAYER 5: Draw output nodes ---
        for i, action_name in enumerate(ACTION_NAMES):
            ox, oy = output_positions[i]
            activation = nn_output[i] if i < len(nn_output) else 0
            is_selected = (i == action)

            # Background box
            box_width = 80
            box_height = 35
            bg_color = COLORS['nn_output'] if is_selected else COLORS['bg_dark']
            pygame.draw.rect(content_surface, bg_color,
                           (ox - 10, oy - box_height//2, box_width, box_height), border_radius=6)

            # Glow for selected
            if is_selected:
                for r in range(3):
                    glow_rect = (ox - 10 - r*2, oy - box_height//2 - r*2,
                                box_width + r*4, box_height + r*4)
                    pygame.draw.rect(content_surface, COLORS['nn_output'], glow_rect, 1, border_radius=8)

            # Border
            border_color = COLORS['text_highlight'] if is_selected else COLORS['text_secondary']
            pygame.draw.rect(content_surface, border_color,
                           (ox - 10, oy - box_height//2, box_width, box_height), 2, border_radius=6)

            # Label
            text_color = COLORS['bg_dark'] if is_selected else COLORS['text_primary']
            action_text = self.font_medium.render(action_name, True, text_color)
            text_rect = action_text.get_rect(center=(ox + box_width//2 - 10, oy - 5))
            content_surface.blit(action_text, text_rect)

            # Activation value
            val_color = COLORS['bg_dark'] if is_selected else COLORS['text_secondary']
            val_text = self.font_small.render(f"{activation:.2f}", True, val_color)
            val_rect = val_text.get_rect(center=(ox + box_width//2 - 10, oy + 10))
            content_surface.blit(val_text, val_rect)

        # Selected action indicator (large)
        action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else "?"
        selected_text = self.font_large.render(f"ACTION: {action_name}", True, COLORS['nn_output'])
        content_surface.blit(selected_text, (output_x + 100, section3_y + 100))

        # Calculate max scroll
        self.max_scroll = max(0, content_height - (self.WINDOW_HEIGHT - 110))

        # Blit content
        self.screen.blit(content_surface, (20, 80),
                         (0, self.scroll_y, self.WINDOW_WIDTH - 40, self.WINDOW_HEIGHT - 110))

        # Scroll indicator
        if self.max_scroll > 0:
            scroll_ratio = self.scroll_y / self.max_scroll if self.max_scroll > 0 else 0
            indicator_height = max(30, (self.WINDOW_HEIGHT - 110) ** 2 / content_height)
            indicator_y = 80 + scroll_ratio * (self.WINDOW_HEIGHT - 110 - indicator_height)
            pygame.draw.rect(self.screen, COLORS['timeline_fill'],
                             (self.WINDOW_WIDTH - 15, int(indicator_y), 10, int(indicator_height)),
                             border_radius=5)

        # Controls hint
        hint = self.font_small.render(
            "ESC: Close | SPACE: Play/Pause | Scroll: Navigate",
            True, COLORS['text_secondary'])
        self.screen.blit(hint, (20, self.WINDOW_HEIGHT - 25))

        pygame.display.flip()

    def run_frame(self) -> bool:
        """Run one frame of the detail window, return False to close."""
        if not self.handle_events():
            return False
        self.draw()
        return True


class VisualizerApp:
    """Main visualizer application"""

    def __init__(self, data_dir: str = "simulation_data"):
        pygame.init()
        pygame.display.set_caption("GENREG Evolution Visualizer")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # Load fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Playback state
        self.playback_state = PlaybackState.PAUSED
        self.playback_speed = 1.0
        self.current_generation = 0
        self.current_frame = 0
        self.frames_per_generation = PLAYBACK_CONFIG['frames_per_generation']

        # Animation state
        self.particles: List[Particle] = []
        self.active_events: List[Dict] = []
        self.event_animations: Dict[str, float] = {}

        # Panel surfaces
        self.snake_panel = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT))
        self.protein_panel = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT))
        self.population_panel = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT))
        self.control_panel = pygame.Surface((WINDOW_WIDTH, CONTROL_HEIGHT))

        # Detailed protein window
        self.detail_window = None
        self.detail_button_rect = pygame.Rect(0, 0, 0, 0)  # Will be set in draw_controls

        # Try to load data
        self.data_loaded = False
        self.logger = None
        self.stage_exemplars = {}
        self.current_stage = 'early'
        self.current_game_frame = 0

        if os.path.exists(data_dir):
            self.load_data(data_dir)
        else:
            print(f"Data directory '{data_dir}' not found. Run simulation first.")

    def load_data(self, data_dir: str) -> bool:
        """Load pre-computed simulation data"""
        try:
            self.logger = EvolutionLogger.load_all(data_dir)
            self.stage_exemplars = self.logger.stage_exemplars
            self.data_loaded = True
            print(f"Loaded {len(self.logger.generation_snapshots)} generations")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def get_current_snapshot(self):
        """Get generation snapshot for current playback position"""
        if not self.data_loaded or not self.logger.generation_snapshots:
            return None
        idx = min(self.current_generation, len(self.logger.generation_snapshots) - 1)
        return self.logger.generation_snapshots[idx]

    def get_current_stage_data(self):
        """Get stage exemplar data for current generation"""
        snapshot = self.get_current_snapshot()
        if not snapshot:
            return None

        stage = snapshot.stage
        if stage != self.current_stage:
            self.current_stage = stage
            self.current_game_frame = 0

        return self.stage_exemplars.get(stage)

    def handle_events(self) -> None:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.toggle_playback()
                elif event.key == pygame.K_LEFT:
                    self.seek(-10)
                elif event.key == pygame.K_RIGHT:
                    self.seek(10)
                elif event.key == pygame.K_UP:
                    self.change_speed(1)
                elif event.key == pygame.K_DOWN:
                    self.change_speed(-1)
                elif event.key == pygame.K_r:
                    self.reset_playback()
                elif event.key == pygame.K_1:
                    self.jump_to_stage('early')
                elif event.key == pygame.K_2:
                    self.jump_to_stage('mid')
                elif event.key == pygame.K_3:
                    self.jump_to_stage('late')
                elif event.key == pygame.K_p:
                    self.open_detail_window()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_click(event.pos)

    def toggle_playback(self) -> None:
        """Toggle play/pause"""
        if self.playback_state == PlaybackState.PLAYING:
            self.playback_state = PlaybackState.PAUSED
        else:
            self.playback_state = PlaybackState.PLAYING

    def seek(self, delta_generations: int) -> None:
        """Seek forward/backward in timeline"""
        if not self.data_loaded:
            return
        max_gen = len(self.logger.generation_snapshots) - 1
        self.current_generation = max(0, min(max_gen, self.current_generation + delta_generations))
        self.current_frame = 0

    def change_speed(self, direction: int) -> None:
        """Change playback speed"""
        speeds = PLAYBACK_CONFIG['speed_options']
        current_idx = speeds.index(self.playback_speed) if self.playback_speed in speeds else 2
        new_idx = max(0, min(len(speeds) - 1, current_idx + direction))
        self.playback_speed = speeds[new_idx]

    def reset_playback(self) -> None:
        """Reset to beginning"""
        self.current_generation = 0
        self.current_frame = 0
        self.current_game_frame = 0
        self.playback_state = PlaybackState.PAUSED

    def jump_to_stage(self, stage: str) -> None:
        """Jump to a specific evolution stage"""
        if not self.data_loaded:
            return

        for i, snapshot in enumerate(self.logger.generation_snapshots):
            if snapshot.stage == stage:
                self.current_generation = i
                self.current_frame = 0
                self.current_game_frame = 0
                break

    def open_detail_window(self) -> None:
        """Open the detailed protein cascade window"""
        if self.detail_window is None:
            # Save main window state
            main_caption = pygame.display.get_caption()

            # Run detailed window
            self.detail_window = DetailedProteinWindow(self)

            # Run the detail window in a loop until closed
            detail_clock = pygame.time.Clock()
            while self.detail_window.run_frame():
                # Also update main app state so detail window stays in sync
                self.update()
                detail_clock.tick(FPS)

            # Restore main window
            self.detail_window = None
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("GENREG Evolution Visualizer")

    def handle_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse clicks"""
        x, y = pos

        # Check if click is in control area
        if y > PANEL_HEIGHT:
            control_y = y - PANEL_HEIGHT

            # Check detail button click
            button_rect_screen = pygame.Rect(
                self.detail_button_rect.x,
                self.detail_button_rect.y + PANEL_HEIGHT,
                self.detail_button_rect.width,
                self.detail_button_rect.height
            )
            if button_rect_screen.collidepoint(x, y):
                self.open_detail_window()
                return

            # Timeline click
            if 50 < control_y < 80:
                timeline_x = x - 100
                timeline_width = WINDOW_WIDTH - 200
                if 0 <= timeline_x <= timeline_width and self.data_loaded:
                    progress = timeline_x / timeline_width
                    max_gen = len(self.logger.generation_snapshots) - 1
                    self.current_generation = int(progress * max_gen)
                    self.current_frame = 0

    def update(self) -> None:
        """Update simulation state"""
        if self.playback_state == PlaybackState.PLAYING and self.data_loaded:
            # Advance frame
            self.current_frame += self.playback_speed
            frames_needed = self.frames_per_generation / self.playback_speed

            if self.current_frame >= frames_needed:
                self.current_frame = 0
                self.current_generation += 1

                if self.current_generation >= len(self.logger.generation_snapshots):
                    self.current_generation = len(self.logger.generation_snapshots) - 1
                    self.playback_state = PlaybackState.PAUSED

        # Update game frame for stage exemplar
        stage_data = self.get_current_stage_data()
        if stage_data and self.playback_state == PlaybackState.PLAYING:
            self.current_game_frame += self.playback_speed * 0.5
            if self.current_game_frame >= len(stage_data.game_frames):
                self.current_game_frame = 0

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

        # Update event animations
        for event_id in list(self.event_animations.keys()):
            self.event_animations[event_id] -= 0.016
            if self.event_animations[event_id] <= 0:
                del self.event_animations[event_id]

    def draw_snake_panel(self) -> None:
        """Draw the Snake Game Evolution panel (left)"""
        self.snake_panel.fill(COLORS['bg_panel'])

        # Title
        title = self.font_large.render("Snake Evolution", True, COLORS['text_primary'])
        self.snake_panel.blit(title, (20, 10))

        # Stage indicator
        snapshot = self.get_current_snapshot()
        stage = snapshot.stage if snapshot else 'early'
        stage_colors = {
            'early': COLORS['stage_early'],
            'mid': COLORS['stage_mid'],
            'late': COLORS['stage_late']
        }
        stage_text = self.font_medium.render(f"Stage: {stage.upper()}", True, stage_colors[stage])
        self.snake_panel.blit(stage_text, (20, 45))

        # Get stage exemplar data
        stage_data = self.get_current_stage_data()

        # Draw game grid (10x10 official size)
        grid_offset_x = 40
        grid_offset_y = 80
        cell_size = 35  # Larger cells for 10x10 grid
        grid_size = SIMULATION_CONFIG['grid_size']

        # Draw grid background
        grid_rect = pygame.Rect(
            grid_offset_x - 2, grid_offset_y - 2,
            grid_size * cell_size + 4, grid_size * cell_size + 4
        )
        pygame.draw.rect(self.snake_panel, COLORS['grid'], grid_rect)
        pygame.draw.rect(self.snake_panel, COLORS['wall'], grid_rect, 2)

        # Draw grid lines
        for i in range(grid_size + 1):
            # Vertical
            x = grid_offset_x + i * cell_size
            pygame.draw.line(self.snake_panel, COLORS['bg_dark'],
                           (x, grid_offset_y), (x, grid_offset_y + grid_size * cell_size), 1)
            # Horizontal
            y = grid_offset_y + i * cell_size
            pygame.draw.line(self.snake_panel, COLORS['bg_dark'],
                           (grid_offset_x, y), (grid_offset_x + grid_size * cell_size, y), 1)

        if stage_data and stage_data.game_frames:
            frame_idx = int(self.current_game_frame) % len(stage_data.game_frames)
            frame = stage_data.game_frames[frame_idx]

            # Draw food with glow
            fx, fy = frame['food_position']
            food_center = (
                grid_offset_x + fx * cell_size + cell_size // 2,
                grid_offset_y + fy * cell_size + cell_size // 2
            )
            # Glow effect
            for r in range(20, 8, -2):
                pygame.draw.circle(self.snake_panel, COLORS['food_glow'], food_center, r)
            pygame.draw.circle(self.snake_panel, COLORS['food'], food_center, 10)

            # Draw snake
            snake_positions = frame['snake_positions']
            for i, (sx, sy) in enumerate(snake_positions):
                if i == 0:
                    color = COLORS['snake_head']
                elif i < len(snake_positions) - 1:
                    t = i / max(len(snake_positions) - 1, 1)
                    color = tuple(int(COLORS['snake_head'][j] * (1 - t) +
                                      COLORS['snake_tail'][j] * t) for j in range(3))
                else:
                    color = COLORS['snake_tail']

                rect = pygame.Rect(
                    grid_offset_x + sx * cell_size + 2,
                    grid_offset_y + sy * cell_size + 2,
                    cell_size - 4, cell_size - 4
                )
                pygame.draw.rect(self.snake_panel, color, rect, border_radius=5)

            # Stats panel below grid
            stats_y = grid_offset_y + grid_size * cell_size + 15

            # Score
            score_text = self.font_medium.render(f"Score: {frame['score']}", True, COLORS['text_primary'])
            self.snake_panel.blit(score_text, (20, stats_y))

            # Energy bar
            energy = frame.get('energy', 25)
            max_energy = 25 + frame['score'] * 2  # Increases with food eaten
            energy_ratio = min(1.0, energy / max(max_energy, 1))

            energy_label = self.font_small.render("Energy:", True, COLORS['text_secondary'])
            self.snake_panel.blit(energy_label, (20, stats_y + 30))

            bar_x = 90
            bar_width = 200
            bar_height = 16
            pygame.draw.rect(self.snake_panel, COLORS['energy_bar_bg'],
                           (bar_x, stats_y + 30, bar_width, bar_height), border_radius=4)

            energy_color = COLORS['energy_high'] if energy_ratio > 0.3 else COLORS['energy_low']
            fill_width = int(energy_ratio * bar_width)
            if fill_width > 0:
                pygame.draw.rect(self.snake_panel, energy_color,
                               (bar_x, stats_y + 30, fill_width, bar_height), border_radius=4)

            energy_text = self.font_small.render(f"{int(energy)}", True, COLORS['text_primary'])
            self.snake_panel.blit(energy_text, (bar_x + bar_width + 10, stats_y + 30))

            # Trust display
            trust_delta = frame.get('trust_delta', 0)
            cumulative_trust = frame.get('cumulative_trust', 0)

            trust_label = self.font_small.render("Trust:", True, COLORS['text_secondary'])
            self.snake_panel.blit(trust_label, (20, stats_y + 55))

            trust_color = COLORS['trust_delta_positive'] if cumulative_trust >= 0 else COLORS['trust_delta_negative']
            trust_value = self.font_medium.render(f"{cumulative_trust:.1f}", True, trust_color)
            self.snake_panel.blit(trust_value, (90, stats_y + 52))

            # Trust delta indicator
            if trust_delta != 0:
                delta_color = COLORS['trust_delta_positive'] if trust_delta > 0 else COLORS['trust_delta_negative']
                delta_sign = "+" if trust_delta > 0 else ""
                delta_text = self.font_small.render(f"({delta_sign}{trust_delta:.2f})", True, delta_color)
                self.snake_panel.blit(delta_text, (180, stats_y + 55))

            # Action indicator
            action = frame.get('selected_action', 0)
            action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else "?"
            action_text = self.font_medium.render(f"Action: {action_name}", True, COLORS['nn_output'])
            self.snake_panel.blit(action_text, (20, stats_y + 80))

            # Frame counter
            frame_text = self.font_small.render(
                f"Frame: {frame_idx}/{len(stage_data.game_frames)}",
                True, COLORS['text_secondary']
            )
            self.snake_panel.blit(frame_text, (200, stats_y + 85))

        # Description
        if stage_data:
            desc_lines = self._wrap_text(stage_data.description, self.font_small, PANEL_WIDTH - 40)
            y_pos = PANEL_HEIGHT - 60
            for line in desc_lines:
                text = self.font_small.render(line, True, COLORS['text_secondary'])
                self.snake_panel.blit(text, (20, y_pos))
                y_pos += 20

    def draw_protein_panel(self) -> None:
        """Draw the Protein Cascade + Neural Network panel (center)"""
        self.protein_panel.fill(COLORS['bg_panel'])

        # Title
        title = self.font_large.render("GENREG Dual Architecture", True, COLORS['text_primary'])
        self.protein_panel.blit(title, (20, 10))

        stage_data = self.get_current_stage_data()
        if not stage_data or not stage_data.game_frames:
            no_data = self.font_medium.render("No data loaded", True, COLORS['text_secondary'])
            self.protein_panel.blit(no_data, (PANEL_WIDTH // 2 - 60, PANEL_HEIGHT // 2))
            return

        frame_idx = int(self.current_game_frame) % len(stage_data.game_frames)
        frame = stage_data.game_frames[frame_idx]
        protein_outputs = frame.get('protein_outputs', {})
        signals = frame.get('signals', {})
        nn_hidden = frame.get('nn_hidden_activations', [])
        nn_output = frame.get('nn_output_activations', [])
        trust_delta = frame.get('trust_delta', 0)
        cumulative_trust = frame.get('cumulative_trust', 0)

        # === SECTION 1: Protein Cascade (top half) ===
        section1_y = 40
        section_label = self.font_medium.render("Protein Cascade (TRUST)", True, COLORS['trust_modifier_protein'])
        self.protein_panel.blit(section_label, (20, section1_y))

        # Draw protein types in cascade order
        protein_x_positions = {
            'sensor': 60,
            'trend': 150,
            'comparator': 150,
            'integrator': 230,
            'gate': 230,
            'trust_modifier': 350
        }

        # Organize proteins by type from genome
        if stage_data.genome_snapshot:
            proteins_data = stage_data.genome_snapshot.get('proteins', [])
            proteins_by_type = {}

            for p_data in proteins_data:
                ptype = p_data.get('type', 'sensor')
                if ptype not in proteins_by_type:
                    proteins_by_type[ptype] = []
                proteins_by_type[ptype].append(p_data)

            # Draw each protein type column
            y_offset = section1_y + 30

            # Draw sensors (compact - just show count and sample activations)
            sensor_label = self.font_small.render("Sensors (11)", True, COLORS['sensor_protein'])
            self.protein_panel.blit(sensor_label, (20, y_offset))

            # Show a few key signal values
            key_signals = ['dist_to_food', 'energy', 'near_wall', 'food_dx', 'food_dy']
            for i, sig_name in enumerate(key_signals[:4]):
                sig_val = signals.get(sig_name, 0)
                color = COLORS['sensor_protein'] if abs(sig_val) > 0.1 else COLORS['protein_inactive']
                short_name = sig_name[:8]
                sig_text = self.font_small.render(f"{short_name}: {sig_val:.1f}", True, color)
                self.protein_panel.blit(sig_text, (20, y_offset + 20 + i * 18))

            # Draw processing proteins (trend, comparator, integrator, gate)
            proc_x = 180
            proc_y = y_offset

            proc_types = ['trend', 'comparator', 'integrator', 'gate']
            proc_colors = [COLORS['trend_protein'], COLORS['comparator_protein'],
                          COLORS['integrator_protein'], COLORS['gate_protein']]

            for j, (ptype, pcolor) in enumerate(zip(proc_types, proc_colors)):
                type_proteins = proteins_by_type.get(ptype, [])
                if type_proteins:
                    # Draw type label
                    label = self.font_small.render(ptype.title(), True, pcolor)
                    self.protein_panel.blit(label, (proc_x, proc_y + j * 50))

                    # Draw protein nodes
                    for k, p_data in enumerate(type_proteins[:2]):  # Show max 2
                        name = p_data.get('name', '?')
                        output = protein_outputs.get(name, 0)
                        node_x = proc_x + 80 + k * 50
                        node_y = proc_y + j * 50 + 8

                        # Draw small node
                        radius = 12
                        intensity = min(1.0, abs(output))
                        node_color = tuple(int(COLORS['protein_inactive'][i] * (1 - intensity) +
                                              pcolor[i] * intensity) for i in range(3))
                        pygame.draw.circle(self.protein_panel, node_color, (node_x, node_y), radius)
                        pygame.draw.circle(self.protein_panel, COLORS['text_secondary'],
                                         (node_x, node_y), radius, 1)

            # Draw trust modifiers and trust output
            trust_x = 350
            trust_y = y_offset

            trust_label = self.font_medium.render("TRUST", True, COLORS['trust_modifier_protein'])
            self.protein_panel.blit(trust_label, (trust_x, trust_y))

            # Trust delta bar
            bar_y = trust_y + 30
            bar_width = 100
            bar_height = 20

            pygame.draw.rect(self.protein_panel, COLORS['trust_bar_bg'],
                           (trust_x, bar_y, bar_width, bar_height), border_radius=4)

            # Fill based on trust delta (centered, extends left or right)
            center_x = trust_x + bar_width // 2
            delta_normalized = max(-1, min(1, trust_delta / 2))  # Normalize to [-1, 1]

            if trust_delta > 0:
                fill_width = int(delta_normalized * bar_width / 2)
                pygame.draw.rect(self.protein_panel, COLORS['trust_delta_positive'],
                               (center_x, bar_y, fill_width, bar_height), border_radius=4)
            elif trust_delta < 0:
                fill_width = int(-delta_normalized * bar_width / 2)
                pygame.draw.rect(self.protein_panel, COLORS['trust_delta_negative'],
                               (center_x - fill_width, bar_y, fill_width, bar_height), border_radius=4)

            # Center line
            pygame.draw.line(self.protein_panel, COLORS['text_primary'],
                           (center_x, bar_y), (center_x, bar_y + bar_height), 2)

            # Trust delta value
            delta_color = COLORS['trust_delta_positive'] if trust_delta >= 0 else COLORS['trust_delta_negative']
            delta_sign = "+" if trust_delta > 0 else ""
            delta_text = self.font_medium.render(f"{delta_sign}{trust_delta:.2f}", True, delta_color)
            self.protein_panel.blit(delta_text, (trust_x, bar_y + 25))

            # Cumulative trust
            cum_text = self.font_small.render(f"Total: {cumulative_trust:.1f}", True, COLORS['text_secondary'])
            self.protein_panel.blit(cum_text, (trust_x, bar_y + 48))

        # === SECTION 2: Neural Controller (bottom half) ===
        section2_y = 280
        pygame.draw.line(self.protein_panel, COLORS['text_secondary'],
                        (20, section2_y - 10), (PANEL_WIDTH - 20, section2_y - 10), 1)

        nn_label = self.font_medium.render("Neural Controller (ACTION)", True, COLORS['nn_output'])
        self.protein_panel.blit(nn_label, (20, section2_y))

        # Draw neural network visualization
        nn_start_y = section2_y + 30

        # Input layer (11 signals) - compact
        input_x = 40
        input_label = self.font_small.render("Inputs", True, COLORS['nn_input'])
        self.protein_panel.blit(input_label, (input_x - 10, nn_start_y))

        input_y_start = nn_start_y + 20
        for i in range(min(11, len(SIGNAL_NAMES))):
            sig_name = SIGNAL_NAMES[i]
            sig_val = signals.get(sig_name, 0)

            # Small bar representation
            bar_x = input_x
            bar_y = input_y_start + i * 22
            bar_width = 80
            bar_height = 14

            pygame.draw.rect(self.protein_panel, COLORS['bg_dark'],
                           (bar_x, bar_y, bar_width, bar_height), border_radius=2)

            # Normalize and fill
            normalized = min(1.0, abs(sig_val) / 20)  # Rough normalization
            fill_width = int(normalized * bar_width)
            fill_color = COLORS['nn_input'] if sig_val >= 0 else COLORS['nn_weight_negative']
            if fill_width > 0:
                pygame.draw.rect(self.protein_panel, fill_color,
                               (bar_x, bar_y, fill_width, bar_height), border_radius=2)

            # Label
            short_name = sig_name[:6]
            name_text = self.font_small.render(short_name, True, COLORS['text_secondary'])
            self.protein_panel.blit(name_text, (bar_x + bar_width + 5, bar_y))

        # Hidden layer (32 neurons) - show subset
        hidden_x = 220
        hidden_label = self.font_small.render("Hidden (32)", True, COLORS['nn_hidden'])
        self.protein_panel.blit(hidden_label, (hidden_x, nn_start_y))

        # Draw 16 neurons in 2 columns
        neurons_to_show = min(16, len(nn_hidden)) if nn_hidden else 16
        for i in range(neurons_to_show):
            col = i // 8
            row = i % 8

            nx = hidden_x + col * 45
            ny = nn_start_y + 25 + row * 28

            activation = nn_hidden[i] if i < len(nn_hidden) else 0
            intensity = (activation + 1) / 2  # tanh output is [-1, 1]

            color = tuple(int(COLORS['protein_inactive'][j] * (1 - intensity) +
                            COLORS['nn_hidden'][j] * intensity) for j in range(3))

            pygame.draw.circle(self.protein_panel, color, (nx + 15, ny + 10), 10)
            pygame.draw.circle(self.protein_panel, COLORS['text_secondary'],
                             (nx + 15, ny + 10), 10, 1)

        # Output layer (4 actions)
        output_x = 360
        output_label = self.font_small.render("Actions", True, COLORS['nn_output'])
        self.protein_panel.blit(output_label, (output_x, nn_start_y))

        action = frame.get('selected_action', 0)

        for i, action_name in enumerate(ACTION_NAMES):
            ay = nn_start_y + 25 + i * 55

            activation = nn_output[i] if i < len(nn_output) else 0
            is_selected = (i == action)

            # Background
            bg_color = COLORS['nn_output'] if is_selected else COLORS['bg_dark']
            pygame.draw.rect(self.protein_panel, bg_color,
                           (output_x, ay, 80, 40), border_radius=6)

            # Border
            border_color = COLORS['text_highlight'] if is_selected else COLORS['text_secondary']
            pygame.draw.rect(self.protein_panel, border_color,
                           (output_x, ay, 80, 40), 2, border_radius=6)

            # Action name
            text_color = COLORS['bg_dark'] if is_selected else COLORS['text_primary']
            action_text = self.font_medium.render(action_name, True, text_color)
            text_rect = action_text.get_rect(center=(output_x + 40, ay + 15))
            self.protein_panel.blit(action_text, text_rect)

            # Activation value
            val_color = COLORS['bg_dark'] if is_selected else COLORS['text_secondary']
            val_text = self.font_small.render(f"{activation:.2f}", True, val_color)
            val_rect = val_text.get_rect(center=(output_x + 40, ay + 32))
            self.protein_panel.blit(val_text, val_rect)

        # Draw particles for active signals
        for particle in self.particles:
            pygame.draw.circle(self.protein_panel, particle.color,
                               (int(particle.x), int(particle.y)), int(particle.size))

    def _draw_protein_node(self, x: int, y: int, pid: str, activation: float,
                           ptype: str, label: str = None) -> None:
        """Draw a single protein node"""
        # Base color by type
        base_colors = {
            'sensor': COLORS['sensor_protein'],
            'hidden': COLORS['hidden_protein'],
            'motor': COLORS['motor_protein']
        }
        base_color = base_colors.get(ptype, COLORS['protein_inactive'])

        # Interpolate between inactive and active color
        inactive = COLORS['protein_inactive']
        active = base_color

        color = tuple(int(inactive[i] * (1 - activation) + active[i] * activation)
                      for i in range(3))

        # Draw glow for active proteins
        if activation > 0.5:
            glow_alpha = int((activation - 0.5) * 2 * 100)
            for r in range(18, 10, -2):
                pygame.draw.circle(self.protein_panel, base_color, (x, y), r)

        # Draw protein circle
        radius = 12 + int(activation * 4)
        pygame.draw.circle(self.protein_panel, color, (x, y), radius)
        pygame.draw.circle(self.protein_panel, COLORS['text_secondary'], (x, y), radius, 1)

        # Draw label
        display_label = label if label else pid
        text = self.font_small.render(display_label, True, COLORS['text_primary'])
        text_rect = text.get_rect(center=(x, y))
        self.protein_panel.blit(text, text_rect)

    def draw_population_panel(self) -> None:
        """Draw the Population Genetics panel (right)"""
        self.population_panel.fill(COLORS['bg_panel'])

        # Title
        title = self.font_large.render("Population Genetics", True, COLORS['text_primary'])
        self.population_panel.blit(title, (20, 10))

        snapshot = self.get_current_snapshot()
        if not snapshot:
            no_data = self.font_medium.render("No data loaded", True, COLORS['text_secondary'])
            self.population_panel.blit(no_data, (PANEL_WIDTH // 2 - 60, PANEL_HEIGHT // 2))
            return

        # Trust score summary
        stats_y = 50
        best_text = self.font_medium.render(
            f"Best Trust: {snapshot.best_trust:.2f}", True, COLORS['trust_high']
        )
        self.population_panel.blit(best_text, (20, stats_y))

        avg_text = self.font_medium.render(
            f"Avg Trust: {snapshot.avg_trust:.2f}", True, COLORS['text_primary']
        )
        self.population_panel.blit(avg_text, (20, stats_y + 25))

        worst_text = self.font_medium.render(
            f"Worst Trust: {snapshot.worst_trust:.2f}", True, COLORS['trust_low']
        )
        self.population_panel.blit(worst_text, (20, stats_y + 50))

        # Draw top organisms
        org_y = 140
        organisms_label = self.font_medium.render("Top Organisms:", True, COLORS['text_secondary'])
        self.population_panel.blit(organisms_label, (20, org_y))

        for i, org in enumerate(snapshot.organisms[:5]):
            y = org_y + 30 + i * 60

            # Organism circle
            trust = org.get('trust_score', 0)
            max_trust = snapshot.best_trust if snapshot.best_trust > 0 else 1
            health = trust / max_trust

            # Color based on health
            color = tuple(int(COLORS['organism_weak'][j] * (1 - health) +
                              COLORS['organism_healthy'][j] * health) for j in range(3))

            # Elite indicator
            if i == 0:
                pygame.draw.circle(self.population_panel, COLORS['organism_elite'], (50, y + 15), 22)

            pygame.draw.circle(self.population_panel, color, (50, y + 15), 18)

            # Organism info
            id_text = self.font_small.render(f"ID: {org.get('organism_id', '?')[:6]}", True,
                                             COLORS['text_primary'])
            self.population_panel.blit(id_text, (80, y))

            trust_text = self.font_small.render(f"Trust: {trust:.1f}", True, COLORS['text_secondary'])
            self.population_panel.blit(trust_text, (80, y + 18))

            score_text = self.font_small.render(f"Best: {org.get('best_score', 0)}", True,
                                                COLORS['text_secondary'])
            self.population_panel.blit(score_text, (80, y + 36))

        # Draw recent genetic events
        events_y = PANEL_HEIGHT - 200
        events_label = self.font_medium.render("Recent Events:", True, COLORS['text_secondary'])
        self.population_panel.blit(events_label, (20, events_y))

        event_colors = {
            'mutation': COLORS['mutation'],
            'crossbreed': COLORS['crossbreed'],
            'cull': COLORS['cull'],
            'birth': COLORS['organism_healthy'],
            'elite_preserve': COLORS['organism_elite']
        }

        for i, event in enumerate(snapshot.genetic_events[-5:]):
            y = events_y + 25 + i * 25
            event_type = event.get('event_type', 'unknown')
            color = event_colors.get(event_type, COLORS['text_secondary'])

            # Event indicator
            pygame.draw.circle(self.population_panel, color, (30, y + 8), 6)

            # Event text
            event_text = self.font_small.render(event_type.replace('_', ' ').title(),
                                                True, COLORS['text_primary'])
            self.population_panel.blit(event_text, (45, y))

    def draw_controls(self) -> None:
        """Draw the control panel"""
        self.control_panel.fill(COLORS['bg_control'])

        # Divider line
        pygame.draw.line(self.control_panel, COLORS['text_secondary'],
                         (0, 0), (WINDOW_WIDTH, 0), 2)

        # Generation counter
        gen_text = self.font_large.render(
            f"Generation: {self.current_generation}",
            True, COLORS['text_primary']
        )
        self.control_panel.blit(gen_text, (20, 15))

        # Playback state
        state_text = "PLAYING" if self.playback_state == PlaybackState.PLAYING else "PAUSED"
        state_color = COLORS['trust_high'] if self.playback_state == PlaybackState.PLAYING else COLORS['text_secondary']
        state_render = self.font_medium.render(state_text, True, state_color)
        self.control_panel.blit(state_render, (20, 55))

        # Speed indicator
        speed_text = self.font_medium.render(f"Speed: {self.playback_speed}x", True, COLORS['text_primary'])
        self.control_panel.blit(speed_text, (150, 55))

        # Timeline
        timeline_x = 300
        timeline_y = 40
        timeline_width = WINDOW_WIDTH - 400
        timeline_height = 20

        # Timeline background
        pygame.draw.rect(self.control_panel, COLORS['timeline_bg'],
                         (timeline_x, timeline_y, timeline_width, timeline_height), border_radius=5)

        # Timeline progress
        if self.data_loaded and self.logger.generation_snapshots:
            progress = self.current_generation / (len(self.logger.generation_snapshots) - 1)
            fill_width = int(progress * timeline_width)
            pygame.draw.rect(self.control_panel, COLORS['timeline_fill'],
                             (timeline_x, timeline_y, fill_width, timeline_height), border_radius=5)

            # Stage markers
            total_gens = len(self.logger.generation_snapshots)
            mid_x = timeline_x + int(0.4 * timeline_width)
            late_x = timeline_x + int(0.8 * timeline_width)

            pygame.draw.line(self.control_panel, COLORS['stage_mid'],
                             (mid_x, timeline_y - 5), (mid_x, timeline_y + timeline_height + 5), 2)
            pygame.draw.line(self.control_panel, COLORS['stage_late'],
                             (late_x, timeline_y - 5), (late_x, timeline_y + timeline_height + 5), 2)

            # Playhead
            playhead_x = timeline_x + fill_width
            pygame.draw.circle(self.control_panel, COLORS['timeline_marker'],
                               (playhead_x, timeline_y + timeline_height // 2), 8)

        # Detail View button
        button_x = WINDOW_WIDTH - 180
        button_y = 15
        button_width = 160
        button_height = 35
        self.detail_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        # Check hover state
        mouse_pos = pygame.mouse.get_pos()
        button_screen_rect = pygame.Rect(button_x, PANEL_HEIGHT + button_y, button_width, button_height)
        is_hover = button_screen_rect.collidepoint(mouse_pos)

        button_color = COLORS['button_hover'] if is_hover else COLORS['button']
        pygame.draw.rect(self.control_panel, button_color, self.detail_button_rect, border_radius=5)
        pygame.draw.rect(self.control_panel, COLORS['hidden_protein'], self.detail_button_rect, 2, border_radius=5)

        button_text = self.font_medium.render("Protein Detail (P)", True, COLORS['text_primary'])
        text_rect = button_text.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
        self.control_panel.blit(button_text, text_rect)

        # Controls help text
        help_text = "SPACE: Play/Pause | LEFT/RIGHT: Seek | UP/DOWN: Speed | 1/2/3: Jump to Stage | R: Reset | P: Protein Detail"
        help_render = self.font_small.render(help_text, True, COLORS['text_secondary'])
        self.control_panel.blit(help_render, (timeline_x, 70))

    def draw(self) -> None:
        """Main draw routine"""
        # Draw panels
        self.draw_snake_panel()
        self.draw_protein_panel()
        self.draw_population_panel()
        self.draw_controls()

        # Blit panels to screen
        self.screen.blit(self.snake_panel, (0, 0))
        self.screen.blit(self.protein_panel, (PANEL_WIDTH, 0))
        self.screen.blit(self.population_panel, (PANEL_WIDTH * 2, 0))
        self.screen.blit(self.control_panel, (0, PANEL_HEIGHT))

        # Draw panel dividers
        pygame.draw.line(self.screen, COLORS['text_secondary'],
                         (PANEL_WIDTH, 0), (PANEL_WIDTH, PANEL_HEIGHT), 2)
        pygame.draw.line(self.screen, COLORS['text_secondary'],
                         (PANEL_WIDTH * 2, 0), (PANEL_WIDTH * 2, PANEL_HEIGHT), 2)

        pygame.display.flip()

    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def run(self) -> None:
        """Main application loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main():
    """Run the visualizer"""
    import sys

    data_dir = "simulation_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=" * 60)
    print("GENREG Evolution Visualizer")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"\nData directory '{data_dir}' not found!")
        print("Running pre-computation first...")
        print()

        # Run simulation
        from simulation import GENREGSimulation
        sim = GENREGSimulation()
        sim.run_full_simulation()
        print()

    print("Starting visualizer...")
    app = VisualizerApp(data_dir)
    app.run()


if __name__ == "__main__":
    main()
