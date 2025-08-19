"""
Visualization Tools for Human-Exoskeleton System

Interactive visualization and analysis tools for evaluating the human-exoskeleton 
training system with real-time plotting and performance comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize
import argparse
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import deque
from dataclasses import dataclass
import json

from human_elbow_env import HumanElbowEnv
from exoskeleton_env import ExoskeletonEnv
from intent_signals import IntentConfig


@dataclass
class VisualizationData:
    """Container for visualization data."""
    time_points: List[float] = None
    angle_targets: List[float] = None
    angle_actuals: List[float] = None
    angle_errors: List[float] = None
    assistance_torques: List[float] = None
    muscle_activations: List[List[float]] = None
    intent_signals: List[Dict[str, float]] = None
    proprioception_signals: List[Dict[str, float]] = None
    rewards: List[float] = None
    
    def __post_init__(self):
        if self.time_points is None:
            self.time_points = []
        if self.angle_targets is None:
            self.angle_targets = []
        if self.angle_actuals is None:
            self.angle_actuals = []
        if self.angle_errors is None:
            self.angle_errors = []
        if self.assistance_torques is None:
            self.assistance_torques = []
        if self.muscle_activations is None:
            self.muscle_activations = []
        if self.intent_signals is None:
            self.intent_signals = []
        if self.proprioception_signals is None:
            self.proprioception_signals = []
        if self.rewards is None:
            self.rewards = []
    
    def add_data_point(self, 
                      time: float,
                      target_angle: float,
                      actual_angle: float,
                      assistance_torque: float = 0.0,
                      muscle_acts: Optional[np.ndarray] = None,
                      intent: Optional[Dict[str, float]] = None,
                      proprio: Optional[Dict[str, float]] = None,
                      reward: float = 0.0):
        """Add a new data point to the visualization data."""
        self.time_points.append(time)
        self.angle_targets.append(target_angle)
        self.angle_actuals.append(actual_angle)
        self.angle_errors.append(abs(target_angle - actual_angle))
        self.assistance_torques.append(assistance_torque)
        self.rewards.append(reward)
        
        if muscle_acts is not None:
            self.muscle_activations.append(muscle_acts.tolist())
        else:
            self.muscle_activations.append([0.0] * 6)
        
        if intent is not None:
            self.intent_signals.append(intent.copy())
        else:
            self.intent_signals.append({})
        
        if proprio is not None:
            self.proprioception_signals.append(proprio.copy())
        else:
            self.proprioception_signals.append({})
    
    def clear(self):
        """Clear all data."""
        self.time_points.clear()
        self.angle_targets.clear()
        self.angle_actuals.clear()
        self.angle_errors.clear()
        self.assistance_torques.clear()
        self.muscle_activations.clear()
        self.intent_signals.clear()
        self.proprioception_signals.clear()
        self.rewards.clear()


class HumanExoskeletonVisualizer:
    """Interactive visualizer for human-exoskeleton system."""
    
    def __init__(self,
                 human_model_path: str,
                 exoskeleton_model_path: Optional[str] = None,
                 human_vec_normalize_path: Optional[str] = None,
                 proprioception_enabled: bool = True,
                 max_history: int = 1000):
        """
        Initialize the visualizer.
        
        Args:
            human_model_path: Path to trained human agent
            exoskeleton_model_path: Path to trained exoskeleton agent (optional)
            human_vec_normalize_path: Path to VecNormalize stats
            proprioception_enabled: Whether proprioception is enabled
            max_history: Maximum number of data points to keep
        """
        self.human_model_path = human_model_path
        self.exoskeleton_model_path = exoskeleton_model_path
        self.human_vec_normalize_path = human_vec_normalize_path
        self.proprioception_enabled = proprioception_enabled
        self.max_history = max_history
        
        # Load models
        self.human_agent = PPO.load(human_model_path)
        self.exoskeleton_agent = None
        if exoskeleton_model_path:
            try:
                self.exoskeleton_agent = PPO.load(exoskeleton_model_path)
            except:
                try:
                    self.exoskeleton_agent = SAC.load(exoskeleton_model_path)
                except:
                    print(f"Warning: Could not load exoskeleton model from {exoskeleton_model_path}")
        
        # Create environments
        self.human_env = HumanElbowEnv(
            proprioception_enabled=proprioception_enabled,
            continuous_perturbations=True,
            perturb_scale=5.0
        )
        
        if self.exoskeleton_agent:
            self.exoskeleton_env = ExoskeletonEnv(
                human_model_path=human_model_path,
                human_vec_normalize_path=human_vec_normalize_path,
                proprioception_enabled=proprioception_enabled
            )
        
        # Visualization data
        self.human_only_data = VisualizationData()
        self.with_exoskeleton_data = VisualizationData()
        
        # Control state
        self.show_exoskeleton = True
        self.show_proprioception = True
        self.show_intent_signals = True
        self.paused = False
        self.current_time = 0.0
        
        # Setup matplotlib figure
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.setup_plots()
    
    def setup_plots(self):
        """Setup the matplotlib plotting interface."""
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 12))
        self.fig.suptitle('Human-Exoskeleton System Visualization', fontsize=16)
        
        # Angle tracking plot
        ax = self.axes[0, 0]
        ax.set_title('Elbow Angle Tracking')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        self.lines['target_angle'], = ax.plot([], [], 'k--', label='Target', linewidth=2)
        self.lines['human_angle'], = ax.plot([], [], 'b-', label='Human Only', linewidth=2)
        if self.exoskeleton_agent:
            self.lines['exo_angle'], = ax.plot([], [], 'r-', label='With Exoskeleton', linewidth=2)
        ax.legend()
        ax.grid(True)
        
        # Angle error plot
        ax = self.axes[0, 1]
        ax.set_title('Angle Tracking Error')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        self.lines['human_error'], = ax.plot([], [], 'b-', label='Human Only', linewidth=2)
        if self.exoskeleton_agent:
            self.lines['exo_error'], = ax.plot([], [], 'r-', label='With Exoskeleton', linewidth=2)
        ax.legend()
        ax.grid(True)
        
        # Muscle activation plot
        ax = self.axes[1, 0]
        ax.set_title('Muscle Activations')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Activation')
        muscle_names = ['Muscle 1', 'Muscle 2', 'Muscle 3', 'Muscle 4', 'Muscle 5', 'Muscle 6']
        colors = plt.cm.tab10(np.linspace(0, 1, 6))
        self.muscle_lines = []
        for i, (name, color) in enumerate(zip(muscle_names, colors)):
            line, = ax.plot([], [], color=color, label=name, alpha=0.7)
            self.muscle_lines.append(line)
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 1)
        
        # Assistance torque plot
        ax = self.axes[1, 1]
        ax.set_title('Exoskeleton Assistance Torque')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        if self.exoskeleton_agent:
            self.lines['assistance'], = ax.plot([], [], 'g-', label='Assistance Torque', linewidth=2)
        ax.grid(True)
        
        # Intent signals plot
        ax = self.axes[2, 0]
        ax.set_title('Intent Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal Value')
        self.lines['velocity_intent'], = ax.plot([], [], 'c-', label='Velocity Intent', alpha=0.7)
        self.lines['effort_level'], = ax.plot([], [], 'm-', label='Effort Level', alpha=0.7)
        self.lines['cocontraction'], = ax.plot([], [], 'y-', label='Co-contraction', alpha=0.7)
        ax.legend()
        ax.grid(True)
        
        # Proprioception signals plot
        ax = self.axes[2, 1]
        ax.set_title('Proprioceptive Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force/Torque (N/Nm)')
        if self.proprioception_enabled:
            self.lines['external_force'], = ax.plot([], [], 'orange', label='External Force', alpha=0.7)
            self.lines['joint_torque'], = ax.plot([], [], 'purple', label='Joint Torque', alpha=0.7)
        ax.legend()
        ax.grid(True)
        
        # Add control buttons
        self.setup_controls()
        
        plt.tight_layout()
    
    def setup_controls(self):
        """Setup interactive controls."""
        # Add buttons for control
        ax_button_pause = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.button_pause = Button(ax_button_pause, 'Pause')
        self.button_pause.on_clicked(self.toggle_pause)
        
        ax_button_reset = plt.axes([0.12, 0.95, 0.08, 0.04])
        self.button_reset = Button(ax_button_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_visualization)
        
        # Checkboxes for display options
        ax_check = plt.axes([0.85, 0.90, 0.12, 0.08])
        labels = ['Exoskeleton', 'Proprioception', 'Intent Signals']
        visibility = [self.show_exoskeleton, self.show_proprioception, self.show_intent_signals]
        self.checkboxes = CheckButtons(ax_check, labels, visibility)
        self.checkboxes.on_clicked(self.toggle_display)
    
    def toggle_pause(self, event):
        """Toggle pause state."""
        self.paused = not self.paused
        self.button_pause.label.set_text('Resume' if self.paused else 'Pause')
    
    def reset_visualization(self, event):
        """Reset the visualization."""
        self.human_only_data.clear()
        self.with_exoskeleton_data.clear()
        self.current_time = 0.0
        self.update_plots()
    
    def toggle_display(self, label):
        """Toggle display options."""
        if label == 'Exoskeleton':
            self.show_exoskeleton = not self.show_exoskeleton
        elif label == 'Proprioception':
            self.show_proprioception = not self.show_proprioception
        elif label == 'Intent Signals':
            self.show_intent_signals = not self.show_intent_signals
        self.update_plots()
    
    def run_comparison(self, episode_length: int = 1000):
        """Run a comparison between human-only and human+exoskeleton performance."""
        print("Running human-only episode...")
        self.run_human_only_episode(episode_length)
        
        if self.exoskeleton_agent:
            print("Running human+exoskeleton episode...")
            self.run_exoskeleton_episode(episode_length)
        
        print("Comparison complete. Displaying results...")
        self.update_plots()
        plt.show()
    
    def run_human_only_episode(self, episode_length: int):
        """Run an episode with human agent only."""
        obs = self.human_env.reset()
        self.current_time = 0.0
        dt = 0.005  # Assuming 200Hz simulation
        
        for step in range(episode_length):
            # Get human action
            action, _ = self.human_agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = self.human_env.step(action)
            
            # Extract data
            target_angle = obs.get('target_angle', [0.0])[0]
            current_angle = obs.get('current_angle', [0.0])[0]
            muscle_acts = obs.get('act', action)
            
            # Extract proprioception signals
            proprio_signals = {}
            if self.proprioception_enabled:
                for key in ['external_force', 'joint_torque', 'gravity_torque']:
                    if key in obs:
                        proprio_signals[key] = obs[key][0] if hasattr(obs[key], '__len__') else obs[key]
            
            # Add data point
            self.human_only_data.add_data_point(
                time=self.current_time,
                target_angle=target_angle,
                actual_angle=current_angle,
                muscle_acts=muscle_acts,
                proprio=proprio_signals,
                reward=reward
            )
            
            self.current_time += dt
            
            if done:
                break
    
    def run_exoskeleton_episode(self, episode_length: int):
        """Run an episode with exoskeleton assistance."""
        if not self.exoskeleton_agent:
            return
        
        obs, _ = self.exoskeleton_env.reset()
        self.current_time = 0.0
        dt = 0.005
        
        for step in range(episode_length):
            # Get exoskeleton action
            exo_action, _ = self.exoskeleton_agent.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, _, info = self.exoskeleton_env.step(exo_action)
            
            # Extract data
            target_angle = info.get('target_angle', 0.0)
            current_angle = info.get('current_angle', 0.0)
            assistance_torque = info.get('assistance_torque', 0.0)
            
            # Extract human muscle activations (from exoskeleton env)
            human_obs = self.exoskeleton_env.human_obs
            muscle_acts = human_obs.get('act', np.zeros(6))
            
            # Extract intent signals
            intent_signals = {}
            if self.exoskeleton_env.intent_signals:
                intent = self.exoskeleton_env.intent_signals
                intent_signals = {
                    'velocity_intent': intent.get('velocity_error', np.array([0.0]))[0],
                    'effort_level': intent.get('effort_level', np.array([0.0]))[0],
                    'cocontraction': intent.get('cocontraction', np.array([0.0]))[0]
                }
            
            # Extract proprioception
            proprio_signals = {}
            if self.proprioception_enabled and human_obs:
                for key in ['external_force', 'joint_torque']:
                    if key in human_obs:
                        value = human_obs[key]
                        proprio_signals[key] = value[0] if hasattr(value, '__len__') else value
            
            # Add data point
            self.with_exoskeleton_data.add_data_point(
                time=self.current_time,
                target_angle=target_angle,
                actual_angle=current_angle,
                assistance_torque=assistance_torque,
                muscle_acts=muscle_acts,
                intent=intent_signals,
                proprio=proprio_signals,
                reward=reward
            )
            
            self.current_time += dt
            
            if done:
                break
    
    def update_plots(self):
        """Update all plots with current data."""
        # Update angle tracking
        if self.human_only_data.time_points:
            self.lines['target_angle'].set_data(
                self.human_only_data.time_points, 
                self.human_only_data.angle_targets
            )
            self.lines['human_angle'].set_data(
                self.human_only_data.time_points, 
                self.human_only_data.angle_actuals
            )
        
        if self.show_exoskeleton and self.with_exoskeleton_data.time_points and 'exo_angle' in self.lines:
            self.lines['exo_angle'].set_data(
                self.with_exoskeleton_data.time_points,
                self.with_exoskeleton_data.angle_actuals
            )
        
        # Update error plots
        if self.human_only_data.time_points:
            self.lines['human_error'].set_data(
                self.human_only_data.time_points,
                self.human_only_data.angle_errors
            )
        
        if self.show_exoskeleton and self.with_exoskeleton_data.time_points and 'exo_error' in self.lines:
            self.lines['exo_error'].set_data(
                self.with_exoskeleton_data.time_points,
                self.with_exoskeleton_data.angle_errors
            )
        
        # Update muscle activations (use human-only data)
        if self.human_only_data.time_points and self.human_only_data.muscle_activations:
            muscle_data = np.array(self.human_only_data.muscle_activations).T
            for i, line in enumerate(self.muscle_lines):
                if i < len(muscle_data):
                    line.set_data(self.human_only_data.time_points, muscle_data[i])
        
        # Update assistance torque
        if (self.show_exoskeleton and self.with_exoskeleton_data.time_points and 
            'assistance' in self.lines):
            self.lines['assistance'].set_data(
                self.with_exoskeleton_data.time_points,
                self.with_exoskeleton_data.assistance_torques
            )
        
        # Update intent signals
        if (self.show_intent_signals and self.with_exoskeleton_data.time_points and 
            self.with_exoskeleton_data.intent_signals):
            
            velocity_intent = [s.get('velocity_intent', 0.0) for s in self.with_exoskeleton_data.intent_signals]
            effort_level = [s.get('effort_level', 0.0) for s in self.with_exoskeleton_data.intent_signals]
            cocontraction = [s.get('cocontraction', 0.0) for s in self.with_exoskeleton_data.intent_signals]
            
            self.lines['velocity_intent'].set_data(self.with_exoskeleton_data.time_points, velocity_intent)
            self.lines['effort_level'].set_data(self.with_exoskeleton_data.time_points, effort_level)
            self.lines['cocontraction'].set_data(self.with_exoskeleton_data.time_points, cocontraction)
        
        # Update proprioception signals
        if (self.show_proprioception and self.proprioception_enabled and 
            self.human_only_data.time_points and self.human_only_data.proprioception_signals):
            
            external_force = [s.get('external_force', 0.0) for s in self.human_only_data.proprioception_signals]
            joint_torque = [s.get('joint_torque', 0.0) for s in self.human_only_data.proprioception_signals]
            
            if 'external_force' in self.lines:
                self.lines['external_force'].set_data(self.human_only_data.time_points, external_force)
            if 'joint_torque' in self.lines:
                self.lines['joint_torque'].set_data(self.human_only_data.time_points, joint_torque)
        
        # Auto-scale axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.relim()
                ax.autoscale_view()
        
        self.fig.canvas.draw()
    
    def save_results(self, filepath: str):
        """Save visualization results to file."""
        results = {
            'human_only': {
                'time': self.human_only_data.time_points,
                'angle_errors': self.human_only_data.angle_errors,
                'rewards': self.human_only_data.rewards
            }
        }
        
        if self.with_exoskeleton_data.time_points:
            results['with_exoskeleton'] = {
                'time': self.with_exoskeleton_data.time_points,
                'angle_errors': self.with_exoskeleton_data.angle_errors,
                'assistance_torques': self.with_exoskeleton_data.assistance_torques,
                'rewards': self.with_exoskeleton_data.rewards
            }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_performance_summary(self):
        """Print performance comparison summary."""
        if not self.human_only_data.angle_errors:
            print("No human-only data available")
            return
        
        human_mean_error = np.mean(self.human_only_data.angle_errors)
        human_std_error = np.std(self.human_only_data.angle_errors)
        human_mean_reward = np.mean(self.human_only_data.rewards)
        
        print("\n=== Performance Summary ===")
        print(f"Human Only:")
        print(f"  Mean Angle Error: {human_mean_error:.4f} ± {human_std_error:.4f} rad")
        print(f"  Mean Reward: {human_mean_reward:.4f}")
        
        if self.with_exoskeleton_data.angle_errors:
            exo_mean_error = np.mean(self.with_exoskeleton_data.angle_errors)
            exo_std_error = np.std(self.with_exoskeleton_data.angle_errors)
            exo_mean_reward = np.mean(self.with_exoskeleton_data.rewards)
            mean_assistance = np.mean(np.abs(self.with_exoskeleton_data.assistance_torques))
            
            print(f"\nWith Exoskeleton:")
            print(f"  Mean Angle Error: {exo_mean_error:.4f} ± {exo_std_error:.4f} rad")
            print(f"  Mean Reward: {exo_mean_reward:.4f}")
            print(f"  Mean Assistance Torque: {mean_assistance:.4f} Nm")
            
            improvement = (human_mean_error - exo_mean_error) / human_mean_error * 100
            print(f"\nImprovement: {improvement:.1f}% reduction in tracking error")


def main():
    """Main function for running the visualizer."""
    parser = argparse.ArgumentParser(description='Visualize human-exoskeleton system')
    
    parser.add_argument('--human-model', type=str, required=True,
                       help='Path to trained human agent model')
    parser.add_argument('--exoskeleton-model', type=str, default=None,
                       help='Path to trained exoskeleton agent model')
    parser.add_argument('--human-normalize', type=str, default=None,
                       help='Path to VecNormalize stats for human agent')
    parser.add_argument('--proprioception', action='store_true', default=True,
                       help='Human was trained with proprioception')
    parser.add_argument('--no-proprioception', action='store_true', default=False,
                       help='Human was trained without proprioception')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='Length of evaluation episodes')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save visualization results')
    
    args = parser.parse_args()
    
    proprioception_enabled = args.proprioception and not args.no_proprioception
    
    # Create visualizer
    visualizer = HumanExoskeletonVisualizer(
        human_model_path=args.human_model,
        exoskeleton_model_path=args.exoskeleton_model,
        human_vec_normalize_path=args.human_normalize,
        proprioception_enabled=proprioception_enabled
    )
    
    # Run comparison
    visualizer.run_comparison(episode_length=args.episode_length)
    
    # Print summary
    visualizer.print_performance_summary()
    
    # Save results if requested
    if args.save_results:
        visualizer.save_results(args.save_results)


if __name__ == '__main__':
    main()