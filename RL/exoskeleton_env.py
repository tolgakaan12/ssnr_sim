"""
Exoskeleton Environment for Training Assistive Agent

This environment integrates a pre-trained human agent with an exoskeleton agent
that learns to provide optimal assistance for elbow control tasks.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import copy

from human_elbow_env import HumanElbowEnv
from intent_signals import IntentSignalProcessor, IntentConfig


class ExoskeletonEnv(gym.Env):
    """
    Environment for training exoskeleton assistance agent.
    
    The human agent is frozen and treated as part of the environment.
    The exoskeleton agent observes human intent and provides torque assistance.
    """
    
    def __init__(self,
                 human_model_path: str,
                 human_vec_normalize_path: Optional[str] = None,
                 proprioception_enabled: bool = True,
                 perturb_scale: float = 5.0,
                 max_assistance_torque: float = 10.0,
                 assistance_penalty_weight: float = 0.01,
                 interference_penalty_weight: float = 0.1,
                 intent_config: Optional[IntentConfig] = None,
                 episode_length: int = 1500,
                 seed: Optional[int] = None):
        """
        Initialize exoskeleton environment.
        
        Args:
            human_model_path: Path to trained human agent model
            human_vec_normalize_path: Path to VecNormalize stats for human agent
            proprioception_enabled: Whether human was trained with proprioception
            perturb_scale: Perturbation scale used in human training
            max_assistance_torque: Maximum torque the exoskeleton can apply
            assistance_penalty_weight: Weight for penalizing assistance effort
            interference_penalty_weight: Weight for penalizing interference
            intent_config: Configuration for intent signal processing
            episode_length: Maximum episode length
            seed: Random seed
        """
        super().__init__()
        
        # Store configuration
        self.proprioception_enabled = proprioception_enabled
        self.perturb_scale = perturb_scale
        self.max_assistance_torque = max_assistance_torque
        self.assistance_penalty_weight = assistance_penalty_weight
        self.interference_penalty_weight = interference_penalty_weight
        self.episode_length = episode_length
        
        # Load human agent
        self.human_agent = PPO.load(human_model_path)
        self.human_vec_normalize = None
        if human_vec_normalize_path:
            self.human_vec_normalize = VecNormalize.load(human_vec_normalize_path, venv=None)
        
        # Create human environment
        self.human_env = HumanElbowEnv(
            perturb_scale=perturb_scale,
            proprioception_enabled=proprioception_enabled,
            continuous_perturbations=True,
            seed=seed
        )
        
        # Intent signal processor
        self.intent_processor = IntentSignalProcessor(
            config=intent_config or IntentConfig(),
            sampling_rate=200.0  # Assuming 200Hz simulation
        )
        
        # Define action space for exoskeleton (assistance torque)
        self.action_space = spaces.Box(
            low=-max_assistance_torque,
            high=max_assistance_torque,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space
        self._setup_observation_space()
        
        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self.human_obs = None
        self.human_action = None
        self.intent_signals = None
        self.prev_angle_error = None
        self.total_assistance = 0.0
        self.total_interference = 0.0
        
        # Performance tracking
        self.human_alone_performance = []
        self.with_assistance_performance = []
    
    def _setup_observation_space(self):
        """Setup the observation space for the exoskeleton agent."""
        # Get a sample observation to determine dimensions
        sample_human_obs = self.human_env.reset()
        sample_intent = self._get_sample_intent()
        
        # Human state dimensions
        human_obs_flat = self._flatten_human_obs(sample_human_obs)
        human_state_dim = len(human_obs_flat)
        
        # Intent signal dimensions
        intent_vector = self.intent_processor.get_intent_vector(sample_intent)
        intent_dim = len(intent_vector)
        
        # Additional exoskeleton-specific observations
        exo_specific_dim = 4  # prev_assistance, assistance_effectiveness, time_in_episode, target_progress
        
        total_dim = human_state_dim + intent_dim + exo_specific_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        # Store dimensions for later use
        self.human_state_dim = human_state_dim
        self.intent_dim = intent_dim
        self.exo_specific_dim = exo_specific_dim
    
    def _get_sample_intent(self) -> Dict[str, np.ndarray]:
        """Get a sample intent signal for observation space setup."""
        return {
            'muscle_intent': np.zeros(6),
            'joint_velocity': np.array([0.0]),
            'joint_acceleration': np.array([0.0]),
            'cocontraction': np.array([0.0]),
            'position_error': np.array([0.0]),
            'velocity_error': np.array([0.0]),
            'movement_phase': np.array([0.0]),
            'effort_level': np.array([0.0])
        }
    
    def _flatten_human_obs(self, human_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten human observation dictionary to vector."""
        obs_vector = []
        
        # Define order of observations
        obs_keys = [
            'pelvis_pos', 'body_qpos', 'body_qvel', 
            'target_angle', 'current_angle', 'angle_error'
        ]
        
        # Add proprioceptive signals if enabled
        if self.proprioception_enabled:
            obs_keys.extend(['external_force', 'joint_torque', 'gravity_torque'])
        
        for key in obs_keys:
            if key in human_obs:
                value = human_obs[key]
                if hasattr(value, '__len__') and len(value) > 1:
                    obs_vector.extend(value.flatten())
                else:
                    obs_vector.append(float(value))
        
        return np.array(obs_vector, dtype=np.float32)
    
    def _get_human_action(self, human_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get action from the human agent."""
        # Normalize human observation if we have the normalizer
        if self.human_vec_normalize:
            # Convert to format expected by VecNormalize
            obs_array = self._flatten_human_obs(human_obs).reshape(1, -1)
            normalized_obs = self.human_vec_normalize.normalize_obs(obs_array)[0]
            # Convert back to dict format expected by human agent
            human_obs_normalized = self._reconstruct_human_obs(normalized_obs, human_obs)
        else:
            human_obs_normalized = human_obs
        
        # Get human action (deterministic for consistency)
        human_action, _ = self.human_agent.predict(human_obs_normalized, deterministic=True)
        
        return human_action
    
    def _reconstruct_human_obs(self, obs_vector: np.ndarray, original_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Reconstruct observation dictionary from flattened vector."""
        # For simplicity, return the original obs structure
        # In practice, might need more sophisticated reconstruction
        return original_obs
    
    def _extract_intent_signals(self, 
                               human_action: np.ndarray,
                               human_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract intent signals from human behavior."""
        # Get required information for intent extraction
        joint_state = {
            'qvel': human_obs.get('body_qvel', np.array([0.0]))
        }
        
        muscle_activations = human_obs.get('act', human_action)
        target_angle = human_obs.get('target_angle', np.array([1.0]))[0]
        current_angle = human_obs.get('current_angle', np.array([0.0]))[0]
        
        # Extract intent using processor
        intent = self.intent_processor.extract_intent(
            human_action=human_action,
            joint_state=joint_state,
            muscle_activations=muscle_activations,
            target_angle=target_angle,
            current_angle=current_angle
        )
        
        return intent
    
    def _build_exoskeleton_observation(self, 
                                     human_obs: Dict[str, np.ndarray],
                                     intent_signals: Dict[str, np.ndarray],
                                     prev_assistance: float) -> np.ndarray:
        """Build observation for the exoskeleton agent."""
        # Human state
        human_state = self._flatten_human_obs(human_obs)
        
        # Intent signals
        intent_vector = self.intent_processor.get_intent_vector(intent_signals)
        
        # Exoskeleton-specific observations
        assistance_effectiveness = self._calculate_assistance_effectiveness()
        time_progress = self.step_count / self.episode_length
        target_progress = self._calculate_target_progress(human_obs)
        
        exo_specific = np.array([
            prev_assistance,
            assistance_effectiveness,
            time_progress,
            target_progress
        ], dtype=np.float32)
        
        # Concatenate all observations
        full_obs = np.concatenate([human_state, intent_vector, exo_specific])
        
        return full_obs.astype(np.float32)
    
    def _calculate_assistance_effectiveness(self) -> float:
        """Calculate how effective the assistance has been."""
        if self.prev_angle_error is None or len(self.human_alone_performance) == 0:
            return 0.0
        
        # Compare current performance to baseline
        current_error = abs(self.human_obs.get('angle_error', np.array([0.0]))[0])
        baseline_error = np.mean(self.human_alone_performance[-10:]) if len(self.human_alone_performance) >= 10 else current_error
        
        # Effectiveness is reduction in error
        effectiveness = max(0.0, (baseline_error - current_error) / (baseline_error + 1e-6))
        
        return float(effectiveness)
    
    def _calculate_target_progress(self, human_obs: Dict[str, np.ndarray]) -> float:
        """Calculate progress towards target."""
        if 'angle_error' not in human_obs:
            return 0.0
        
        angle_error = abs(human_obs['angle_error'][0])
        max_error = 2.27  # Maximum possible error (full range)
        
        progress = 1.0 - (angle_error / max_error)
        return float(np.clip(progress, 0.0, 1.0))
    
    def _calculate_exoskeleton_reward(self,
                                    human_obs: Dict[str, np.ndarray],
                                    exo_action: np.ndarray,
                                    human_reward: float) -> float:
        """Calculate reward for the exoskeleton agent."""
        # Base reward: improvement in human performance
        angle_error = abs(human_obs.get('angle_error', np.array([0.0]))[0])
        
        # Target achievement reward
        target_reward = np.exp(-5.0 * angle_error)
        
        # Improvement reward (compared to previous step)
        improvement_reward = 0.0
        if self.prev_angle_error is not None:
            error_reduction = self.prev_angle_error - angle_error
            improvement_reward = 5.0 * error_reduction  # Reward for reducing error
        
        # Assistance effort penalty
        assistance_magnitude = abs(exo_action[0])
        effort_penalty = -self.assistance_penalty_weight * assistance_magnitude
        
        # Interference penalty (when assistance opposes human intent)
        interference_penalty = self._calculate_interference_penalty(exo_action)
        
        # Stability reward (smooth assistance)
        stability_reward = self._calculate_stability_reward(exo_action)
        
        # Total reward
        total_reward = (
            target_reward +
            improvement_reward +
            effort_penalty +
            interference_penalty +
            stability_reward
        )
        
        self.prev_angle_error = angle_error
        
        return total_reward
    
    def _calculate_interference_penalty(self, exo_action: np.ndarray) -> float:
        """Calculate penalty for interfering with human intent."""
        if self.intent_signals is None:
            return 0.0
        
        # Get human's intended movement direction
        velocity_error = self.intent_signals.get('velocity_error', np.array([0.0]))[0]
        position_error = self.intent_signals.get('position_error', np.array([0.0]))[0]
        
        # Human's desired assistance direction
        desired_direction = np.sign(position_error)
        assistance_direction = np.sign(exo_action[0])
        
        # Penalty if assistance opposes desired direction
        if abs(position_error) > 0.1 and desired_direction * assistance_direction < 0:
            interference = self.interference_penalty_weight * abs(exo_action[0])
            self.total_interference += interference
            return -interference
        
        return 0.0
    
    def _calculate_stability_reward(self, exo_action: np.ndarray) -> float:
        """Reward smooth, stable assistance."""
        if not hasattr(self, 'prev_exo_action'):
            self.prev_exo_action = exo_action
            return 0.0
        
        # Penalize large changes in assistance
        action_change = abs(exo_action[0] - self.prev_exo_action[0])
        stability_reward = -0.1 * action_change
        
        self.prev_exo_action = exo_action
        
        return stability_reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        # Reset human environment
        self.human_obs = self.human_env.reset(seed=seed)
        
        # Reset intent processor
        self.intent_processor.reset()
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1
        self.prev_angle_error = None
        self.total_assistance = 0.0
        self.total_interference = 0.0
        
        # Get initial human action and intent
        self.human_action = self._get_human_action(self.human_obs)
        self.intent_signals = self._extract_intent_signals(self.human_action, self.human_obs)
        
        # Build initial observation for exoskeleton
        exo_obs = self._build_exoskeleton_observation(
            self.human_obs, self.intent_signals, prev_assistance=0.0
        )
        
        info = {
            'episode_count': self.episode_count,
            'human_target': self.human_obs.get('target_angle', np.array([0.0]))[0]
        }
        
        return exo_obs, info
    
    def step(self, exo_action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step of the environment."""
        # Clip exoskeleton action to valid range
        exo_action = np.clip(exo_action, -self.max_assistance_torque, self.max_assistance_torque)
        
        # Apply exoskeleton assistance to human environment
        assistance_torque = exo_action[0]
        elbow_dof = self.human_env.elbow_dof
        self.human_env.sim.data.qfrc_applied[elbow_dof] += assistance_torque
        
        # Execute human action in environment
        self.human_obs, human_reward, done, human_info = self.human_env.step(self.human_action)
        
        # Track assistance
        self.total_assistance += abs(assistance_torque)
        
        # Get new human action and intent for next step
        if not done:
            self.human_action = self._get_human_action(self.human_obs)
            self.intent_signals = self._extract_intent_signals(self.human_action, self.human_obs)
        
        # Calculate exoskeleton reward
        exo_reward = self._calculate_exoskeleton_reward(self.human_obs, exo_action, human_reward)
        
        # Build observation for next step
        exo_obs = self._build_exoskeleton_observation(
            self.human_obs, self.intent_signals, prev_assistance=assistance_torque
        )
        
        # Update step counter
        self.step_count += 1
        
        # Check if episode should end
        if self.step_count >= self.episode_length:
            done = True
        
        # Prepare info
        info = {
            'human_reward': human_reward,
            'assistance_torque': assistance_torque,
            'total_assistance': self.total_assistance,
            'total_interference': self.total_interference,
            'angle_error': abs(self.human_obs.get('angle_error', np.array([0.0]))[0]),
            'target_angle': self.human_obs.get('target_angle', np.array([0.0]))[0],
            'current_angle': self.human_obs.get('current_angle', np.array([0.0]))[0]
        }
        
        return exo_obs, exo_reward, done, False, info
    
    def enable_intent_corruption(self, **corruption_params):
        """Enable intent signal corruption for robustness testing."""
        self.intent_processor.enable_corruption(**corruption_params)
    
    def disable_intent_corruption(self):
        """Disable intent signal corruption."""
        self.intent_processor.disable_corruption()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for evaluation."""
        return {
            'total_assistance': self.total_assistance,
            'total_interference': self.total_interference,
            'assistance_efficiency': self.total_assistance / max(1, self.step_count),
            'interference_rate': self.total_interference / max(1, self.total_assistance)
        }