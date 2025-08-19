"""
Intent Signal Processing Module

This module handles extraction, processing, and corruption of human intent signals
for the exoskeleton training system.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.signal import butter, lfilter, savgol_filter
from dataclasses import dataclass
import copy


@dataclass
class IntentConfig:
    """Configuration for intent signal processing."""
    muscle_weight: float = 1.0
    velocity_weight: float = 0.5
    acceleration_weight: float = 0.3
    cocontraction_weight: float = 0.2
    target_error_weight: float = 0.8
    
    # Filtering parameters
    enable_filtering: bool = True
    filter_cutoff: float = 10.0  # Hz
    filter_order: int = 2
    
    # Smoothing parameters
    enable_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_poly_order: int = 2
    
    # Corruption parameters
    noise_std: float = 0.0
    delay_steps: int = 0
    dropout_rate: float = 0.0
    bias_drift: float = 0.0


class IntentSignalProcessor:
    """
    Processes human intent signals for exoskeleton training.
    
    Extracts meaningful intent from:
    - Muscle activation patterns
    - Joint kinematics (position, velocity, acceleration)
    - Co-contraction levels
    - Target tracking error
    
    Supports signal corruption for robustness testing.
    """
    
    def __init__(self, config: IntentConfig = None, sampling_rate: float = 200.0):
        """
        Initialize intent signal processor.
        
        Args:
            config: Configuration for signal processing
            sampling_rate: Sampling rate of the system (Hz)
        """
        self.config = config or IntentConfig()
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # History for filtering and smoothing
        self.signal_history: List[Dict[str, np.ndarray]] = []
        self.max_history_length = max(50, self.config.delay_steps + 10)
        
        # Previous values for acceleration calculation
        self.prev_velocity = None
        self.prev_acceleration = None
        
        # Filter design
        if self.config.enable_filtering:
            nyquist = 0.5 * sampling_rate
            normal_cutoff = self.config.filter_cutoff / nyquist
            self.filter_b, self.filter_a = butter(
                self.config.filter_order, normal_cutoff, btype='low', analog=False
            )
        
        # Corruption state
        self.corruption_enabled = False
        self.noise_generator = np.random.RandomState(42)
        self.bias_accumulator = 0.0
    
    def extract_intent(self, 
                      human_action: np.ndarray,
                      joint_state: Dict[str, np.ndarray],
                      muscle_activations: np.ndarray,
                      target_angle: float,
                      current_angle: float) -> Dict[str, np.ndarray]:
        """
        Extract intent signals from human behavior.
        
        Args:
            human_action: Raw muscle activation commands from human agent
            joint_state: Dictionary containing joint position and velocity
            muscle_activations: Current muscle activation levels
            target_angle: Target elbow angle
            current_angle: Current elbow angle
            
        Returns:
            Dictionary of processed intent signals
        """
        # Raw intent components
        raw_intent = self._extract_raw_intent(
            human_action, joint_state, muscle_activations, target_angle, current_angle
        )
        
        # Apply filtering and smoothing
        processed_intent = self._process_signals(raw_intent)
        
        # Apply corruption if enabled
        if self.corruption_enabled:
            processed_intent = self._apply_corruption(processed_intent)
        
        # Store in history
        self._update_history(processed_intent)
        
        return processed_intent
    
    def _extract_raw_intent(self,
                           human_action: np.ndarray,
                           joint_state: Dict[str, np.ndarray],
                           muscle_activations: np.ndarray,
                           target_angle: float,
                           current_angle: float) -> Dict[str, np.ndarray]:
        """Extract raw intent signals without processing."""
        
        # Muscle intent (primary signal)
        muscle_intent = human_action.copy()
        
        # Joint kinematics
        joint_velocity = joint_state.get('qvel', np.array([0.0]))[0] if 'qvel' in joint_state else 0.0
        joint_position = current_angle
        
        # Calculate acceleration
        if self.prev_velocity is not None:
            joint_acceleration = (joint_velocity - self.prev_velocity) / self.dt
        else:
            joint_acceleration = 0.0
        
        self.prev_velocity = joint_velocity
        
        # Co-contraction level
        cocontraction = self._calculate_cocontraction(muscle_activations)
        
        # Target tracking error
        position_error = target_angle - current_angle
        
        # Velocity intent (desired velocity based on position error)
        desired_velocity = np.clip(position_error * 2.0, -1.0, 1.0)  # Simple P controller
        velocity_error = desired_velocity - joint_velocity
        
        # Movement phase detection
        movement_phase = self._detect_movement_phase(joint_velocity, joint_acceleration)
        
        # Effort level (total muscle activation)
        effort_level = np.sum(muscle_activations) / len(muscle_activations) if len(muscle_activations) > 0 else 0.0
        
        return {
            'muscle_intent': muscle_intent,
            'joint_position': np.array([joint_position]),
            'joint_velocity': np.array([joint_velocity]),
            'joint_acceleration': np.array([joint_acceleration]),
            'cocontraction': np.array([cocontraction]),
            'position_error': np.array([position_error]),
            'velocity_error': np.array([velocity_error]),
            'movement_phase': np.array([movement_phase]),
            'effort_level': np.array([effort_level]),
            'target_angle': np.array([target_angle])
        }
    
    def _calculate_cocontraction(self, muscle_activations: np.ndarray) -> float:
        """
        Calculate co-contraction level from muscle activations.
        
        Args:
            muscle_activations: Array of muscle activation levels
            
        Returns:
            Co-contraction level (0-1)
        """
        if len(muscle_activations) < 2:
            return 0.0
        
        # Assume muscles are paired (flexor-extensor pairs)
        n_muscles = len(muscle_activations)
        cocontraction = 0.0
        n_pairs = 0
        
        for i in range(0, n_muscles, 2):
            if i + 1 < n_muscles:
                # Co-contraction is minimum of antagonist pair
                pair_cocontraction = min(muscle_activations[i], muscle_activations[i + 1])
                cocontraction += pair_cocontraction
                n_pairs += 1
        
        return cocontraction / n_pairs if n_pairs > 0 else 0.0
    
    def _detect_movement_phase(self, velocity: float, acceleration: float) -> float:
        """
        Detect current movement phase.
        
        Args:
            velocity: Joint velocity
            acceleration: Joint acceleration
            
        Returns:
            Movement phase encoding (0: static, 0.5: accelerating, 1.0: decelerating)
        """
        vel_threshold = 0.05
        acc_threshold = 0.1
        
        if abs(velocity) < vel_threshold:
            return 0.0  # Static/holding
        elif velocity * acceleration > acc_threshold:
            return 0.5  # Accelerating in direction of movement
        elif velocity * acceleration < -acc_threshold:
            return 1.0  # Decelerating
        else:
            return 0.25  # Constant velocity
    
    def _process_signals(self, raw_intent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply filtering and smoothing to intent signals."""
        processed_intent = copy.deepcopy(raw_intent)
        
        # Apply filtering if enabled and we have enough history
        if self.config.enable_filtering and len(self.signal_history) > self.config.filter_order:
            processed_intent = self._apply_filtering(processed_intent)
        
        # Apply smoothing if enabled and we have enough history
        if self.config.enable_smoothing and len(self.signal_history) >= self.config.smoothing_window:
            processed_intent = self._apply_smoothing(processed_intent)
        
        return processed_intent
    
    def _apply_filtering(self, intent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply low-pass filtering to signals."""
        filtered_intent = copy.deepcopy(intent)
        
        # Only filter signals that benefit from it (not discrete states)
        filter_keys = ['joint_velocity', 'joint_acceleration', 'position_error', 'velocity_error']
        
        for key in filter_keys:
            if key in intent and len(self.signal_history) > self.config.filter_order:
                # Get signal history
                signal_values = [h[key][0] for h in self.signal_history[-self.config.filter_order:]]
                signal_values.append(intent[key][0])
                
                # Apply filter
                filtered_value = lfilter(self.filter_b, self.filter_a, signal_values)[-1]
                filtered_intent[key] = np.array([filtered_value])
        
        return filtered_intent
    
    def _apply_smoothing(self, intent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply smoothing to signals using Savitzky-Golay filter."""
        smoothed_intent = copy.deepcopy(intent)
        
        # Only smooth signals that benefit from it
        smooth_keys = ['joint_velocity', 'joint_acceleration', 'effort_level']
        
        for key in smooth_keys:
            if key in intent and len(self.signal_history) >= self.config.smoothing_window:
                # Get signal history
                signal_values = [h[key][0] for h in self.signal_history[-self.config.smoothing_window:]]
                signal_values.append(intent[key][0])
                
                # Apply Savitzky-Golay smoothing
                if len(signal_values) >= self.config.smoothing_window:
                    smoothed_values = savgol_filter(
                        signal_values, 
                        self.config.smoothing_window, 
                        self.config.smoothing_poly_order
                    )
                    smoothed_intent[key] = np.array([smoothed_values[-1]])
        
        return smoothed_intent
    
    def _apply_corruption(self, intent: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply corruption to intent signals for robustness testing."""
        corrupted_intent = copy.deepcopy(intent)
        
        for key, signal in corrupted_intent.items():
            # Skip discrete signals
            if key in ['movement_phase', 'target_angle']:
                continue
            
            # Add noise
            if self.config.noise_std > 0:
                noise = self.noise_generator.normal(0, self.config.noise_std, signal.shape)
                signal += noise
            
            # Add bias drift
            if self.config.bias_drift > 0:
                self.bias_accumulator += self.noise_generator.normal(0, self.config.bias_drift)
                signal += self.bias_accumulator
            
            # Apply dropout
            if self.config.dropout_rate > 0:
                if self.noise_generator.random() < self.config.dropout_rate:
                    signal *= 0
            
            corrupted_intent[key] = signal
        
        return corrupted_intent
    
    def _update_history(self, intent: Dict[str, np.ndarray]):
        """Update signal history."""
        self.signal_history.append(copy.deepcopy(intent))
        
        # Maintain maximum history length
        if len(self.signal_history) > self.max_history_length:
            self.signal_history.pop(0)
    
    def get_delayed_intent(self, delay_steps: int = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get intent signal with specified delay.
        
        Args:
            delay_steps: Number of steps to delay (uses config default if None)
            
        Returns:
            Delayed intent signal or None if insufficient history
        """
        if delay_steps is None:
            delay_steps = self.config.delay_steps
        
        if delay_steps == 0:
            return self.signal_history[-1] if self.signal_history else None
        
        if len(self.signal_history) > delay_steps:
            return self.signal_history[-(delay_steps + 1)]
        
        return None
    
    def enable_corruption(self, 
                         noise_std: float = None,
                         delay_steps: int = None,
                         dropout_rate: float = None,
                         bias_drift: float = None):
        """
        Enable signal corruption with specified parameters.
        
        Args:
            noise_std: Standard deviation of additive noise
            delay_steps: Number of steps to delay signals
            dropout_rate: Probability of signal dropout
            bias_drift: Standard deviation of bias drift
        """
        self.corruption_enabled = True
        
        if noise_std is not None:
            self.config.noise_std = noise_std
        if delay_steps is not None:
            self.config.delay_steps = delay_steps
        if dropout_rate is not None:
            self.config.dropout_rate = dropout_rate
        if bias_drift is not None:
            self.config.bias_drift = bias_drift
    
    def disable_corruption(self):
        """Disable signal corruption."""
        self.corruption_enabled = False
        self.bias_accumulator = 0.0
    
    def reset(self):
        """Reset processor state."""
        self.signal_history.clear()
        self.prev_velocity = None
        self.prev_acceleration = None
        self.bias_accumulator = 0.0
    
    def get_intent_vector(self, intent: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert intent dictionary to a single vector for RL observation.
        
        Args:
            intent: Intent signal dictionary
            
        Returns:
            Flattened intent vector
        """
        # Define the order of signals in the vector
        signal_order = [
            'muscle_intent',
            'joint_velocity', 
            'joint_acceleration',
            'cocontraction',
            'position_error',
            'velocity_error',
            'movement_phase',
            'effort_level'
        ]
        
        intent_vector = []
        
        for key in signal_order:
            if key in intent:
                signal = intent[key]
                if hasattr(signal, '__len__') and len(signal) > 1:
                    intent_vector.extend(signal.flatten())
                else:
                    intent_vector.append(float(signal))
        
        return np.array(intent_vector)
    
    def get_intent_vector_size(self, muscle_dim: int = 6) -> int:
        """
        Get the size of the intent vector.
        
        Args:
            muscle_dim: Dimensionality of muscle intent
            
        Returns:
            Size of the intent vector
        """
        # muscle_intent + 7 scalar values
        return muscle_dim + 7