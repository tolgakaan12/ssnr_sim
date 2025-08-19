import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
from etils import epath

from hmi_elbow_base import HmiElbowBase
from proprioception import ProprioceptiveModule


class HumanElbowEnv(HmiElbowBase):
    """
    Enhanced human elbow environment with proprioception and improved perturbation system.
    
    This environment extends the base HmiElbowBase with:
    - Modular proprioceptive sensing
    - Continuous perturbations during episodes
    - Enhanced target angle variation
    - Better reward shaping for elbow angle tracking
    """
    
    def __init__(self, 
                 perturb_scale: float = 5.0,
                 proprioception_enabled: bool = True,
                 proprioception_sensing_types: Optional[list] = None,
                 proprioception_noise_std: float = 0.0,
                 continuous_perturbations: bool = True,
                 perturbation_frequency: float = 0.01,
                 target_angle_range: Tuple[float, float] = (0.0, 2.27),
                 target_change_frequency: float = 0.005,
                 **kwargs):
        """
        Initialize enhanced human elbow environment.
        
        Args:
            perturb_scale: Scale of perturbation forces
            proprioception_enabled: Whether to include proprioceptive sensing
            proprioception_sensing_types: List of proprioceptive sensing modalities
            proprioception_noise_std: Noise level for proprioceptive sensors
            continuous_perturbations: Whether to apply perturbations during episodes
            perturbation_frequency: Probability of applying perturbation per step
            target_angle_range: Range for target elbow angles (min, max) in radians
            target_change_frequency: Probability of changing target per step
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize proprioception module
        self.proprioception = ProprioceptiveModule(
            enabled=proprioception_enabled,
            sensing_types=proprioception_sensing_types,
            noise_std=proprioception_noise_std
        )
        
        # Perturbation settings
        self.continuous_perturbations = continuous_perturbations
        self.perturbation_frequency = perturbation_frequency
        
        # Target angle settings
        self.target_angle_range = target_angle_range
        self.target_change_frequency = target_change_frequency
        self.current_target_angle = None
        
        # Step counter for tracking
        self.step_count = 0
        self.total_perturbation = 0.0
        
        # Initialize parent class
        super().__init__(perturb_scale=perturb_scale, **kwargs)
        
        # Find elbow joint DOF after initialization
        self._find_elbow_dof()
        
        # Set initial target angle
        self._update_target_angle()
    
    def _find_elbow_dof(self):
        """Find the degree of freedom index for the elbow joint."""
        try:
            # Try to find elbow joint by name
            elbow_joint_names = ['r_elbow_flex', 'elbow_flex', 'elbow']
            self.elbow_dof = None
            
            for joint_name in elbow_joint_names:
                try:
                    joint_id = self.sim.model.joint_name2id(joint_name)
                    self.elbow_dof = self.sim.model.jnt_dofadr[joint_id]
                    self.elbow_joint_name = joint_name
                    break
                except:
                    continue
            
            if self.elbow_dof is None:
                # Fallback: assume first joint is elbow
                self.elbow_dof = 0
                self.elbow_joint_name = self.sim.model.joint_id2name(0)
                
        except Exception as e:
            print(f"Warning: Could not find elbow joint, using DOF 0: {e}")
            self.elbow_dof = 0
            self.elbow_joint_name = "unknown"
    
    def _update_target_angle(self):
        """Update the target elbow angle."""
        min_angle, max_angle = self.target_angle_range
        self.current_target_angle = self.np_random.uniform(min_angle, max_angle)
    
    def get_obs_dict(self, sim) -> Dict[str, np.ndarray]:
        """
        Get observation dictionary including proprioceptive sensing.
        
        Args:
            sim: MuJoCo simulation object
            
        Returns:
            Dictionary containing all observations including proprioceptive data
        """
        # Get base observations
        obs_dict = super().get_obs_dict(sim)
        
        # Add proprioceptive observations
        if self.proprioception.enabled:
            # Get current muscle activations for stiffness estimation
            muscle_activations = sim.data.act if sim.model.na > 0 else None
            
            # Get proprioceptive observations
            proprio_obs = self.proprioception.get_proprioceptive_obs(
                sim, self.elbow_dof, muscle_activations
            )
            
            # Add to observation dictionary
            obs_dict.update(proprio_obs)
        
        # Add current target angle
        obs_dict['target_angle'] = np.array([self.current_target_angle])
        
        # Add current elbow angle
        if self.elbow_dof < len(sim.data.qpos):
            current_angle = sim.data.qpos[self.elbow_dof]
        else:
            current_angle = 0.0
        obs_dict['current_angle'] = np.array([current_angle])
        
        # Add angle error
        angle_error = self.current_target_angle - current_angle
        obs_dict['angle_error'] = np.array([angle_error])
        
        return obs_dict
    
    def get_reward_dict(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate reward components for the human agent.
        
        Args:
            obs_dict: Current observation dictionary
            
        Returns:
            Dictionary of reward components
        """
        # Get base rewards (if any)
        try:
            rwd_dict = super().get_reward_dict(obs_dict)
        except:
            rwd_dict = {}
        
        # Current elbow angle
        if 'current_angle' in obs_dict:
            current_angle = obs_dict['current_angle'][0]
        else:
            current_angle = 0.0
        
        # Angle tracking reward
        angle_error = abs(self.current_target_angle - current_angle)
        angle_reward = np.exp(-5.0 * angle_error)  # Exponential reward for accuracy
        
        # Smoothness reward (penalize large changes in muscle activation)
        smoothness_reward = 0.0
        if hasattr(self, 'prev_action') and self.prev_action is not None:
            if 'act' in obs_dict:
                action_diff = np.linalg.norm(obs_dict['act'] - self.prev_action)
                smoothness_reward = -0.1 * action_diff
        
        # Energy efficiency reward (penalize high muscle activations)
        efficiency_reward = 0.0
        if 'act' in obs_dict:
            total_activation = np.sum(obs_dict['act'])
            efficiency_reward = -0.01 * total_activation
        
        # Stability reward (penalize high joint velocities)
        stability_reward = 0.0
        if 'body_qvel' in obs_dict and len(obs_dict['body_qvel']) > self.elbow_dof:
            elbow_velocity = abs(obs_dict['body_qvel'][self.elbow_dof])
            stability_reward = -0.05 * elbow_velocity
        
        # Update reward dictionary
        rwd_dict.update({
            'angle_tracking': angle_reward,
            'smoothness': smoothness_reward,
            'efficiency': efficiency_reward,
            'stability': stability_reward
        })
        
        # Calculate total reward
        reward_weights = {
            'angle_tracking': 10.0,
            'smoothness': 1.0,
            'efficiency': 0.5,
            'stability': 1.0
        }
        
        total_reward = sum(reward_weights.get(key, 1.0) * value 
                          for key, value in rwd_dict.items() 
                          if isinstance(value, (int, float)))
        
        rwd_dict['total'] = total_reward
        
        return rwd_dict
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Execute one environment step with enhanced perturbations and target updates.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Store previous action for smoothness calculation
        if hasattr(self, 'obs_dict') and 'act' in self.obs_dict:
            self.prev_action = self.obs_dict['act'].copy()
        else:
            self.prev_action = None
        
        # Apply continuous perturbations during episode
        if (self.continuous_perturbations and 
            self.step_count > 0 and 
            self.np_random.random() < self.perturbation_frequency):
            
            perturbation = self.perturb_scale * self.np_random.randn()
            self.sim.data.qfrc_applied[self.elbow_dof] += perturbation
            self.total_perturbation += abs(perturbation)
        
        # Randomly change target angle
        if (self.step_count > 0 and 
            self.np_random.random() < self.target_change_frequency):
            self._update_target_angle()
        
        # Execute parent step
        obs, reward, done, info = super().step(action)
        
        # Update step counter
        self.step_count += 1
        
        # Add perturbation info
        info['total_perturbation'] = self.total_perturbation
        info['current_target'] = self.current_target_angle
        info['proprioception_enabled'] = self.proprioception.enabled
        
        return obs, reward, done, info
    
    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Reset the environment with new target angle and perturbation.
        
        Args:
            **kwargs: Additional reset arguments
            
        Returns:
            Initial observation
        """
        # Reset counters
        self.step_count = 0
        self.total_perturbation = 0.0
        self.prev_action = None
        
        # Update target angle
        self._update_target_angle()
        
        # Reset parent environment
        obs = super().reset(**kwargs)
        
        return obs
    
    def toggle_proprioception(self) -> bool:
        """
        Toggle proprioception on/off during runtime.
        
        Returns:
            New proprioception state
        """
        return self.proprioception.toggle_proprioception()
    
    def set_perturbation_params(self, 
                               continuous: bool = None, 
                               frequency: float = None, 
                               scale: float = None):
        """
        Update perturbation parameters during runtime.
        
        Args:
            continuous: Whether to enable continuous perturbations
            frequency: Perturbation frequency
            scale: Perturbation scale
        """
        if continuous is not None:
            self.continuous_perturbations = continuous
        if frequency is not None:
            self.perturbation_frequency = frequency
        if scale is not None:
            self.perturb_scale = scale
    
    def set_target_params(self, 
                         angle_range: Tuple[float, float] = None,
                         change_frequency: float = None):
        """
        Update target angle parameters during runtime.
        
        Args:
            angle_range: New target angle range
            change_frequency: Target change frequency
        """
        if angle_range is not None:
            self.target_angle_range = angle_range
        if change_frequency is not None:
            self.target_change_frequency = change_frequency
    
    def get_proprioception_info(self) -> Dict[str, Any]:
        """
        Get information about proprioception system.
        
        Returns:
            Dictionary with proprioception details
        """
        return {
            'enabled': self.proprioception.enabled,
            'sensing_types': self.proprioception.sensing_types,
            'noise_std': self.proprioception.noise_std,
            'observation_size': self.proprioception.get_observation_space_size()
        }