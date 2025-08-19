import numpy as np
from typing import Dict, List, Optional


class ProprioceptiveModule:
    """
    Modular proprioceptive sensing system for human agent.
    Provides various types of force and torque feedback that can be easily enabled/disabled.
    """
    
    def __init__(self, 
                 enabled: bool = True, 
                 sensing_types: List[str] = None,
                 noise_std: float = 0.0):
        """
        Initialize proprioceptive module.
        
        Args:
            enabled: Whether proprioception is active
            sensing_types: List of sensing modalities to include
            noise_std: Standard deviation of sensor noise
        """
        self.enabled = enabled
        self.noise_std = noise_std
        
        # Default sensing types
        if sensing_types is None:
            sensing_types = ['external_force', 'joint_torque', 'gravity_torque']
        
        self.sensing_types = sensing_types
        
        # Available sensing modalities
        self.available_sensors = {
            'external_force',      # Direct external forces (qfrc_applied)
            'joint_torque',        # Total joint torque (qfrc_total)
            'gravity_torque',      # Gravitational torque (qfrc_bias)
            'muscle_torque',       # Muscle-generated torque (qfrc_actuator)
            'net_external_torque', # Estimated external torque (total - gravity - muscle)
            'joint_stiffness'      # Estimated joint stiffness from co-contraction
        }
        
        # Validate sensing types
        invalid_types = set(sensing_types) - self.available_sensors
        if invalid_types:
            raise ValueError(f"Invalid sensing types: {invalid_types}. "
                           f"Available types: {self.available_sensors}")
    
    def get_proprioceptive_obs(self, 
                              sim, 
                              joint_dof: int,
                              muscle_activations: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract proprioceptive observations from simulation state.
        
        Args:
            sim: MuJoCo simulation object
            joint_dof: Degree of freedom index for the joint of interest
            muscle_activations: Current muscle activation levels
            
        Returns:
            Dictionary of proprioceptive observations
        """
        if not self.enabled:
            return {}
        
        obs = {}
        
        # External force sensing (qfrc_applied)
        if 'external_force' in self.sensing_types:
            external_force = sim.data.qfrc_applied[joint_dof]
            obs['external_force'] = self._add_noise(external_force)
        
        # Total joint torque
        if 'joint_torque' in self.sensing_types:
            joint_torque = sim.data.qfrc_total[joint_dof]
            obs['joint_torque'] = self._add_noise(joint_torque)
        
        # Gravitational torque
        if 'gravity_torque' in self.sensing_types:
            gravity_torque = sim.data.qfrc_bias[joint_dof]
            obs['gravity_torque'] = self._add_noise(gravity_torque)
        
        # Muscle-generated torque
        if 'muscle_torque' in self.sensing_types:
            muscle_torque = sim.data.qfrc_actuator[joint_dof]
            obs['muscle_torque'] = self._add_noise(muscle_torque)
        
        # Net external torque (estimated)
        if 'net_external_torque' in self.sensing_types:
            net_external = (sim.data.qfrc_total[joint_dof] - 
                          sim.data.qfrc_bias[joint_dof] - 
                          sim.data.qfrc_actuator[joint_dof])
            obs['net_external_torque'] = self._add_noise(net_external)
        
        # Joint stiffness estimation from muscle co-contraction
        if 'joint_stiffness' in self.sensing_types and muscle_activations is not None:
            stiffness = self._estimate_joint_stiffness(muscle_activations)
            obs['joint_stiffness'] = self._add_noise(stiffness)
        
        return obs
    
    def _estimate_joint_stiffness(self, muscle_activations: np.ndarray) -> float:
        """
        Estimate joint stiffness based on muscle co-contraction.
        
        Args:
            muscle_activations: Array of muscle activation levels
            
        Returns:
            Estimated joint stiffness value
        """
        if len(muscle_activations) < 2:
            return 0.0
        
        # Simple co-contraction measure: minimum of antagonist pairs
        # For elbow: assume flexors and extensors in pairs
        n_muscles = len(muscle_activations)
        co_contraction = 0.0
        
        # Calculate pairwise minimum activations (co-contraction)
        for i in range(0, n_muscles, 2):
            if i + 1 < n_muscles:
                co_contraction += min(muscle_activations[i], muscle_activations[i + 1])
        
        # Scale by number of pairs
        n_pairs = n_muscles // 2
        if n_pairs > 0:
            co_contraction /= n_pairs
        
        return co_contraction
    
    def _add_noise(self, value: float) -> float:
        """
        Add sensor noise to proprioceptive measurements.
        
        Args:
            value: Original sensor value
            
        Returns:
            Noisy sensor value
        """
        if self.noise_std <= 0:
            return value
        
        noise = np.random.normal(0, self.noise_std)
        return value + noise
    
    def get_observation_space_size(self) -> int:
        """
        Get the size of the proprioceptive observation space.
        
        Returns:
            Number of proprioceptive observation dimensions
        """
        if not self.enabled:
            return 0
        
        return len(self.sensing_types)
    
    def toggle_proprioception(self) -> bool:
        """
        Toggle proprioception on/off.
        
        Returns:
            New enabled state
        """
        self.enabled = not self.enabled
        return self.enabled
    
    def set_noise_level(self, noise_std: float):
        """
        Set the noise level for proprioceptive sensors.
        
        Args:
            noise_std: Standard deviation of sensor noise
        """
        self.noise_std = max(0.0, noise_std)
    
    def add_sensing_type(self, sensing_type: str):
        """
        Add a new sensing modality.
        
        Args:
            sensing_type: Type of sensing to add
        """
        if sensing_type in self.available_sensors and sensing_type not in self.sensing_types:
            self.sensing_types.append(sensing_type)
    
    def remove_sensing_type(self, sensing_type: str):
        """
        Remove a sensing modality.
        
        Args:
            sensing_type: Type of sensing to remove
        """
        if sensing_type in self.sensing_types:
            self.sensing_types.remove(sensing_type)
    
    def __str__(self) -> str:
        status = "Enabled" if self.enabled else "Disabled"
        return f"ProprioceptiveModule({status}, types={self.sensing_types}, noise_std={self.noise_std})"