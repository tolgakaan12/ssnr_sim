"""
Configuration System for Human-Exoskeleton Training

Centralized configuration management for experiments with easy parameter sweeps
and modular toggling of features.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
import json
import os
from pathlib import Path


@dataclass
class ProprioceptionConfig:
    """Configuration for proprioceptive sensing."""
    enabled: bool = True
    sensing_types: List[str] = field(default_factory=lambda: ['external_force', 'joint_torque', 'gravity_torque'])
    noise_std: float = 0.0
    
    def toggle(self) -> bool:
        """Toggle proprioception on/off."""
        self.enabled = not self.enabled
        return self.enabled


@dataclass
class PerturbationConfig:
    """Configuration for perturbation system."""
    scale: float = 5.0
    continuous: bool = True
    frequency: float = 0.01  # Probability per step
    reset_only: bool = False
    
    def set_mild(self):
        """Set mild perturbation parameters."""
        self.scale = 2.0
        self.frequency = 0.005
    
    def set_moderate(self):
        """Set moderate perturbation parameters."""
        self.scale = 5.0
        self.frequency = 0.01
    
    def set_severe(self):
        """Set severe perturbation parameters."""
        self.scale = 10.0
        self.frequency = 0.02


@dataclass
class TargetConfig:
    """Configuration for target angle system."""
    angle_range: Tuple[float, float] = (0.0, 2.27)  # radians
    change_frequency: float = 0.005  # Probability per step
    fixed_target: Optional[float] = None  # Use fixed target if specified
    
    def set_full_range(self):
        """Set full elbow range."""
        self.angle_range = (0.0, 2.27)
    
    def set_limited_range(self):
        """Set limited range for easier task."""
        self.angle_range = (0.5, 1.8)


@dataclass
class HumanTrainingConfig:
    """Configuration for human agent training."""
    # Environment settings
    proprioception: ProprioceptionConfig = field(default_factory=ProprioceptionConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    
    # Training hyperparameters
    total_timesteps: int = 800_000
    episode_length: int = 1500
    num_envs: int = 4
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    
    # Evaluation
    eval_freq: int = 10_000
    checkpoint_freq: int = 50_000
    
    # Output
    experiment_name: str = "human_default"
    device: str = 'cpu'
    
    def set_quick_test(self):
        """Set parameters for quick testing."""
        self.total_timesteps = 50_000
        self.episode_length = 500
        self.eval_freq = 2_000
        self.checkpoint_freq = 10_000
    
    def set_long_training(self):
        """Set parameters for thorough training."""
        self.total_timesteps = 1_500_000
        self.episode_length = 2000
        self.eval_freq = 20_000


@dataclass
class IntentProcessingConfig:
    """Configuration for intent signal processing."""
    # Signal weights
    muscle_weight: float = 1.0
    velocity_weight: float = 0.5
    acceleration_weight: float = 0.3
    cocontraction_weight: float = 0.2
    target_error_weight: float = 0.8
    
    # Filtering
    enable_filtering: bool = True
    filter_cutoff: float = 10.0  # Hz
    filter_order: int = 2
    
    # Smoothing
    enable_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_poly_order: int = 2
    
    # Corruption parameters
    corruption_enabled: bool = False
    noise_std: float = 0.0
    delay_steps: int = 0
    dropout_rate: float = 0.0
    bias_drift: float = 0.0
    
    def enable_corruption(self, 
                         noise_std: float = 0.05,
                         delay_steps: int = 2,
                         dropout_rate: float = 0.02,
                         bias_drift: float = 0.01):
        """Enable intent signal corruption."""
        self.corruption_enabled = True
        self.noise_std = noise_std
        self.delay_steps = delay_steps
        self.dropout_rate = dropout_rate
        self.bias_drift = bias_drift
    
    def disable_corruption(self):
        """Disable intent signal corruption."""
        self.corruption_enabled = False
        self.noise_std = 0.0
        self.delay_steps = 0
        self.dropout_rate = 0.0
        self.bias_drift = 0.0


@dataclass
class ExoskeletonTrainingConfig:
    """Configuration for exoskeleton agent training."""
    # Human model settings
    human_model_path: str = ""
    human_vec_normalize_path: Optional[str] = None
    human_proprioception_enabled: bool = True
    human_perturb_scale: float = 5.0
    
    # Exoskeleton settings
    max_assistance_torque: float = 10.0
    assistance_penalty_weight: float = 0.01
    interference_penalty_weight: float = 0.1
    
    # Intent processing
    intent_processing: IntentProcessingConfig = field(default_factory=IntentProcessingConfig)
    
    # Training hyperparameters
    algorithm: str = 'PPO'  # 'PPO' or 'SAC'
    total_timesteps: int = 500_000
    episode_length: int = 1500
    learning_rate: float = 3e-4
    batch_size: int = 64
    
    # Evaluation
    eval_freq: int = 5_000
    checkpoint_freq: int = 25_000
    
    # Output
    experiment_name: str = "exoskeleton_default"
    device: str = 'cpu'
    
    def set_high_assistance(self):
        """Configure for high assistance capability."""
        self.max_assistance_torque = 20.0
        self.assistance_penalty_weight = 0.005
    
    def set_low_assistance(self):
        """Configure for low assistance capability."""
        self.max_assistance_torque = 5.0
        self.assistance_penalty_weight = 0.02
    
    def set_corruption_training(self):
        """Enable intent corruption for robust training."""
        self.intent_processing.enable_corruption()


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Phase configuration
    train_human: bool = True
    train_exoskeleton: bool = True
    
    # Individual phase configs
    human_config: HumanTrainingConfig = field(default_factory=HumanTrainingConfig)
    exoskeleton_config: ExoskeletonTrainingConfig = field(default_factory=ExoskeletonTrainingConfig)
    
    # Global settings
    base_output_dir: str = "./experiments"
    random_seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Ensure exoskeleton config references human model."""
        if self.train_human and self.train_exoskeleton:
            # Auto-configure exoskeleton to use human model
            human_model_dir = f"{self.base_output_dir}/human_agent_training_{self.human_config.experiment_name}"
            self.exoskeleton_config.human_model_path = f"{human_model_dir}/models/human_agent_final"
            self.exoskeleton_config.human_vec_normalize_path = f"{human_model_dir}/models/vec_normalize.pkl"
            self.exoskeleton_config.human_proprioception_enabled = self.human_config.proprioception.enabled
            self.exoskeleton_config.human_perturb_scale = self.human_config.perturbation.scale


class ConfigManager:
    """Utility class for managing experiment configurations."""
    
    @staticmethod
    def create_proprioception_study() -> List[ExperimentConfig]:
        """Create configs for proprioception ablation study."""
        configs = []
        
        # With proprioception
        config_with = ExperimentConfig()
        config_with.human_config.experiment_name = "with_proprioception"
        config_with.human_config.proprioception.enabled = True
        config_with.exoskeleton_config.experiment_name = "exo_with_proprioception"
        configs.append(config_with)
        
        # Without proprioception
        config_without = ExperimentConfig()
        config_without.human_config.experiment_name = "without_proprioception"
        config_without.human_config.proprioception.enabled = False
        config_without.exoskeleton_config.experiment_name = "exo_without_proprioception"
        configs.append(config_without)
        
        return configs
    
    @staticmethod
    def create_perturbation_study() -> List[ExperimentConfig]:
        """Create configs for perturbation scale study."""
        configs = []
        perturbation_scales = [2.0, 5.0, 10.0]
        
        for scale in perturbation_scales:
            config = ExperimentConfig()
            config.human_config.experiment_name = f"perturb_scale_{scale}"
            config.human_config.perturbation.scale = scale
            config.exoskeleton_config.experiment_name = f"exo_perturb_scale_{scale}"
            configs.append(config)
        
        return configs
    
    @staticmethod
    def create_assistance_study() -> List[ExperimentConfig]:
        """Create configs for assistance torque study."""
        configs = []
        assistance_levels = [5.0, 10.0, 20.0]
        
        for max_torque in assistance_levels:
            config = ExperimentConfig()
            config.human_config.experiment_name = "baseline"
            config.exoskeleton_config.experiment_name = f"assist_torque_{max_torque}"
            config.exoskeleton_config.max_assistance_torque = max_torque
            config.train_human = False  # Use pre-trained human
            configs.append(config)
        
        return configs
    
    @staticmethod
    def create_corruption_study() -> List[ExperimentConfig]:
        """Create configs for intent corruption robustness study."""
        configs = []
        
        # No corruption
        config_clean = ExperimentConfig()
        config_clean.human_config.experiment_name = "baseline"
        config_clean.exoskeleton_config.experiment_name = "exo_no_corruption"
        config_clean.train_human = False
        configs.append(config_clean)
        
        # With corruption
        config_corrupt = ExperimentConfig()
        config_corrupt.human_config.experiment_name = "baseline"
        config_corrupt.exoskeleton_config.experiment_name = "exo_with_corruption"
        config_corrupt.exoskeleton_config.set_corruption_training()
        config_corrupt.train_human = False
        configs.append(config_corrupt)
        
        return configs
    
    @staticmethod
    def save_config(config: ExperimentConfig, filepath: str):
        """Save configuration to JSON file."""
        config_dict = asdict(config)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> ExperimentConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        human_config = HumanTrainingConfig(**config_dict['human_config'])
        exoskeleton_config = ExoskeletonTrainingConfig(**config_dict['exoskeleton_config'])
        
        config = ExperimentConfig(
            train_human=config_dict['train_human'],
            train_exoskeleton=config_dict['train_exoskeleton'],
            human_config=human_config,
            exoskeleton_config=exoskeleton_config,
            base_output_dir=config_dict['base_output_dir'],
            random_seed=config_dict['random_seed'],
            verbose=config_dict['verbose']
        )
        
        return config
    
    @staticmethod
    def create_quick_test_config() -> ExperimentConfig:
        """Create a configuration for quick testing."""
        config = ExperimentConfig()
        config.human_config.set_quick_test()
        config.human_config.experiment_name = "quick_test"
        config.exoskeleton_config.experiment_name = "exo_quick_test"
        config.exoskeleton_config.total_timesteps = 25_000
        config.exoskeleton_config.eval_freq = 1_000
        return config
    
    @staticmethod
    def create_production_config() -> ExperimentConfig:
        """Create a configuration for production training."""
        config = ExperimentConfig()
        config.human_config.set_long_training()
        config.human_config.experiment_name = "production_human"
        config.exoskeleton_config.experiment_name = "production_exoskeleton"
        config.exoskeleton_config.total_timesteps = 1_000_000
        return config


def main():
    """Example usage of the configuration system."""
    # Create a quick test configuration
    config = ConfigManager.create_quick_test_config()
    
    # Save it
    ConfigManager.save_config(config, "./configs/quick_test.json")
    
    # Create proprioception study
    prop_configs = ConfigManager.create_proprioception_study()
    for i, cfg in enumerate(prop_configs):
        ConfigManager.save_config(cfg, f"./configs/proprioception_study_{i}.json")
    
    print("Example configurations created in ./configs/")


if __name__ == '__main__':
    main()