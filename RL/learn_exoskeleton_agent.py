"""
Exoskeleton Agent Training Script

This script trains an exoskeleton agent to assist a pre-trained human agent
in elbow control tasks. The human agent is frozen and treated as part of the environment.
"""

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, SAC
import numpy as np
import os
import json
import argparse
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from exoskeleton_env import ExoskeletonEnv
from intent_signals import IntentConfig


# Training configuration
TOTAL_TIMESTEPS = 500_000  # Exoskeleton training typically faster than human
EVAL_FREQ = 5_000
CHECKPOINT_FREQ = 25_000


class PerformanceTrackingCallback(BaseCallback):
    """Custom callback to track exoskeleton performance metrics."""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.performance_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Get performance metrics from the environment
            if hasattr(self.training_env.envs[0], 'get_performance_metrics'):
                metrics = self.training_env.envs[0].get_performance_metrics()
                self.performance_history.append({
                    'timestep': self.n_calls,
                    **metrics
                })
                
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Assistance efficiency: {metrics.get('assistance_efficiency', 0):.4f}")
        
        return True


def make_exoskeleton_env(human_model_path: str,
                        human_vec_normalize_path: Optional[str] = None,
                        proprioception_enabled: bool = True,
                        perturb_scale: float = 5.0,
                        max_assistance_torque: float = 10.0,
                        intent_corruption: bool = False,
                        seed: int = 42):
    """
    Create an exoskeleton environment.
    
    Args:
        human_model_path: Path to trained human agent
        human_vec_normalize_path: Path to VecNormalize stats
        proprioception_enabled: Whether human was trained with proprioception
        perturb_scale: Perturbation scale used in human training
        max_assistance_torque: Maximum assistance torque
        intent_corruption: Whether to enable intent signal corruption
        seed: Random seed
        
    Returns:
        Wrapped environment function
    """
    def _init():
        # Create intent configuration
        intent_config = IntentConfig()
        if intent_corruption:
            intent_config.noise_std = 0.05
            intent_config.delay_steps = 2
            intent_config.dropout_rate = 0.02
        
        # Create exoskeleton environment
        env = ExoskeletonEnv(
            human_model_path=human_model_path,
            human_vec_normalize_path=human_vec_normalize_path,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=perturb_scale,
            max_assistance_torque=max_assistance_torque,
            intent_config=intent_config,
            seed=seed
        )
        
        # Enable intent corruption if requested
        if intent_corruption:
            env.enable_intent_corruption()
        
        # Monitor for logging
        env = Monitor(env)
        
        return env
    
    set_random_seed(seed)
    return _init


def create_training_directories(experiment_name: str) -> Dict[str, str]:
    """Create directories for training outputs."""
    base_dir = f"./exoskeleton_training_{experiment_name}"
    dirs = {
        'base': base_dir,
        'models': f"{base_dir}/models",
        'logs': f"{base_dir}/logs",
        'tensorboard': f"{base_dir}/tensorboard",
        'eval': f"{base_dir}/eval",
        'plots': f"{base_dir}/plots"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def train_exoskeleton_agent(human_model_path: str,
                           human_vec_normalize_path: Optional[str] = None,
                           proprioception_enabled: bool = True,
                           perturb_scale: float = 5.0,
                           max_assistance_torque: float = 10.0,
                           intent_corruption: bool = False,
                           algorithm: str = 'PPO',
                           experiment_name: str = "default",
                           device: str = 'cpu',
                           verbose: bool = True) -> str:
    """
    Train the exoskeleton agent.
    
    Args:
        human_model_path: Path to trained human agent
        human_vec_normalize_path: Path to VecNormalize stats
        proprioception_enabled: Whether human was trained with proprioception  
        perturb_scale: Perturbation scale used in human training
        max_assistance_torque: Maximum assistance torque
        intent_corruption: Whether to enable intent signal corruption
        algorithm: RL algorithm to use ('PPO' or 'SAC')
        experiment_name: Name for this training run
        device: Device to use for training
        verbose: Whether to print training progress
        
    Returns:
        Path to the saved final model
    """
    # Validate human model exists
    if not os.path.exists(human_model_path + '.zip') and not os.path.exists(human_model_path):
        raise FileNotFoundError(f"Human model not found: {human_model_path}")
    
    # Create directories
    dirs = create_training_directories(experiment_name)
    
    if verbose:
        print(f"Starting exoskeleton agent training: {experiment_name}")
        print(f"Human model: {human_model_path}")
        print(f"Proprioception: {proprioception_enabled}")
        print(f"Max assistance torque: {max_assistance_torque}")
        print(f"Intent corruption: {intent_corruption}")
        print(f"Algorithm: {algorithm}")
    
    # Create training environment
    env = DummyVecEnv([make_exoskeleton_env(
        human_model_path=human_model_path,
        human_vec_normalize_path=human_vec_normalize_path,
        proprioception_enabled=proprioception_enabled,
        perturb_scale=perturb_scale,
        max_assistance_torque=max_assistance_torque,
        intent_corruption=intent_corruption,
        seed=42
    )])
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_exoskeleton_env(
        human_model_path=human_model_path,
        human_vec_normalize_path=human_vec_normalize_path,
        proprioception_enabled=proprioception_enabled,
        perturb_scale=perturb_scale,
        max_assistance_torque=max_assistance_torque,
        intent_corruption=False,  # No corruption in evaluation
        seed=12345
    )])
    eval_env = VecMonitor(eval_env)
    
    # Configure model based on algorithm
    if algorithm == 'PPO':
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn='tanh'
        )
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1 if verbose else 0,
            tensorboard_log=dirs['tensorboard']
        )
    
    elif algorithm == 'SAC':
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128]),
            activation_fn='relu'
        )
        
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1 if verbose else 0,
            tensorboard_log=dirs['tensorboard']
        )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['models'],
        log_path=dirs['eval'],
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=dirs['models'],
        name_prefix='exoskeleton_checkpoint'
    )
    
    performance_callback = PerformanceTrackingCallback(
        eval_freq=1000,
        verbose=1 if verbose else 0
    )
    
    callbacks = [eval_callback, checkpoint_callback, performance_callback]
    
    # Train the model
    if verbose:
        print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=verbose
    )
    
    # Save final model
    final_model_path = f"{dirs['models']}/exoskeleton_agent_final"
    model.save(final_model_path)
    
    # Save training configuration
    config = {
        'human_model_path': human_model_path,
        'human_vec_normalize_path': human_vec_normalize_path,
        'proprioception_enabled': proprioception_enabled,
        'perturb_scale': perturb_scale,
        'max_assistance_torque': max_assistance_torque,
        'intent_corruption': intent_corruption,
        'algorithm': algorithm,
        'total_timesteps': TOTAL_TIMESTEPS,
        'experiment_name': experiment_name
    }
    
    with open(f"{dirs['base']}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save performance history
    if hasattr(performance_callback, 'performance_history'):
        performance_data = performance_callback.performance_history
        with open(f"{dirs['logs']}/performance_history.json", 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Plot performance metrics
        if performance_data:
            plot_performance_metrics(performance_data, dirs['plots'])
    
    if verbose:
        print(f"Training completed! Models saved to: {dirs['models']}")
        print(f"Final model: {final_model_path}")
    
    return final_model_path


def plot_performance_metrics(performance_data: list, plot_dir: str):
    """Plot training performance metrics."""
    if not performance_data:
        return
    
    timesteps = [d['timestep'] for d in performance_data]
    assistance_efficiency = [d.get('assistance_efficiency', 0) for d in performance_data]
    interference_rate = [d.get('interference_rate', 0) for d in performance_data]
    
    # Plot assistance efficiency
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, assistance_efficiency)
    plt.title('Assistance Efficiency Over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Assistance Efficiency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(timesteps, interference_rate)
    plt.title('Interference Rate Over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Interference Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_exoskeleton_agent(model_path: str,
                              human_model_path: str,
                              human_vec_normalize_path: Optional[str] = None,
                              n_episodes: int = 50,
                              proprioception_enabled: bool = True,
                              perturb_scale: float = 5.0,
                              intent_corruption: bool = False,
                              render: bool = False) -> Dict[str, float]:
    """
    Evaluate a trained exoskeleton agent.
    
    Args:
        model_path: Path to trained exoskeleton model
        human_model_path: Path to human model
        human_vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of evaluation episodes
        proprioception_enabled: Whether proprioception was used
        perturb_scale: Perturbation scale
        intent_corruption: Whether to test with corrupted intent
        render: Whether to render episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Determine algorithm from model
    try:
        model = PPO.load(model_path)
        algorithm = 'PPO'
    except:
        try:
            model = SAC.load(model_path)
            algorithm = 'SAC'
        except:
            raise ValueError(f"Could not load model from {model_path}")
    
    # Create evaluation environment
    env = ExoskeletonEnv(
        human_model_path=human_model_path,
        human_vec_normalize_path=human_vec_normalize_path,
        proprioception_enabled=proprioception_enabled,
        perturb_scale=perturb_scale,
        intent_config=IntentConfig()
    )
    
    if intent_corruption:
        env.enable_intent_corruption(noise_std=0.05, delay_steps=2, dropout_rate=0.02)
    
    # Run evaluation episodes
    episode_rewards = []
    angle_errors = []
    assistance_amounts = []
    interference_amounts = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_angle_errors = []
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            # Track metrics
            if 'angle_error' in info:
                episode_angle_errors.append(info['angle_error'])
            
            if render and episode < 3:  # Only render first few episodes
                env.render()
        
        episode_rewards.append(episode_reward)
        if episode_angle_errors:
            angle_errors.append(np.mean(episode_angle_errors))
        
        # Get final performance metrics
        perf_metrics = env.get_performance_metrics()
        assistance_amounts.append(perf_metrics.get('assistance_efficiency', 0))
        interference_amounts.append(perf_metrics.get('interference_rate', 0))
    
    # Calculate evaluation metrics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_angle_error': float(np.mean(angle_errors)) if angle_errors else 0.0,
        'std_angle_error': float(np.std(angle_errors)) if angle_errors else 0.0,
        'mean_assistance_efficiency': float(np.mean(assistance_amounts)),
        'mean_interference_rate': float(np.mean(interference_amounts)),
        'success_rate': float(np.mean([err < 0.1 for err in angle_errors])) if angle_errors else 0.0
    }
    
    return metrics


def main():
    """Main training script with command line arguments."""
    parser = argparse.ArgumentParser(description='Train exoskeleton agent')
    
    parser.add_argument('--human-model', type=str, required=True,
                       help='Path to trained human agent model')
    parser.add_argument('--human-normalize', type=str, default=None,
                       help='Path to VecNormalize stats for human agent')
    parser.add_argument('--proprioception', action='store_true', default=True,
                       help='Human was trained with proprioception')
    parser.add_argument('--no-proprioception', action='store_true', default=False,
                       help='Human was trained without proprioception')
    parser.add_argument('--perturb-scale', type=float, default=5.0,
                       help='Perturbation scale used in human training')
    parser.add_argument('--max-assistance', type=float, default=10.0,
                       help='Maximum assistance torque')
    parser.add_argument('--intent-corruption', action='store_true', default=False,
                       help='Enable intent signal corruption during training')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC'],
                       help='RL algorithm to use')
    parser.add_argument('--experiment-name', type=str, default='default',
                       help='Name for this training experiment')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--evaluate', type=str, default=None,
                       help='Path to model to evaluate instead of training')
    
    args = parser.parse_args()
    
    # Handle proprioception flag
    proprioception_enabled = args.proprioception and not args.no_proprioception
    
    if args.evaluate:
        # Evaluation mode
        print(f"Evaluating exoskeleton model: {args.evaluate}")
        metrics = evaluate_exoskeleton_agent(
            model_path=args.evaluate,
            human_model_path=args.human_model,
            human_vec_normalize_path=args.human_normalize,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale,
            intent_corruption=args.intent_corruption
        )
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    else:
        # Training mode
        final_model_path = train_exoskeleton_agent(
            human_model_path=args.human_model,
            human_vec_normalize_path=args.human_normalize,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale,
            max_assistance_torque=args.max_assistance,
            intent_corruption=args.intent_corruption,
            algorithm=args.algorithm,
            experiment_name=args.experiment_name,
            device=args.device
        )
        
        print(f"\nTraining completed! Final model saved to: {final_model_path}")
        
        # Quick evaluation
        print("\nRunning quick evaluation...")
        metrics = evaluate_exoskeleton_agent(
            model_path=final_model_path,
            human_model_path=args.human_model,
            human_vec_normalize_path=args.human_normalize,
            n_episodes=10,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale
        )
        
        print("\nQuick Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()