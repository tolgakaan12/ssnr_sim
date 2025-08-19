"""
Human Agent Training Script for Elbow Control with Proprioception

This script trains a human agent to control elbow angle with:
- Enhanced perturbations during episodes
- Proprioceptive sensing (modular, can be disabled)
- Target angle tracking
- Comprehensive evaluation and logging
"""

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import numpy as np
import os
from typing import Dict, Any
import argparse

from human_elbow_env import HumanElbowEnv


# Training configuration
EPISODE_LENGTH = 1500  # Increased for more complex task
TOTAL_TIMESTEPS = 800_000  # Extended training
EVAL_FREQ = 10_000
CHECKPOINT_FREQ = 50_000


def make_env(rank: int = 0, 
             proprioception_enabled: bool = True,
             perturb_scale: float = 5.0,
             continuous_perturbations: bool = True,
             seed: int = 42):
    """
    Create a monitored human elbow environment.
    
    Args:
        rank: Environment rank for parallel training
        proprioception_enabled: Whether to include proprioceptive sensing
        perturb_scale: Scale of perturbation forces
        continuous_perturbations: Whether to apply perturbations during episodes
        seed: Random seed
        
    Returns:
        Wrapped environment function
    """
    def _init():
        # Create enhanced human environment
        env = HumanElbowEnv(
            perturb_scale=perturb_scale,
            proprioception_enabled=proprioception_enabled,
            continuous_perturbations=continuous_perturbations,
            perturbation_frequency=0.01,  # 1% chance per step
            target_change_frequency=0.005,  # 0.5% chance per step
            seed=seed + rank
        )
        
        # Apply time limit
        env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)
        
        # Monitor for logging
        env = Monitor(env)
        
        return env
    
    set_random_seed(seed + rank)
    return _init


def create_training_directories(experiment_name: str) -> Dict[str, str]:
    """
    Create directories for training outputs.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary of directory paths
    """
    base_dir = f"./human_agent_training_{experiment_name}"
    dirs = {
        'base': base_dir,
        'models': f"{base_dir}/models",
        'logs': f"{base_dir}/logs", 
        'tensorboard': f"{base_dir}/tensorboard",
        'eval': f"{base_dir}/eval"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def train_human_agent(proprioception_enabled: bool = True,
                     perturb_scale: float = 5.0,
                     continuous_perturbations: bool = True,
                     num_envs: int = 4,
                     experiment_name: str = "default",
                     device: str = 'cpu',
                     verbose: bool = True) -> str:
    """
    Train the human agent with specified configuration.
    
    Args:
        proprioception_enabled: Whether to include proprioceptive sensing
        perturb_scale: Scale of perturbation forces
        continuous_perturbations: Whether to apply perturbations during episodes
        num_envs: Number of parallel environments
        experiment_name: Name for this training run
        device: Device to use for training ('cpu' or 'cuda')
        verbose: Whether to print training progress
        
    Returns:
        Path to the saved final model
    """
    # Create directories
    dirs = create_training_directories(experiment_name)
    
    if verbose:
        print(f"Starting human agent training: {experiment_name}")
        print(f"Proprioception: {proprioception_enabled}")
        print(f"Perturbation scale: {perturb_scale}")
        print(f"Continuous perturbations: {continuous_perturbations}")
        print(f"Number of environments: {num_envs}")
    
    # Create vectorized training environments
    env_fns = [make_env(
        rank=i,
        proprioception_enabled=proprioception_enabled,
        perturb_scale=perturb_scale,
        continuous_perturbations=continuous_perturbations
    ) for i in range(num_envs)]
    
    env = DummyVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(
        rank=999,  # Different seed for evaluation
        proprioception_enabled=proprioception_enabled,
        perturb_scale=perturb_scale,
        continuous_perturbations=continuous_perturbations,
        seed=12345
    )])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env = VecMonitor(eval_env)
    
    # Configure PPO model
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
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1 if verbose else 0,
        tensorboard_log=dirs['tensorboard']
    )
    
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
        name_prefix='human_agent_checkpoint'
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # Train the model
    if verbose:
        print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=verbose
    )
    
    # Save final model
    final_model_path = f"{dirs['models']}/human_agent_final"
    model.save(final_model_path)
    
    # Save training configuration
    config = {
        'proprioception_enabled': proprioception_enabled,
        'perturb_scale': perturb_scale,
        'continuous_perturbations': continuous_perturbations,
        'num_envs': num_envs,
        'total_timesteps': TOTAL_TIMESTEPS,
        'episode_length': EPISODE_LENGTH,
        'experiment_name': experiment_name
    }
    
    import json
    with open(f"{dirs['base']}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save environment normalization stats
    env.save(f"{dirs['models']}/vec_normalize.pkl")
    
    if verbose:
        print(f"Training completed! Models saved to: {dirs['models']}")
        print(f"Final model: {final_model_path}")
    
    return final_model_path


def evaluate_human_agent(model_path: str,
                         n_episodes: int = 100,
                         proprioception_enabled: bool = True,
                         perturb_scale: float = 5.0,
                         render: bool = False) -> Dict[str, float]:
    """
    Evaluate a trained human agent.
    
    Args:
        model_path: Path to the trained model
        n_episodes: Number of evaluation episodes
        proprioception_enabled: Whether proprioception was used in training
        perturb_scale: Perturbation scale used in training
        render: Whether to render episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = HumanElbowEnv(
        perturb_scale=perturb_scale,
        proprioception_enabled=proprioception_enabled,
        continuous_perturbations=True
    )
    env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)
    
    # Run evaluation episodes
    episode_rewards = []
    angle_errors = []
    total_perturbations = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_angle_errors = []
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Track angle error
            if 'angle_error' in obs:
                episode_angle_errors.append(abs(obs['angle_error'][0]))
            
            if render and episode < 5:  # Only render first few episodes
                env.render()
        
        episode_rewards.append(episode_reward)
        if episode_angle_errors:
            angle_errors.append(np.mean(episode_angle_errors))
        if 'total_perturbation' in info:
            total_perturbations.append(info['total_perturbation'])
    
    # Calculate metrics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_angle_error': float(np.mean(angle_errors)) if angle_errors else 0.0,
        'std_angle_error': float(np.std(angle_errors)) if angle_errors else 0.0,
        'mean_perturbation': float(np.mean(total_perturbations)) if total_perturbations else 0.0,
        'success_rate': float(np.mean([err < 0.1 for err in angle_errors])) if angle_errors else 0.0
    }
    
    return metrics


def main():
    """Main training script with command line arguments."""
    parser = argparse.ArgumentParser(description='Train human agent for elbow control')
    
    parser.add_argument('--proprioception', action='store_true', default=True,
                       help='Enable proprioceptive sensing')
    parser.add_argument('--no-proprioception', action='store_true', default=False,
                       help='Disable proprioceptive sensing')
    parser.add_argument('--perturb-scale', type=float, default=5.0,
                       help='Scale of perturbation forces')
    parser.add_argument('--continuous-perturbations', action='store_true', default=True,
                       help='Enable continuous perturbations during episodes')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments')
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
        print(f"Evaluating model: {args.evaluate}")
        metrics = evaluate_human_agent(
            model_path=args.evaluate,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale,
            render=False
        )
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    else:
        # Training mode
        final_model_path = train_human_agent(
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale,
            continuous_perturbations=args.continuous_perturbations,
            num_envs=args.num_envs,
            experiment_name=args.experiment_name,
            device=args.device
        )
        
        print(f"\nTraining completed! Final model saved to: {final_model_path}")
        
        # Quick evaluation
        print("\nRunning quick evaluation...")
        metrics = evaluate_human_agent(
            model_path=final_model_path,
            n_episodes=20,
            proprioception_enabled=proprioception_enabled,
            perturb_scale=args.perturb_scale
        )
        
        print("\nQuick Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()