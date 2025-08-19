# Human-Exoskeleton Training System

A complete implementation of a two-phase training system where a human agent learns elbow control with perturbations, followed by an exoskeleton agent that learns to assist the human.

## Features

- **Modular Proprioception**: Toggle proprioceptive sensing on/off for ablation studies
- **Enhanced Perturbations**: Continuous perturbations during episodes for robustness
- **Intent Signal Processing**: Extract and process human intent with optional corruption
- **Two-Phase Training**: Human → Exoskeleton sequential training pipeline
- **Comprehensive Visualization**: Real-time plotting and performance comparison
- **Configuration Management**: Centralized config system for experiments

## Quick Start

### Phase 1: Train Human Agent

```bash
# Train human with proprioception (default)
python learn_human_agent.py --experiment-name human_with_proprio

# Train human without proprioception
python learn_human_agent.py --no-proprioception --experiment-name human_no_proprio

# Quick test (reduced timesteps)
python learn_human_agent.py --experiment-name quick_test --perturb-scale 2.0
```

### Phase 2: Train Exoskeleton Agent

```bash
# Train exoskeleton to assist human agent
python learn_exoskeleton_agent.py \
    --human-model ./human_agent_training_human_with_proprio/models/human_agent_final \
    --human-normalize ./human_agent_training_human_with_proprio/models/vec_normalize.pkl \
    --experiment-name exo_assistant

# Train with intent corruption for robustness
python learn_exoskeleton_agent.py \
    --human-model ./human_agent_training_human_with_proprio/models/human_agent_final \
    --intent-corruption \
    --experiment-name exo_robust
```

### Visualization and Evaluation

```bash
# Interactive visualization comparing human-only vs human+exoskeleton
python visualise_human_exoskeleton.py \
    --human-model ./human_agent_training_human_with_proprio/models/human_agent_final \
    --exoskeleton-model ./exoskeleton_training_exo_assistant/models/exoskeleton_agent_final \
    --episode-length 1000

# Evaluate trained models
python learn_human_agent.py --evaluate ./path/to/human_model
python learn_exoskeleton_agent.py --evaluate ./path/to/exoskeleton_model --human-model ./path/to/human_model
```

## File Structure

```
RL/
├── proprioception.py           # Modular proprioceptive sensing system
├── human_elbow_env.py          # Enhanced human environment with perturbations
├── learn_human_agent.py        # Human agent training script  
├── intent_signals.py           # Intent signal processing and corruption
├── exoskeleton_env.py          # Exoskeleton-assisted environment
├── learn_exoskeleton_agent.py  # Exoskeleton agent training script
├── config.py                   # Configuration management system
├── visualise_human_exoskeleton.py # Visualization and analysis tools
└── README.md                   # This file
```

## Key Configuration Options

### Human Training
- `--proprioception / --no-proprioception`: Enable/disable proprioceptive sensing
- `--perturb-scale`: Scale of perturbation forces (default: 5.0)
- `--continuous-perturbations`: Apply perturbations during episodes
- `--num-envs`: Number of parallel training environments (default: 4)

### Exoskeleton Training  
- `--max-assistance`: Maximum assistance torque (default: 10.0 Nm)
- `--intent-corruption`: Enable intent signal corruption for robustness
- `--algorithm`: RL algorithm ('PPO' or 'SAC', default: 'PPO')

## Modular Features

### Proprioception System
- **Toggle-able**: Easy enable/disable for comparative studies
- **Multiple modalities**: External force, joint torque, gravity compensation
- **Noise simulation**: Configurable sensor noise levels

### Perturbation System
- **Continuous**: Apply perturbations throughout episodes
- **Configurable**: Adjustable magnitude and frequency
- **Reset-only mode**: Traditional perturbation at episode start only

### Intent Signal Processing
- **Multi-modal**: Muscle activations, joint kinematics, co-contraction
- **Filtering**: Low-pass filtering and smoothing options
- **Corruption**: Noise, delays, dropouts for robustness testing

## Example Experimental Workflows

### Proprioception Ablation Study
```bash
# Train human agents with and without proprioception
python learn_human_agent.py --experiment-name human_with_proprio
python learn_human_agent.py --no-proprioception --experiment-name human_no_proprio

# Train corresponding exoskeleton agents
python learn_exoskeleton_agent.py --human-model ./human_agent_training_human_with_proprio/models/human_agent_final --experiment-name exo_with_proprio
python learn_exoskeleton_agent.py --human-model ./human_agent_training_human_no_proprio/models/human_agent_final --no-proprioception --experiment-name exo_no_proprio
```

### Perturbation Robustness Study
```bash
# Train humans with different perturbation levels
for scale in 2.0 5.0 10.0; do
    python learn_human_agent.py --perturb-scale $scale --experiment-name human_perturb_$scale
done

# Test exoskeleton performance across perturbation levels
for scale in 2.0 5.0 10.0; do
    python learn_exoskeleton_agent.py --human-model ./human_agent_training_human_perturb_5.0/models/human_agent_final --experiment-name exo_test_perturb_$scale
done
```

## Dependencies

- `stable-baselines3`: RL algorithms
- `myosuite`: Musculoskeletal simulation environment  
- `mujoco`: Physics simulation
- `numpy`, `scipy`: Numerical computing
- `matplotlib`: Visualization
- `gymnasium`: Environment interface

## Notes

- Models are automatically saved during training with checkpoints
- TensorBoard logs are generated for monitoring training progress
- All environments support both human muscle control and exoskeleton torque assistance
- The system is designed for easy extension to other joints and assistance modalities