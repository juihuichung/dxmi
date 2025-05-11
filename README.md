# DxMI

This repository reproduces the **DxMI** (Diffusion by Maximum Entropy Inverse RL) algorithm from [(Yoon et al., 2024)](https://arxiv.org/abs/2407.00626$0). 

# CIFAR-10 Experiments

We chose to reproduce the the experiments from the paper on the CIFAR-10 dataset to determine the paper's reproducability. Because a full image generation codebase is quite hefty, we chose to only reimplement the file (cifar10/models/DxMI/trainer.py), which follows the key reinforcement learning DxMI algorithm from the paper. 


# MNIST Experiments

Instead of the original U-Net, we use simple MLPs for both the diffusion (policy) network and the value (energy) network to facilitate fast ablations in (diffusion_by_maximum_entropy_irl.ipynb).


- **Goal**: Fine-tune a pretrained diffusion model via maximum-entropy inverse RL.
- **Key idea**: Treat the reverse diffusion process as a policy and learn a reward (energy) function that recovers expert trajectories under maximum entropy.
- **Our setup**:  
  1. Train a [DDPM](https://arxiv.org/abs/2006.11239$0) diffusion model on MNIST with 1000 timesteps in reverse process.  
  2. Use [FastDPM](https://arxiv.org/abs/2106.00132$0) to shrink the variance schedule to **5 timesteps**.  
  3. Perform reinforcement learning to finetune the diffusion model, using algorithm similar to the [Soft Actor-Critic](https://arxiv.org/abs/1801.01290$0) method with energy-based reward model.  
  4. Compute FID features to track training.


We further add a target network on the temporal-difference update to stabilize training, maintaining a delayed copy of the value network parameters when computing TD targets

## Hyperparameters

```
# Experiment settings
seed                     = 42          # Random seed for reproducibility
data_size                = (1, 8, 8)   # (channels, height, width)
limit_samples            = 20000       # Max samples for training
label                    = None        # 0–9 or None (all digits)
train_batch_size         = 128         # Batch size

# DDPM pretraining
diffusion_num_timesteps  = 1000        # Reverse-process timesteps
diffusion_num_epochs     = 500         # Epochs for DDPM
diffusion_learning_rate  = 1e-3        # Optimizer LR
diffusion_embedding_size = 100         # Embedding dim
diffusion_hidden_size    = 512         # MLP hidden size
diffusion_hidden_layers  = 5           # MLP layers

# FastDPM reduction
n_reduced_timesteps      = 5           # Timesteps after reduction

# DxMI finetuning
dxmi_num_epochs          = 1000        # Epochs for DxMI
learning_rate_policy     = 1e-7        # Policy network LR
learning_rate_value      = 1e-6        # Value network LR
dxmi_target_network      = False       # Use target network?
dxmi_entropy_coef        = 0.0         # Entropy regularizer
dxmi_cost_coef           = 0.1         # Cost/loss coefficient
dxmi_gradient_clipping   = 0.0         # Gradient clipping (0 = none)
```
