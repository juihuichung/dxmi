"""
Diffusion by MaxEnt IRL Trainer implemented by Christina, Sabrina, Jui-Hui
DxMI trainer, is a single network that represents both value and energy. 
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ..difusion import extract, make_beta_schedule
from ..modules import process_single_t
from models.DxMI.openai_diffusion import OpenAIDiffusion
from models.cm.nn import append_dims
from typing import Optional, Sequence, Union, Dict, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class DxMIConfig:
    batch_size: int
    n_timesteps: int = 10
    tau1: float = 0.0
    tau2: float = 0.0
    gamma: Optional[float] = None
    q_beta_schedule: str = 'constant'
    q_beta_start: float = 1.0
    q_beta_end: float = 1.0
    adavelreg: Optional[float] = None
    value_update_order: str = 'backward'
    entropy_in_value: Optional[int] = None
    velocity_in_value: Optional[int] = None
    use_sampler_beta: bool = False
    time_cost: Optional[float] = None
    time_cost_sig: Optional[float] = None
    repeat_value_update: int = 1
    value_resample: bool = False
    value_grad_clip: bool = False
    skip_sampler_tau: int = 0


class DxMI_Trainer:
    def __init__(self, config: DxMIConfig):
        self.cfg = config
        self._init_beta_schedule()

        # placeholders for the models and optimizers
        self.f = self.v = self.sampler = None
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

    def _init_beta_schedule(self) -> None:
        cfg = self.cfg
        if cfg.use_sampler_beta:
            # defer loading until sampler is set
            self.betas_for_q = None
        else:
            self.betas_for_q = make_beta_schedule(
                schedule=cfg.q_beta_schedule,
                n_timesteps=cfg.n_timesteps,
                start=cfg.q_beta_start,
                end=cfg.q_beta_end,
            )

    def set_models(
        self, 
        energy_fn: Optional[torch.nn.Module],
        value_fn: torch.nn.Module,
        sampler: Any,
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        """
        Set the models and optimizers for training.
        """
        self.f = energy_fn
        self.v = value_fn
        self.sampler = sampler
        self.optimizers = optimizers

        if self.cfg_use_sampler_beta:
            # use the sampler's beta schedule
            self.load_sampler_betas()
        
    def load_sampler_betas(self) -> None:
        """
        Load the beta schedule from the sampler.
        """
        s = self.sampler
        if hasattr(s, 'user_defined_eta'):
            betas = torch.tensor(s.user_defined_eta, dtype=torch.float32)
        elif hasattr(s, 'log_betas'):
            betas = torch.exp(s.net.log_betas).detach()
        self.betas_for_q = betas
        print("Loaded beta schedule from sampler:", betas)


    def get_running_cost(
        self, 
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute the running cost from line 7:  
        tau / (2 * s_t^2) * (x_t - x_{t+1})^2
        """
        idx = self.cfg.n_timesteps - t - 1
        beta_next = extract(self.betas_for_q, idx, x_t).to(x_t.device)
        cost = ((x_tp1 - x_t)**2) / (2 * beta_next)
        return cost.view(cost.size(0), -1).mean(dim=1)

    def sample_guidance(
        self, 
        num_samples: int,
        device = torch.device,
        x0: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        t_select: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Sample from the model using the guidance scale.
        """
        self.v.eval()
        x = x0 if x0 is not None else torch.randn(num_samples, *self.sampler.sample_shape, device=device)
        scale = guidance_scale or getattr(self.sampler, 'guidance_scale', 1.0)
        
        samples, logs, orig_logs, guides = [], [], [], []

        for t in range(self.cfg.n_timesteps):
            tt = process_single_t(x, t)
            with torch.no_grad():
                step = self.sampler.sample_step(x, tt)
            next_x = step['x_next'].detach().requires_grad_()

            val = self.v(next_x, tt+1).squeeze()
            grad = torch.autograd(val.sum(), next_x)[0]
            guide = grad * scale * step['sigma']

            x = next_x + guide if (t_select is None or t in t_select) else next_t
            orig_dist = Normal(step['mean', step['sigma']])
            logs.apend(step.get('logp', torch.zeros_like(val)))
            orig_logs.append(orig_dist.log_prob(x).mean(dim=list(range(1, x.ndim))))
            guides.append(guide)
            samples.append(x.detach().clone())

        return {
            'sample': x,
            'l_sample': samples,
            'logp_traj': torch.stack(logs).sum(0),
            'logp_on_traj': torch.stack(orig_logs).sum(0),
            'guidance': guides,
        }

    def update_adaptive_vel_reg(self, d_sample: Dict[str, Any]) -> None:
        """
        update betas_for_q
        """
        samples = torch.stack(d_sample['l_sample'])
        # from t=0 to t=T-1
        diffs = (samples[1:] - samples[:-1]**2)
        stats = diffs.view(diffs.size(0), -1).mean(1).flip(0).to(samples.device)
    
        # s_t^2 = a * s_t^2 + (1-a) * (x_t - x_{t-1})^2
        self.betas_for_q = (
            self.betas_for_q.to(samples.device) * self.cfg.adavelreg
            + (1 - self.cfg.adavelreg) * stats
        ).detach()
    
    def update_f_v(
        self, 
        img: torch.Tensor,
        d_sample: Dict[str, Any],
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        cfg = self.cfg
        device = img.device
        if cfg.adavelreg is not None:
            self.update_adaptive_reg(d_sample)
        
        # ENERGY UPDATE
        # set both energy and value function to training mode
        if self.f is not None:
            self.f.train()
        self.v.train()
        # treat the last step of value function as energy
        x0 = d_sample['l_sample'][-1]
        inputs = torch.cat((img.detach(), x0.detach()), 0)
        T = cfg.n_timesteps * torch.ones(inputs.size(0), dtype=torch.long, device=device)
        
        output = self.f(inputs if self.f is not None else self.v(inputs, T))

        # energy of real samples, energy of generated samples
        pos_e, neg_e = output[:img.size(0)], output[img.size(0):]
        # line 5: E(x) - E(x_T)
        d_loss = pos_e.mean() - neg_e.mean()
        if cfg.gamma is not None:
            # line 5: E(x) - E(x_T) + gamma * (E(x)^2 + E(x_T)^2)
            reg = pos_e.pow(2).mean() + neg_e.pow(2).mean()
            d_loss = d_loss + cfg.gamma * reg
        else:
            reg = torch.tensor(0.0, device=device)
        
        energy_opt = (
            self.optimizers['value'] if self.f is None else self.optimizers['energy']                
        )
        energy_opt.zero_grad()
        d_loss.backward()
        energy_opt.step()
        if self.f is not None:
            self.f.eval()


        # VALUE UPDATE
        # temporal difference value estimation
        metrics: Dict[str, float] = {}
        batch_size = cfg.batch_size 
        n_steps = cfg.n_timesteps
        indicies = torch.randperm(state_dict['state'].size(0))

        for i in range(n_steps):
            t_rev = n_steps - i - 1
            mask = state_dict['timestep'][indicies] == t_rev
            idx = indicies[mask]
            if idx.numel == 0:
                continue
            state = state_dict['state'][idx]
            timestep = state_dict['timestep'][idx]

            if cfg.value_resample:
                step = self.sampler.sample_step(state, timestep)
                next_state, mean, sigma = step['sample'], step['mean'], step['sigma']
            else:
                next_state = state_dict['next_state'][idx]
                mean = state_dict['mean'][idx]
                sigma = state_dict['sigma'][idx]
            # line 7: tau / (2 * s_t^2) * (x_t - x_{t+1})^2, velocity cost
            running_cost = self.get_running_cost(state, next_state, mean, sigma, t_rev)
            # H = -log(sigma_t)
            # Line 7: tau * log(pi(x_{t + 1} | x_t)), simplifies to log(sigma_t) since Gaussian
            entropy = torch.log(sigma.squeeze())

            with torch.no_grad():
                if i == n_steps - 1 and self.f is not None:
                    # use energy for the last step
                    v_tp1 = self.f(next_state).squeeze()
                else:
                    v_tp1 = self.v(next_state, timestep + 1).squeeze()

            target = v_tp1 # target = V^(t+1)(x_{t + 1})
            # encourages agent to move more quickly (not mentioned in paper)
            if cfg.time_cost_sig is not None:
                center = n_steps // 2
                target = target + cfg.time_cost_sig * (
                    torch.sigmoid(-timestep + center)
                    - torch.sigmoid(-timestep - 1 + center)
                )
            # encourages agent to terminate earlier
            if cfg.time_cost is not None:
                target = target + cfg.time_cost
            if cfg.velocity_in_value is not None:
                non_term = (timestep < n_steps - cfg.velocity_in_value).float()
                # line 7: tau / (2 * s_t^2) * (x_t - x_{t+1})^2
                target = target + running_cost * cfg.tau2 * non_term
            if cfg.entropy_in_value is not None:
                # do not add entropy to target for the "entropy_in_value" number of last steps 
                non_term_e = (timestep < n_steps - cfg.entropy_in_value).float()
                # Line 7: tau * log(pi(x_{t + 1} | x_t)), simplifies to log(sigma_t) since Gaussian
                target = target - entropy * cfg.tau1 * non_term_e

            # Line 7: V^t(x_t)
            v_loss = F.mse_loss(self.v(state, timestep).squeeze(), target.detch())
            value_opt = self.optimizers['value']
            value_opt.zero_grad()
            v_loss.backward()
            if cfg.value_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.1)
            value.opt.step()

            metrics[f'running_cost_step_{t_rev}'] = running_cost.mean().item()
            metrics[f'value_step_{t_rev}'] = self.v(state, timestep).mean().item()

        # energy metrics
        metrics.update({
            'd_loss': d_loss.item(), 
            'reg': reg.item(),
        })
        return metrics

    ### SAMPLER UPDATE ###
    def update_sampler(
        self, 
        state_dict: Dict[str, torch.Tensor], 
        n_generator: int,
        d_sample: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        cfg = self.cfg
        self.sampler.train()
        self.v.eval()
        if self.f is not None:
            self.f.eval()
        
        perm = torch.randperm(state_dict['state'].size(0))
        max_data = min(perm.numel(), cfg.batch_size * n_generator)
        sampler_opt = self.optimizers['sampler']
        metrics: Dict[str, float] = {}

        for start in range(0, max_data, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            state = state_dict['state'][idx]
            t = state_dict['timestep'][idx]
            step = self.sampler.sample_step(state, t)
            next_state, mean, sigma = step['sample'], step['mean'], step['sigma']
            
            running_cost = self.get_running_cost(state, next_state, mean, sigma, t)
            entropy = torch.log(sigma.squeeze())

            if self.f is None: # use value function if no energy model
                val_loss = self.v(next_state, t + 1).squeeze()
            else: 
                # use value function if none of samples are at final step
                # use energy function if all of samples are at final step
                mask_final = t == cfg.n_timesteps - 1
                f_vals = self.f(next_state[mask_final]).flatten() if mask_final.any() else torch.tensor([])
                v_vals = self.v(next_state[~mask_final], t[~mask_final] + 1).flatten() if (~mask_final).any() else torch.tensor([])
                val_loss = torch.cat((f_vals, v_vals)).mean()

            non_term = (t < cfg.n_timesteps - cfg.skip_sampler_tau).float()
            # Line 11: V^(t+1)(x_{t+1}) + tau * log(pi(x_{t + 1} | x_t)) + velocity cost
            sampler_loss = (
                val_loss + ((running_cost * cfg.tau2) - (entropy * cfg.tau1)) * non_term
            ).mean()
            
            sampler_opt.zero_grad()
            sampler_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), 0.1)
            sampler_opt.step()

            metrics['sampler_loss'] = sampler_loss.item()
            metrics['value_loss'] = val_loss.item()
            metrics['running_cost'] = running_cost.mean().item()
            metrics['entropy'] = entropy.mean().item()

        if getattr(self.sampler, 'trainable_beta', False):
            logb = getattr(self.sampler.net, 'log_betas', None) or getattr(self.sampler.net.module, 'log_betas', None)
            if logb is not None:
                for i, val in enumerate(torch.exp(logb)):
                    metrics[f'sigma_{i}'] = val.item()
        
        return metrics
            




# Buffer functions

def append_buffer(staet_buffer: Dict[str, torch.Tensor], d_sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    x_seq = d_sample['l_sample']
    n_sample = x_seq[0].size(0)
    n_seq = len(x_seq) - 1
    device = x_seq[0].device()

    for t in range(n_seq):
        state_buffer['state'] = torch.cat((state_buffer['state'], x_seq[t].detach()))
        state_buffer['next_state'] = torch.cat((state_buffer['next_state'], x_seq[t + 1].detach()))
        state_buffer['timestep'] = torch.cat((state_buffer['timestep'], torch.full((n_sample,), t, dtype=torch.long, device=device)))
        state_buffer['final'] = torch.cat((state_buffer['final'], x_seq[-1].detach()))
        if 'logp' in d_sample:
            state_buffer['logp'] = torch.cat((state_buffer['logp'], d_sample['logp'][t].detach()))
        if 'control' in d_sample:
            state_buffer['control'] = torch.cat((state_buffer['control'], d_sample['control'][t].detach()))
        if 'entropy' in d_sample:
            state_buffer['entropy'] = torch.cat((state_buffer['entropy'], d_sample['entropy'][t].detach()))
        if 'mean' in d_sample:
            state_buffer['mean'] = torch.cat((state_buffer['mean'], d_sample['mean'][t].detach()))
        if 'sigma' in d_sample:
            state_buffer['sigma'] = torch.cat((state_buffer['sigma'], d_sample['sigma'][t].detach()))
        if 'y' in d_sample:
            state_buffer['y'] = torch.cat((state_buffer['y'], d_sample['y'].detach()))
    return state_buffer

def reset_buffer(device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        'state': torch.empty((0,), dtype=torch.float32, device=device),
        'next_state': torch.empty((0,), dtype=torch.float32, device=device),
        'timestep': torch.empty((0,), dtype=torch.long, device=device),
        'final': torch.empty((0,), dtype=torch.float32, device=device),
        'logp': torch.empty((0,), dtype=torch.float32, device=device),
        'control': torch.empty((0,), dtype=torch.float32, device=device),
        'entropy': torch.empty((0,), dtype=torch.float32, device=device),
        'mean': torch.empty((0,), dtype=torch.float32, device=device),
        'sigma': torch.empty((0,), dtype=torch.float32, device=device),
        'y': torch.empty((0,), dtype=torch.long, device=device),
    }


        



