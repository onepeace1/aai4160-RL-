import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from aai4160.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim, # this is just the number of possible actions, not
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(parameters, learning_rate)

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
            
        action = self(ptu.from_numpy(observation)).sample()

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            # HINT: use torch.distributions.Categorical to define the distribution.
            # outout of logits_net is unnormalized probalbity, dimension of action is always 1
            dist = torch.distributions.Categorical(logits=self.logits_net(obs))
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            # HINT: use torch.distributions.Normal to define the distribution.
            dist = torch.distributions.Normal(self.mean_net(obs),torch.exp(self.logstd)) 

        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert obs.shape[0] == actions.shape[0] == advantages.shape[0]

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        # HINT: don't forget to do `self.optimizer.step()`!
        logp=self.calc_logp(obs,actions)
        loss = -torch.mean(logp*advantages) # since objective is max and gradient is substracted in sgd, we should multiply -1
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(loss),
            "logp": ptu.to_numpy(torch.mean(logp)),
            "advantages": ptu.to_numpy(torch.mean(advantages)),
            
        }

    def ppo_update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_logp: np.ndarray,
        ppo_cliprange: float = 0.2,
    ) -> dict:
        """Implements the policy gradient actor update."""
        assert obs.ndim == 2
        assert advantages.ndim == 1
        assert old_logp.ndim == 1
        assert advantages.shape == old_logp.shape

        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        old_logp = ptu.from_numpy(old_logp)

        # TODO: Implement the ppo update.
        # HINT: calculate logp first, and then caculate ratio and clipped loss.
        # HINT: ratio is the exponential of the difference between logp and old_logp.
        # HINT: You can use torch.clamp to clip values.
        
        logp=self.calc_logp(obs,actions)
        old_cur_ratio=torch.exp(logp-old_logp) # N by 1
        clipped_ratio=torch.clamp(old_cur_ratio,1-ppo_cliprange,1+ppo_cliprange)
        non_ppo_loss=torch.sum(logp*advantages) # N by 1
        #loss=-torch.mean(torch.min(old_cur_ratio*non_ppo_loss,clipped_ratio*non_ppo_loss))
        #loss = -torch.mean(torch.min(old_cur_ratio * advantages, clipped_ratio * advantages))
        loss=-torch.mean(old_cur_ratio*advantages)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"PPO Loss": ptu.to_numpy(loss)}
    
    # return batch*dimesnion of action(for discrete, number of possible action since always number)
    def calc_logp(self,
        obs: torch.Tensor,
        actions: torch.Tensor) :

        logp=self(obs).log_prob(actions)
        
        if logp.ndim==2:
            logp=torch.sum(logp,dim=1)
        assert logp.ndim == 1 and logp.shape[0] == obs.shape[0]
        
        return logp
