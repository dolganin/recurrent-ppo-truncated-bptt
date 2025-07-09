import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class ActorCriticModel(nn.Module):
    def __init__(self, config, device, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape
        self.svd_rank_frac   = config.get("svd_rank_frac", None)
        self.tt_rank_frac    = config.get("tt_rank_frac",  None)
        self.gauss_filter    = config.get("gauss_filter", False)
        self.laplace_filter  = config.get("laplace_filter", False)
        self.enable_vae      = config.get("enable_vae", False)

        if self.svd_rank_frac:   print(f"SVD frac  = {self.svd_rank_frac}")
        if self.tt_rank_frac:    print(f"TT  frac  = {self.tt_rank_frac}")
        if self.gauss_filter:    print("Gaussian filter will be applied")
        if self.laplace_filter:  print("Laplacian filter will be applied")
        if self.enable_vae:      print("VAE filter ENABLED")

        self.device = device

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.to(self.device)
            self.conv_out_size = self.get_conv_output(observation_space.shape, self.device)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        # Recurrent layer (GRU or LSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        
        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))


        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)
        
    def svd_low_rank_safe(self, x: torch.Tensor,
                          energy_frac: float = 1.0, max_try: int = 3) -> torch.Tensor:
        orig_shape = x.shape                    # (..., D)
        x_2d = x.reshape(-1, x.shape[-1])       # (M, D)   без батча
    
        for i in range(max_try):
            try:
                U, S, Vh = torch.linalg.svd(x_2d, full_matrices=False)
                break
            except RuntimeError:
                if i == max_try - 1:
                    return x
                x_2d = x_2d.cpu().double()
    
        if energy_frac < 1.0:
            energy = torch.cumsum(S ** 2, dim=-1) / (S ** 2).sum()
            k = int((energy < energy_frac).sum() + 1)
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    
        x_hat = (U * S.unsqueeze(-2)) @ Vh       # (M, D)
        return x_hat.reshape(orig_shape)

    def tt_low_rank_safe(self, memory: torch.Tensor,
                         energy_frac: float = 1.0,
                         max_rank: int = 200) -> torch.Tensor:
        """
        TT-аппроксимация с удержанием заданной доли энергии.
        Возвращает dense-тензор той же формы.
        """
        if energy_frac >= 1.0:
            return memory                               # ничего не сжимаем
    
        shape = memory.shape
        flat  = memory.reshape(-1, shape[-2], shape[-1])
        full_norm = torch.linalg.norm(flat)
    
        for r in range(1, max_rank + 1):
            mem_tt_r = tn.Tensor(flat, ranks_tt=r)      # построить TT ранга r
            approx   = mem_tt_r.torch()
            if torch.linalg.norm(approx) / full_norm >= energy_frac:
                return approx.reshape_as(memory)

        return approx.reshape_as(memory)

    def apply_gaussian_filter(self, memory: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Gaussian 1D-фильтр по последней размерности (D) → shape: (B, blocks, D)
        """
        B, blocks, D = memory.shape
        kernel_size = int(torch.ceil(torch.tensor(6 * sigma))) | 1
        half = (kernel_size - 1) // 2
        x = torch.arange(-half, half + 1, dtype=memory.dtype, device=memory.device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
    
        mem = memory.view(B * blocks, 1, D)                      # (B*blocks, 1, D)
        smoothed = F.conv1d(mem, kernel, padding=half)
        return smoothed.view(B, blocks, D)

    def apply_laplacian_filter(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Лаплас-фильтр по последней оси (фичи D).
        Работает для памяти (B, blocks, D) и (B, L, blocks, D).
        """
        shift_left  = torch.roll(memory,  1, dims=-1)
        shift_right = torch.roll(memory, -1, dims=-1)
        laplacian = shift_left + shift_right - 2 * memory
        return memory + laplacian

    
        e2 = S.square(); thr = e2.sum()*energy_frac
        k = int((e2.cumsum(0) < thr).sum().clamp(min=1))
        return (U[:, :k]*S[:k]) @ Vh[:k]

    def forward(self, obs:torch.tensor, recurrent_cell:torch.tensor, device:torch.device, sequence_length:int=1):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        dev = next(self.parameters()).device        # где лежат веса
        h = obs.to(dev)
    
        if isinstance(recurrent_cell, tuple):       # LSTM
            recurrent_cell = tuple(t.to(dev) for t in recurrent_cell)
        else:                                       # GRU
            recurrent_cell = recurrent_cell.to(dev)
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Forward reccurent layer (GRU or LSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1) # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent laye
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])
            hidden, memory = recurrent_cell
            
            if self.svd_rank_frac is not None:
                approx = self.svd_low_rank_safe(memory, self.svd_rank_frac)
                memory = approx + (memory - approx).detach()
            elif self.tt_rank_frac is not None:
                approx = self.tt_low_rank_safe(memory, self.tt_rank_frac)
                memory = approx + (memory - approx).detach()
            elif self.gauss_filter:
                memory = self.apply_gaussian_filter(memory)
            elif self.laplace_filter:
                memory = self.apply_laplacian_filter(memory)

                
        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, recurrent_cell

    def get_conv_output(self, shape:tuple, device=None) -> int:
        dummy = torch.zeros(1, *shape, device=device)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

 
    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        return hxs, cxs