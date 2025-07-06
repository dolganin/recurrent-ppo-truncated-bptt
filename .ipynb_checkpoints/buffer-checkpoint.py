from gymnasium import spaces
import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies. """
    def __init__(self, config:dict, observation_space:spaces.Box, action_space_shape:tuple, device:torch.device) -> None:
        """
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.n_workers = config["n_workers"]
        self.worker_steps = config["worker_steps"]
        self.n_mini_batches = config["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batches
        hidden_state_size = config["recurrence"]["hidden_state_size"]
        self.layer_type = config["recurrence"]["layer_type"]
        self.sequence_length = config["recurrence"]["sequence_length"]
        self.true_sequence_length = 0

        # Initialize the buffer's data storage
        self.rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), dtype=torch.long).to(self.device)
        self.dones = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        self.obs        = torch.zeros((self.n_workers, self.worker_steps) + observation_space.shape, device=self.device)
        self.hxs        = torch.zeros((self.n_workers, self.worker_steps, hidden_state_size), device=self.device)
        self.cxs        = torch.zeros_like(self.hxs)
        self.log_probs  = torch.zeros((self.n_workers, self.worker_steps, len(action_space_shape)), device=self.device)
        self.values     = torch.zeros((self.n_workers, self.worker_steps), device=self.device)
        self.advantages = torch.zeros_like(self.values)


    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "obs": self.obs,
            "actions": self.actions,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool)
        }
        
        # Add data concerned with the memory based on recurrence and arrange the entire training data into sequences
        max_sequence_length = 1

        # The loss mask is used for masking the padding while computing the loss function.
        samples["loss_mask"] = torch.ones((self.n_workers, self.worker_steps), dtype=torch.bool)

        # Add collected recurrent cell states to the dictionary
        # Add collected recurrent cell states to the dictionary
        samples["hxs"] =  self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        # Split data into sequences and apply zero-padding
        # Retrieve the indices of dones as these are the last step of a whole episode
        episode_done_indices = []
        for w in range(self.n_workers):
            episode_done_indices.append(list(self.dones[w].nonzero()[0]))
            # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
            if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                episode_done_indices[w].append(self.worker_steps - 1)

        # Retrieve unpadded sequence indices
        self.flat_sequence_indices = np.asarray(self._arange_sequences(
                    np.arange(0, self.n_workers * self.worker_steps).reshape(
                        (self.n_workers, self.worker_steps)), episode_done_indices)[0], dtype=object)
        
        # Split vis_obs, vec_obs, recurrent cell states and actions into episodes and then into sequences
        for key, value in samples.items():
            # Split data into episodes or sequences
            sequences, max_sequence_length = self._arange_sequences(value, episode_done_indices)

            # Apply zero-padding to ensure that each episode has the same length
            # Therfore we can train batches of episodes in parallel instead of one episode at a time
            for i, sequence in enumerate(sequences):
                sequences[i] = self._pad_sequence(sequence, max_sequence_length)

            # Stack sequences (target shape: (Sequence, Step, Data ...) & apply data to the samples dict
            samples[key] = torch.stack(sequences, axis=0)

            if (key == "hxs" or key == "cxs"):
                # Select the very first recurrent cell state of a sequence and add it to the samples
                samples[key] = samples[key][:, 0]

        # Store important information
        self.num_sequences = len(sequences)
            
        self.actual_sequence_length = max_sequence_length
        
        # Add remaining data samples
        samples["values"] = self.values
        samples["log_probs"] = self.log_probs
        samples["advantages"] = self.advantages

        # Flatten samples
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    def _pad_sequence(self, sequence:torch.Tensor, target_length:int) -> torch.Tensor:
        """Pads a sequence to the target length using zeros.
    
        Arguments:
            sequence {torch.Tensor} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence
    
        Returns:
            {torch.Tensor} -- Returns the padded sequence
        """
        delta_length = target_length - len(sequence)
        if delta_length <= 0:
            return sequence
    
        # создаем паддинг на том же устройстве
        device = sequence.device
        if len(sequence.shape) > 1:
            padding = torch.zeros((delta_length, *sequence.shape[1:]),
                                  dtype=sequence.dtype,
                                  device=device)
        else:
            padding = torch.zeros(delta_length,
                                  dtype=sequence.dtype,
                                  device=device)
    
        return torch.cat((sequence, padding), dim=0)


    def _arange_sequences(self, data, episode_done_indices):
        """Splits the povided data into episodes and then into sequences.
        The split points are indicated by the envrinoments' done signals.
        
        Arguments:
            data {torch.tensor} -- The to be split data arrange into num_worker, worker_steps
            episode_done_indices {list} -- Nested list indicating the indices of done signals. Trajectory ends are treated as done
            
        Returns:
            {list} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        for w in range(self.n_workers):
            start_index = 0
            for done_index in episode_done_indices[w]:
                # Split trajectory into episodes
                episode = data[w, start_index:done_index + 1]
                # Split episodes into sequences
                if self.sequence_length > 0:
                    for start in range(0, len(episode), self.sequence_length):
                        end = start + self.sequence_length
                        sequences.append(episode[start:end])
                else:
                    # If the sequence length is not set to a proper value, sequences will be based on episodes
                    sequences.append(episode)
                max_length = len(episode) if len(episode) > max_length else max_length
                start_index = done_index + 1
        return sequences, max_length

    def recurrent_mini_batch_generator(self) -> dict:
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batches
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += 1
    
        # 🔧 сделать индексы сразу на нужном устройстве
        indices = torch.arange(
            0, self.num_sequences * self.actual_sequence_length,
            device=self.device
        ).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences, device=self.device)
    
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
    
            # 🔧 явно переводим unpadded индексы на self.device
            mini_batch_unpadded_indices = self.flat_sequence_indices[sequence_indices[start:end].tolist()]
            mini_batch_unpadded_indices = torch.tensor(
                [i for sublist in mini_batch_unpadded_indices for i in sublist],
                device=self.device, dtype=torch.long
            )
    
            mini_batch = {}
            for key, value in self.samples_flat.items():
                value = value.to(self.device)  # 🔧 обязательно перенести value
    
                if key in ("hxs", "cxs"):
                    mini_batch[key] = value[sequence_indices[start:end]]
                elif key in ("log_probs", "advantages", "values"):
                    mini_batch[key] = value[mini_batch_unpadded_indices]
                else:
                    mini_batch[key] = value[mini_batch_padded_indices]
            start = end
            yield mini_batch

            
    def calc_advantages(self, last_value:torch.tensor, gamma:float, lamda:float) -> None:
        with torch.no_grad():
            mask = torch.tensor(self.dones, device=self.device).logical_not()
            rewards = torch.tensor(self.rewards, device=self.device)
    
            self.values = self.values.to(self.device)
            self.advantages = self.advantages.to(self.device)
            last_value = last_value.to(self.device) 
    
            last_advantage = 0
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]


