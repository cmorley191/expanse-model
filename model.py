from game import *
from phase_rule import *

import sys

import torch
import torch.nn
import torch.nn.functional as F

assert torch.cuda.is_available()
gpu_device = torch.device('cuda')
cpu_device = torch.device('cpu')


phase_rules = get_phases()
print()
print(f'Loaded {len(phase_rules)} phase rules')
#for p in phases:
#  print(f'  - {type(p)}')
print()



class MLP(torch.nn.Module):
  def __init__(self, input_length: int, hidden_lengths: list[int], output_length: int):
    super(MLP, self).__init__()
    self.input_length = input_length
    self.hidden_lengths = hidden_lengths
    self.output_length = output_length
    self.layer_lengths = [input_length] + hidden_lengths + [output_length]

    layers = []
    for i in range(len(self.layer_lengths) - 1):
      layers.append(torch.nn.Linear(self.layer_lengths[i], self.layer_lengths[i+1]))
      if i != len(hidden_lengths):
        layers.append(torch.nn.ReLU())

    self.model = torch.nn.Sequential(*layers)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    input:   (batch, input_length)
    returns: (batch, output_length)
    """
    return self.model(inputs)
  


class ExpanseModel_Centauri(torch.nn.Module):

  def __init__(self, hidden_lengths: list[int], card_embed_length: int):
    super(ExpanseModel_Centauri, self).__init__()
    self.hidden_lengths = hidden_lengths
    self.card_embed_length = card_embed_length

    self.model = MLP(
      input_length=(
        OBS_BOOL_LENGTH
        + OBS_INT_LENGTH
        + (OBS_PILE_CACHED_EMBED_COUNT * self.card_embed_length)
      ),
      hidden_lengths=hidden_lengths,
      output_length=1
    )

    self.card_embeds = torch.nn.Embedding(num_embeddings=CARD_COUNT+EXTRA_CARD_INDEX_COUNT, embedding_dim=card_embed_length)


  def forward(
    self,
    obs_bool: torch.Tensor,
    obs_int: torch.Tensor,
    obs_slot_index: torch.Tensor,
    obs_pile_present: torch.Tensor
  ) -> torch.Tensor:
    """
    obs_bool:                (batch, obs_bool)
    obs_int:                 (batch, obs_int)
    obs_slot_index:          (batch, slots)
    obs_pile_present:        (batch, piles, cards)
    returns:                 (batch, 1)
    """
    BATCH = obs_bool.shape[0]

    obs_slot_embeds = self.card_embeds(obs_slot_index)
    obs_pile_cached_embeds = obs_pile_present @ self.card_embeds.weight
    obs = torch.concat([
      obs_bool.to(obs_pile_cached_embeds.dtype),
      obs_int.to(obs_pile_cached_embeds.dtype),
      obs_slot_embeds.view(BATCH, OBS_SLOT_INDEX_COUNT * self.card_embed_length),
      obs_pile_cached_embeds.view(BATCH, OBS_PILE_PRESENT_COUNT * self.card_embed_length),
    ], dim=1)

    return self.model(obs)

