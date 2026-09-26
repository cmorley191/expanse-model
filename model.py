from game import *
from phase_rule import *

import torch
import torch.nn
import torch.nn.functional as F

assert torch.cuda.is_available()
gpu_device = torch.device('cuda')
cpu_device = torch.device('cpu')


class MLP(torch.nn.Module):
  def __init__(self, input_length: int, hidden_lengths: list[int], output_length: int):
    super(MLP, self).__init__()
    self.input_length = input_length
    self.hidden_lengths = hidden_lengths
    self.output_length = output_length
    self.layer_lengths = [input_length] + hidden_lengths + [output_length]

    layers = []
    for i in range(len(self.layer_lengths) - 1):
      layers.append(torch.nn.Linear(self.layer_lengths[i], self.layer_lengths[i+1], dtype=torch.float32))
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

  def __init__(self, hidden_lengths: list[int], card_embed_length: int, *, log: bool = False):
    super(ExpanseModel_Centauri, self).__init__()
    self.hidden_lengths = hidden_lengths
    self.card_embed_length = card_embed_length

    self.model = MLP(
      input_length=(
        OBS_BOOL_LENGTH
        + OBS_INT_LENGTH
        + (OBS_SLOT_INDEX_COUNT * self.card_embed_length)
        + (OBS_PILE_CACHED_EMBED_COUNT * self.card_embed_length)
      ),
      hidden_lengths=hidden_lengths,
      output_length=1
    )

    self.card_embeds = torch.nn.Embedding(num_embeddings=CARD_COUNT+EXTRA_CARD_INDEX_COUNT, embedding_dim=card_embed_length, dtype=torch.float32)

    if log:
      print(f'{ExpanseModel_Centauri}')
      print(f'Card embedding:')
      print(f'  - card count: {self.card_embeds.num_embeddings}')
      print(f'  - embed length: {self.card_embeds.embedding_dim}')
      print(f'Model:')
      print(f'  - input: {self.model.input_length}')
      print(f'     - obs_bool: {OBS_BOOL_LENGTH}')
      print(f'     - obs_int: {OBS_INT_LENGTH}')
      print(f'     - obs_slot_embed: {OBS_SLOT_INDEX_COUNT} slots * {self.card_embed_length}-embed = {OBS_SLOT_INDEX_COUNT * self.card_embed_length}')
      print(f'     - obs_pile_embed: {OBS_PILE_CACHED_EMBED_COUNT} piles * {self.card_embed_length}-embed = {OBS_PILE_CACHED_EMBED_COUNT * self.card_embed_length}')
      print(f'  - hidden: {self.model.hidden_lengths}')
      print(f'  - output: {self.model.output_length}')


  def forward_train(
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
    obs_pile_embed = obs_pile_present @ self.card_embeds.weight[:CARD_COUNT, :]

    return self.forward_eval(obs_bool, obs_int, obs_slot_index, obs_pile_embed)


  def forward_eval(
    self,
    obs_bool: torch.Tensor,
    obs_int: torch.Tensor,
    obs_slot_index: torch.Tensor,
    obs_pile_cached_embed: torch.Tensor
  ) -> torch.Tensor:
    """
    obs_bool:                (batch, obs_bool)
    obs_int:                 (batch, obs_int)
    obs_slot_index:          (batch, slots)
    obs_pile_cached_embed:   (batch, piles, embed)
    returns:                 (batch, 1)
    """
    BATCH = obs_bool.shape[0]

    obs_slot_embed = self.card_embeds(obs_slot_index.to(torch.long))

    obs = torch.concat([
      obs_bool.to(obs_pile_cached_embed.dtype),
      obs_int.to(obs_pile_cached_embed.dtype),
      obs_slot_embed.view(BATCH, OBS_SLOT_INDEX_COUNT * self.card_embed_length),
      obs_pile_cached_embed.view(BATCH, OBS_PILE_PRESENT_COUNT * self.card_embed_length),
    ], dim=1)

    return self.model(obs)

