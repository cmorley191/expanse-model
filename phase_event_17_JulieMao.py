from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 17
class PhaseEventMeta_17_JulieMao(PhaseEventMeta):

  def card(self):
    return CARD
  
event_pairs = torch.tensor([
  base_indices[
    (base_resource == resource)
    & (base_indices != base)
  ].tolist()
  for resource in range(RESOURCE_COUNT)
  for base in range(BASE_COUNT) if (base_resource[base].item() == resource)
], dtype=torch.long, device=gpu_device)
EVENT_PAIR_COUNT = event_pairs.shape[0]
event_pair_indices = torch.arange(EVENT_PAIR_COUNT, dtype=torch.long, device=gpu_device)

class PhaseEvent_17_JulieMao(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_pairs(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, pair)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, EVENT_PAIR_COUNT)
    state_influence = state.obs_int_influence()
    states.obs_int_influence()[
      :,
      event_pair_indices.view(EVENT_PAIR_COUNT, 1),
      event_pairs,
      :
    ] -= (
      (
        # opponent
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT)
        # has influence
        & state_influence[:, event_pairs, :].bool()
      )
      | (
        # self
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT)
        # opponent has no influence
        & state_influence[:, event_pairs, :].bool().flip(dims=[3]).logical_not()
      )
    ).to(torch.int8)

    mask = (
      # someone has influence
      state_influence[:, event_pairs, :].any(dim=3).all(dim=2)
    )

    return (states, mask)

  def enumerate_singles(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, base)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, BASE_COUNT)
    state_influence = state.obs_int_influence()
    states.obs_int_influence()[:, base_indices, base_indices, :] -= (
      (
        # opponent
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        # has influence
        & state_influence.bool()
      )
      | (
        # self
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        # opponent has no influence
        & state_influence.bool().flip(dims=[2]).logical_not()
      )
    ).to(torch.int8)

    mask = (
      # someone has influence
      state_influence.any(dim=2)
    )

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = state.obs_int_influence().bool().any(dim=1).all(dim=1).logical_not().view(state.batch[0], 1)

    return (states, mask)
  
  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_pairs,
      self.enumerate_singles,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)

  def action_str(self):
    return [
      *[
        ", ".join([f"remove {b}" for b in pair])
        for pair in event_pair_indices.tolist()
      ],
      *[f"remove {b}" for b in range(BASE_COUNT)],
      "Cannot remove"
    ]

