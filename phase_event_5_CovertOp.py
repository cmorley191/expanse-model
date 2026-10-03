from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 5
class PhaseEventMeta_5_CovertOp(PhaseEventMeta):

  def card(self):
    return CARD
  

class PhaseEvent_5_CovertOp(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_exchange(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, b0, b1)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, BASE_COUNT, BASE_COUNT)
    states_influence = states.obs_int_influence()
    #assert PLAYER_COUNT == 2
    states_influence[:, base_indices, :, base_indices, :] += torch.tensor([-1, 1], dtype=torch.int8, device=gpu_device).view(1, 1, 1, PLAYER_COUNT)
    states_influence[:, :, base_indices, base_indices, :] += torch.tensor([1, -1], dtype=torch.int8, device=gpu_device).view(1, 1, 1, PLAYER_COUNT)

    state_influence = state.obs_int_influence()
    mask = (
      (state_influence[:, :, 0] != 0).view(state.batch[0], BASE_COUNT, 1)
      & (state_influence[:, :, 1] != 0).view(state.batch[0], 1, BASE_COUNT)
    )

    states: ExpanseState = states.view(state.batch[0], BASE_COUNT * BASE_COUNT)
    mask = mask.view(state.batch[0], BASE_COUNT * BASE_COUNT)

    return (states, mask)
  
  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = state.obs_int_influence().bool().any(dim=1).all(dim=1).logical_not().view(state.batch[0], 1)

    return (states, mask)

  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_exchange,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[
        f"Swap 1 {player_name[0]} {orbital_name[o0]} influence with 1 {player_name[1]} {orbital_name[o1]} influence"
        for o0 in range(ORBITAL_COUNT)
        for o1 in range(ORBITAL_COUNT)
      ],
      "Cannot perform swap"
    ]

