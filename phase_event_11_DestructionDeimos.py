from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 11
class PhaseEventMeta_11_DestructionDeimos(PhaseEventMeta):

  def card(self):
    return CARD


earth_bases = base_indices[base_orbital == 0]
earth_base_indices = torch.arange(earth_bases.shape[0], dtype=torch.long, device=gpu_device)

class PhaseEvent_11_DestructionDeimos(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, earth base)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    state_influence = state.obs_int_influence()
    mars_bases = base_indices[(base_orbital == 1)]
    #assert PLAYER_COUNT == 2
    states.obs_int_influence()[:, :, mars_bases, :] -= (
      # has influence
      state_influence[:, mars_bases, :].bool()
      & (
        # opponent
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        | (
          # action player
          state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT)
          # opponent does not have influence
          & state_influence[:, mars_bases, :].bool().flip(dims=[2])
        )
      )
    ).to(torch.int8).view(state.batch[0], 1, mars_bases.shape[0], PLAYER_COUNT)

    states: ExpanseState = states.repeat(1, earth_bases.shape[0])
    states.obs_int_influence()[:, earth_base_indices, earth_bases, :] += \
      state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].to(torch.int8).view(state.batch[0], 1, PLAYER_COUNT)

    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    return (states, mask)
  
  def action_str(self):
    return [
      f"place {base_name[b]}"
      for b in earth_bases
    ]

