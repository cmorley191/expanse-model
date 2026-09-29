from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 22
class PhaseEventMeta_22_TheresaYao(PhaseEventMeta):

  def card(self):
    return CARD

event_counts = torch.arange(FLEET_COUNT+1, dtype=torch.int8, device=gpu_device)
class PhaseEvent_22_TheresaYao(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, count, orbital)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
  
    states: ExpanseState = states.repeat(1, FLEET_COUNT+1, ORBITAL_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[:, :, orbital_indices, orbital_indices, :] -= (
      event_counts.view(1, FLEET_COUNT+1, 1, 1)
      + (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT))
    )
    states_fleets.clamp_min_(0)

    mask = (
      (
        # turn player
        (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, PLAYER_COUNT))
        # with enough fleets
        & (state.obs_int_fleets().view(state.batch[0], 1, ORBITAL_COUNT, PLAYER_COUNT) >= event_counts.view(1, FLEET_COUNT+1, 1, 1))
        # exists
      ).any(dim=3)
    )

    states: ExpanseState = states.view(state.batch[0], (FLEET_COUNT+1) * ORBITAL_COUNT)
    mask = mask.view(state.batch[0], (FLEET_COUNT+1) * ORBITAL_COUNT)

    return (states, mask)

  def action_str(self):
    return [
      f"remove {o} - {c} self and {c+1} opponent"
      for c in range(FLEET_COUNT+1)
      for o in orbital_name
    ]

