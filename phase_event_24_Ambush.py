from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 24
class PhaseEventMeta_24_Ambush(PhaseEventMeta):

  def card(self):
    return CARD

class PhaseEvent_24_Ambush(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, p0 count, p1 count, orbital)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1)
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, 2, 2, ORBITAL_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[:, 1, :, orbital_indices, orbital_indices, 0] -= 1
    states_fleets[:, :, 1, orbital_indices, orbital_indices, 1] -= 1

    state_fleets = state.obs_int_fleets()
    mask = torch.ones((state.batch[0], 2, 2, ORBITAL_COUNT), dtype=torch.bool, device=gpu_device)
    mask[:, 1, :, :] &= state_fleets[:, :, 0].bool().view(state.batch[0], 1, ORBITAL_COUNT)
    mask[:, :, 1, :] &= state_fleets[:, :, 1].bool().view(state.batch[0], 1, ORBITAL_COUNT)
    mask[:, 0, 0, :] = False
    mask[:, 0, 0, 0] = True

    states: ExpanseState = states.view(state.batch[0], 2 * 2 * ORBITAL_COUNT)
    mask = mask.view(state.batch[0], 2 * 2 * ORBITAL_COUNT)

    return (states, mask)

  def action_str(self):
    return [
      (
        "don't remove any"
        if (not p0) and (not p1)
        else f"{"don't " if not p0 else ""}remove {player_name[0]} and {"don't " if not p1 else ""}remove {player_name[1]} from {o}"
      )
      for p0 in [False, True]
      for p1 in [False, True]
      for o in orbital_name
    ]
