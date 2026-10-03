from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 14
class PhaseEventMeta_14_Sadavir(PhaseEventMeta):

  def card(self):
    return CARD


class PhaseEvent_14_Sadavir(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    new_state.obs_int_influence()[:, :, :] -= (state.obs_int_influence()[:, :, :] > 2).to(torch.int8)

    return new_state
  
  def action_str(self):
    return ""

