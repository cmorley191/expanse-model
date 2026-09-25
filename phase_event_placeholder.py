from phase_rule import *
from game import *

import torch


class PhaseEvent_Placeholder(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_EVENT])

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    return new_state
  

