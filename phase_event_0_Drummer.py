from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 0
class PhaseEventMeta_0_Drummer(PhaseEventMeta):

  def card(self):
    return CARD


class PhaseEvent_0_Drummer_Start(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION].logical_not())
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE].logical_not())
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_ACTION] = True
    new_state.obs_int[:, OBS_INT_PHASE_AP] = 1

    return new_state
  
  def action_str(self):
    return ""
  

class PhaseEvent_0_Drummer_Done(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == 0)
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE])
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    return new_state
  
  def action_str(self):
    return ""