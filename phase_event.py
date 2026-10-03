from phase_rule import *
from game import *

import torch


event_metas = get_phase_event_metas()
event_implemented = torch.tensor([True if m is not None else False for m in event_metas], dtype=torch.bool, device=gpu_device)

class PhaseEvent_NotImplemented(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & event_implemented[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]].logical_not()
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    return new_state
  
  def action_str(self):
    return "event not implemented"

