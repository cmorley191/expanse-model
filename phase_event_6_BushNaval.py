from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 6
class PhaseEventMeta_6_BushNaval(PhaseEventMeta):

  def card(self):
    return CARD


class PhaseEvent_6_BushNaval(PhaseRule):

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
    player_missing_fleets = (FLEET_COUNT - state.obs_int_fleets().sum(dim=1))
    new_state.obs_int_fleets()[:, player_home_orbital, player_indices] += player_missing_fleets
    new_state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT] += (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8)
      * (
        (
          player_missing_fleets
          * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().to(torch.int8)
        ).bool().any(dim=1).to(torch.int8).view(state.batch[0], 1)
      )
    )

    return new_state
  
  def action_str(self):
    return ""

