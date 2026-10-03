from phase_rule import *
from game import *

import torch


class PhaseNonscoreTurnDone(PhaseRule):
  
  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN].logical_not()
      & state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE]
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_TURN_TYPE_START:OBS_BOOL_PHASE_TURN_TYPE_END] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = False
    new_state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE:OBS_BOOL_PLAYER_PERSPECTIVE+PLAYER_COUNT] = False

    no_draw = (state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1] != CARD_EMPTY_TRACK)
    new_state.obs_bool[no_draw, OBS_BOOL_PHASE_DONE] = False
    new_state.obs_bool[no_draw, OBS_BOOL_PHASE_START] = True
    #assert PLAYER_COUNT == 2
    new_player = state.obs_bool[no_draw, OBS_BOOL_PLAYER_TURN:OBS_BOOL_PLAYER_TURN+PLAYER_COUNT].logical_not()
    new_state.obs_bool[no_draw, OBS_BOOL_PLAYER_TURN:OBS_BOOL_PLAYER_TURN+PLAYER_COUNT] = new_player
    new_state.obs_bool[no_draw, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = new_player
    new_state.obs_bool[no_draw, OBS_BOOL_PLAYER_PERSPECTIVE:OBS_BOOL_PLAYER_PERSPECTIVE+PLAYER_COUNT] = new_player

    return new_state

  def action_str(self):
    return ""
