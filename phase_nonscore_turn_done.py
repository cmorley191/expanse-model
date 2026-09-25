from phase_rule import *
from expanse_game import *

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
    new_state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = False

    return new_state
