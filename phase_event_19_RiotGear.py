from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 19
class PhaseEventMeta_19_RiotGear(PhaseEventMeta):

  def card(self):
    return CARD


event_base_use_indicies = torch.arange(BASE_COUNT+1, dtype=torch.long, device=gpu_device)
event_base_use_ordering_matrix = (
  (
    event_base_use_indicies.view(1, BASE_COUNT+1, 1)
    < event_base_use_indicies.view(1, 1, BASE_COUNT+1)
  )
  | (
    (event_base_use_indicies == BASE_COUNT).view(1, BASE_COUNT+1, 1)
    & (event_base_use_indicies == BASE_COUNT).view(1, 1, BASE_COUNT+1)
  ) 
)

class PhaseEvent_19_RiotGear(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, b0, b1)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, BASE_COUNT+1, BASE_COUNT+1)
    remove_player = state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not()
    states_influence = states.obs_int_influence()
    states_influence[:, base_indices, :, base_indices, :] -= remove_player.bool().to(torch.int8).view(1, state.batch[0], 1, PLAYER_COUNT)
    states_influence[:, :, base_indices, base_indices, :] -= remove_player.bool().to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)

    state_influence = state.obs_int_influence()
    mask = (
      # belt base (yes this is wasteful)
      torch.concat([
        (base_sector == 1).view(1, BASE_COUNT, 1),
        torch.ones((1, 1, 1), dtype=torch.bool, device=gpu_device),
      ], dim=1)
      & torch.concat([
        (base_sector == 1).view(1, 1, BASE_COUNT),
        torch.ones((1, 1, 1,), dtype=torch.bool, device=gpu_device),
      ], dim=2)
      & event_base_use_ordering_matrix
      & (
        # opponent
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT)
        & torch.concat([
          # with influence
          state_influence.bool().view(state.batch[0], BASE_COUNT, 1, PLAYER_COUNT),
          # or skip both
          torch.ones((state.batch[0], 1, 1, PLAYER_COUNT), dtype=torch.bool, device=gpu_device),
        ], dim=1)
        & torch.concat([
          # with influence
          state_influence.bool().view(state.batch[0], 1, BASE_COUNT, PLAYER_COUNT),
          # or skip second
          torch.ones((state.batch[0], 1, 1, PLAYER_COUNT), dtype=torch.bool, device=gpu_device),
        ], dim=2)
        # exists
      ).any(dim=3)
    )

    states: ExpanseState = states.view(state.batch[0], (BASE_COUNT+1) * (BASE_COUNT+1))
    mask = mask.view(state.batch[0], (BASE_COUNT+1) * (BASE_COUNT+1))

    return (states, mask)
  
  def action_str(self):
    return [
      f"{removal0}{removal1}"
      for removal0 in [*[f"remove {b}, " for b in base_name], ""]
      for removal1 in [*[f"remove {b}" for b in base_name], "don't remove"]
    ]

