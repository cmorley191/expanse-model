from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 21
class PhaseEventMeta_21_Nauvoo(PhaseEventMeta):

  def card(self):
    return CARD


class PhaseEvent_21_Nauvoo(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_influence(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, base)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    state_influence = state.obs_int_influence()

    states: ExpanseState = states.repeat(1, BASE_COUNT)
    states.obs_int_influence()[:, base_indices, base_indices, :] -= (
      (
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        & state_influence.bool()
      )
      | (
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT)
        & (state_influence.flip(dims=[2]) == 0)
      )
    ).to(torch.int8)

    mask = (state_influence.bool().any(dim=2))

    return (states, mask)
  
  def enumerate_fleet(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, orbital)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    state_fleets = state.obs_int_fleets()

    states: ExpanseState = states.repeat(1, ORBITAL_COUNT)
    states.obs_int_fleets()[:, orbital_indices, orbital_indices, :] -= (
      (
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        & state_fleets.bool()
      )
      | (
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT)
        & (state_fleets.flip(dims=[2]) == 0)
      )
    ).to(torch.int8)

    mask = (state_fleets.bool().any(dim=2))

    return (states, mask)

  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_influence,
      self.enumerate_fleet,
      # technically there should be a pass option but it is so unlikely, if not impossible,
      # for all players to have no influence and no fleet on the board
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[f"remove influence from {b}" for b in base_name],
      *[f"remove fleet from {o}" for o in orbital_name]
    ]

