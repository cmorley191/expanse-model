from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 25
class PhaseEventMeta_25_CaptainYvgeny(PhaseEventMeta):

  def card(self):
    return CARD

class PhaseEvent_25_CaptainYvgeny(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_remove(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, orbital)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, ORBITAL_COUNT)
    states.obs_int_fleets()[:, orbital_indices, orbital_indices, :] -= (
      state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().to(torch.int8).view(state.batch[0], 1, PLAYER_COUNT)
    )

    mask = (state.obs_int_fleets().bool().all(dim=2))

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = (
      state.obs_int_fleets().bool().all(dim=2).any(dim=1).logical_not()
    ).view(state.batch[0], 1)

    return (states, mask)
  
  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_remove,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[f"remove {o}" for o in orbital_name],
      "Cannot perform"
    ]
