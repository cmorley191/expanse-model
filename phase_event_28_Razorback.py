from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 28
class PhaseEventMeta_28_Razorback(PhaseEventMeta):

  def card(self):
    return CARD

class PhaseEvent_28_Razorback(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_move(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, src, dest)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, ORBITAL_COUNT, ORBITAL_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[:, orbital_indices, :, orbital_indices, :] -= (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8).view(1, state.batch[0], 1, PLAYER_COUNT)
    )
    states_fleets[:, :, orbital_indices, orbital_indices, :] += (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)
    )

    mask = (
      (
        # turn player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, PLAYER_COUNT)
        # with source fleet
        & state.obs_int_fleets().bool().view(state.batch[0], ORBITAL_COUNT, 1, PLAYER_COUNT)
        # exists
      ).any(dim=3)
    ).repeat(1, 1, ORBITAL_COUNT)

    states: ExpanseState = states.view(state.batch[0], ORBITAL_COUNT * ORBITAL_COUNT)
    mask = mask.view(state.batch[0], ORBITAL_COUNT * ORBITAL_COUNT)

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = (
      (
        # turn player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT)
        # with no fleets
        & state.obs_int_fleets().bool().any(dim=1).logical_not().view(state.batch[0], PLAYER_COUNT)
        # exists
      ).any(dim=1)
    ).view(state.batch[0], 1)

    return (states, mask)
  
  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_move,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[
        f"move {o_src} to {o_dest}"
        for o_src in orbital_name
        for o_dest in orbital_name
      ],
      "Cannot perform"
    ]
