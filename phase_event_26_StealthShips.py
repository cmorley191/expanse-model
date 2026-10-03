from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 26
class PhaseEventMeta_26_StealthShips(PhaseEventMeta):

  def card(self):
    return CARD

EVENT_COUNTS_COUNT = 3
event_counts = torch.arange(EVENT_COUNTS_COUNT, dtype=torch.int8, device=gpu_device)
class PhaseEvent_26_StealthShips(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_remove(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, count, src orbital, dest orbital)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1)
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, EVENT_COUNTS_COUNT, ORBITAL_COUNT, ORBITAL_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[:, :, orbital_indices, :, orbital_indices, :] -= (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8).view(1, state.batch[0], 1, 1, PLAYER_COUNT)
    )
    states_fleets[:, :, :, orbital_indices, orbital_indices, :] -= (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().to(torch.int8).view(state.batch[0], 1, 1, 1, PLAYER_COUNT)
      * event_counts.view(1, EVENT_COUNTS_COUNT, 1, 1, 1)
    )

    state_fleets = state.obs_int_fleets()
    mask = (
      (
        orbital_adjacent.view(1, 1, ORBITAL_COUNT, ORBITAL_COUNT)
        | torch.eye(ORBITAL_COUNT, dtype=torch.bool, device=gpu_device).view(1, 1, ORBITAL_COUNT, ORBITAL_COUNT)
      )
      & (
        # opponent
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, 1, PLAYER_COUNT)
        # with dest fleets
        & (state_fleets.view(state.batch[0], 1, 1, ORBITAL_COUNT, PLAYER_COUNT) >= event_counts.view(1, EVENT_COUNTS_COUNT, 1, 1, 1))
        # exists
      ).any(dim=4)
      & (
        # turn player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, 1, PLAYER_COUNT)
        # with source fleet
        & state_fleets.bool().view(state.batch[0], 1, ORBITAL_COUNT, 1, PLAYER_COUNT)
        # exists
      ).any(dim=4)
    )

    states: ExpanseState = states.view(state.batch[0], EVENT_COUNTS_COUNT * ORBITAL_COUNT * ORBITAL_COUNT)
    mask = mask.view(state.batch[0], EVENT_COUNTS_COUNT * ORBITAL_COUNT * ORBITAL_COUNT)

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
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
        # with no source fleets
        & state.obs_int_fleets().bool().any(dim=1).logical_not()
        # exists
      ).any(dim=1)
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
      *[
        f"remove self from {o_src} and remove {c} opponent from {o_dest}"
        for c in range(EVENT_COUNTS_COUNT)
        for o_src in orbital_name
        for o_dest in orbital_name
      ],
      "Cannot perform"
    ]
