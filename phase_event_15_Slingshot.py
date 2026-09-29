from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 15
class PhaseEventMeta_15_Slingshot(PhaseEventMeta):

  def card(self):
    return CARD
  

event_bases = base_indices[base_sector == 2]
EVENT_BASE_COUNT = event_bases.shape[0]
event_base_indices = torch.arange(EVENT_BASE_COUNT, dtype=torch.long, device=gpu_device)

class PhaseEvent_15_Slingshot(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_remove(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, remove base, remove player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    
    states: ExpanseState = states.repeat(1, EVENT_BASE_COUNT, PLAYER_COUNT)
    states.obs_int_influence()[
      :,
      event_base_indices.view(EVENT_BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT),
      event_bases.view(EVENT_BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT)
    ] -= 1

    state_influence = state.obs_int_influence()
    mask = (
      # player influence on base
      state_influence[:, event_bases, :].bool()
      & (
        # remove opponent
        state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        | (
          # remove self
          state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT)
          # opponent has none to remove
          & state_influence[:, event_bases, :].bool().flip(dims=[2]).any(dim=1).logical_not().view(state.batch[0], 1, PLAYER_COUNT)
        )
      )
    )

    states: ExpanseState = states.view(state.batch[0], EVENT_BASE_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], EVENT_BASE_COUNT * PLAYER_COUNT)

    return (states, mask)
  
  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = state.obs_int_influence()[:, event_bases, :].view(state.batch[0], EVENT_BASE_COUNT * PLAYER_COUNT, 1).any(dim=1).logical_not()

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
        f"remove {player_name[p]} {base_name[b]}"
        for b in event_bases.tolist()
        for p in range(PLAYER_COUNT)
      ],
      "Cannot remove",
    ]

