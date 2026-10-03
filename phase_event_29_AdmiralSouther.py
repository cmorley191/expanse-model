from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 29
class PhaseEventMeta_29_AdmiralSouther(PhaseEventMeta):

  def card(self):
    return CARD

class PhaseEvent_29_AdmiralSouther_Setup(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & (state.obs_int[:, OBS_INT_PHASE_EVENT_ACTIONS] == 0)
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_int[:, OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS:OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+ORBITAL_COUNT] = (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8).view(state.batch[0], 1, PLAYER_COUNT)
      * state.obs_int_fleets()
    ).sum(dim=2)
    new_state.obs_int[:, OBS_INT_PHASE_EVENT_ACTIONS] = \
      new_state.obs_int[:, OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS:OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+ORBITAL_COUNT].sum(dim=1).clamp_min(1)
  
    return new_state

  def action_str(self):
    return ""


class PhaseEvent_29_AdmiralSouther_Move(PhaseRule):
  
  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & state.obs_int[:, OBS_INT_PHASE_EVENT_ACTIONS].bool()
    )
  
  def check_done(self, state: ExpanseState):
    state.obs_bool[..., OBS_BOOL_PHASE_EVENT] = state.obs_int[..., OBS_INT_PHASE_EVENT_ACTIONS].bool()
    state.obs_bool[..., OBS_BOOL_PHASE_EVENT_DONE] = (state.obs_int[..., OBS_INT_PHASE_EVENT_ACTIONS] == 0)
    state.obs_bool[..., OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] &= \
      state.obs_int[..., OBS_INT_PHASE_EVENT_ACTIONS:OBS_INT_PHASE_EVENT_ACTIONS+1].bool()
    state.obs_slot_index[..., OBS_SLOT_INDEX_FOCUS] = torch.where(
      state.obs_int[..., OBS_INT_PHASE_EVENT_ACTIONS].bool(),
      state.obs_slot_index[..., OBS_SLOT_INDEX_FOCUS],
      CARD_EMPTY_FOCUS
    )
    state.obs_int[..., OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS:OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+ORBITAL_COUNT] *= \
      state.obs_int[..., OBS_INT_PHASE_EVENT_ACTIONS:OBS_INT_PHASE_EVENT_ACTIONS+1].bool().to(torch.int8)


  def enumerate_move(self, state: ExpanseState, card_embed: torch.nn.Embedding):
    # (batch, count, src, dest, active player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1, 1)

    states: ExpanseState = states.repeat(1, FLEET_COUNT, ORBITAL_COUNT, 1, 1)
    states.obs_int[:, :, :, :, :, OBS_INT_PHASE_EVENT_ACTIONS] -= fleet_indices.view(1, FLEET_COUNT, 1, 1, 1) + 1
    states.obs_int[
      :,
      fleet_indices.view(FLEET_COUNT, 1),
      orbital_indices.view(1, ORBITAL_COUNT),
      :,
      :,
      OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+orbital_indices.view(1, ORBITAL_COUNT)
    ] -= fleet_indices.view(FLEET_COUNT, 1, 1, 1, 1) + 1
    self.check_done(states)

    states: ExpanseState = states.repeat(1, 1, 1, ORBITAL_COUNT, PLAYER_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[
      :,
      fleet_indices.view(FLEET_COUNT, 1, 1),
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      :,
      player_indices.view(1, 1, PLAYER_COUNT), 
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT)
    ] -= fleet_indices.view(FLEET_COUNT, 1, 1, 1, 1) + 1
    states_fleets[
      :, 
      fleet_indices.view(FLEET_COUNT, 1, 1),
      :,
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT), 
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT)
    ] += fleet_indices.view(FLEET_COUNT, 1, 1, 1, 1) + 1
    
    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, 1, PLAYER_COUNT))
      # has enough eligible fleets
      & (
        (
          state.obs_int[:, OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS:OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+ORBITAL_COUNT]
          .view(state.batch[0], 1, ORBITAL_COUNT, 1, 1)
        )
        > fleet_indices.view(1, FLEET_COUNT, 1, 1, 1)
      )
      # adjacent
      & orbital_adjacent.view(1, 1, ORBITAL_COUNT, ORBITAL_COUNT, 1)
    )

    states: ExpanseState = states.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_int[:, :, OBS_INT_PHASE_EVENT_ACTIONS] = 0
    states.obs_int[:, :, OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS:OBS_INT_PHASE_EVENT_ELIGIBLE_FLEETS+ORBITAL_COUNT] = 0
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    # enumerations already flat (1)

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
        f"Move {c+1} from {s} to {d}"
        for c in range(FLEET_COUNT)
        for s in orbital_name
        for d in orbital_name
        for p in range(PLAYER_COUNT)
      ],
      "Pass remaining moves"
    ]
  
