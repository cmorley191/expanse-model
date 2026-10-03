from phase_rule import *
from game import *

import torch
import torch.nn


CARD_MAO_KWIK = 3

class PhaseAPTurn_MaoKwik(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT].logical_not()
      & state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_MAO_KWIK]
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, use mao-kwik)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_CHOOSE_MAO_KWIK] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_ACTION] = True

    states: ExpanseState = states.repeat(1, 2)
    states.obs_int[:, 1, OBS_INT_PHASE_AP] = 4
    states.obs_pile_present[:, 1, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MAO_KWIK] = False
    states.obs_pile_cached_embed[:, 1, OBS_PILE_CACHED_EMBED_KEPT:OBS_PILE_CACHED_EMBED_KEPT+PLAYER_COUNT, :] -= (
      card_embeds.weight[CARD_MAO_KWIK, :].view(1, 1, state.CARD_EMBED_LENGTH)
      * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.float32).view(state.batch[0], PLAYER_COUNT, 1)
    )

    mask = torch.concat([
      torch.ones((state.batch[0], 1), dtype=torch.bool, device=gpu_device),
      (
        # action player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
        # has mao-kwik kept
        & state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MAO_KWIK]
        # exists
      ).any(dim=1).view(state.batch[0], 1),
    ], dim=1)

    return (states, mask)
  
  def action_str(self):
    return [
      "use focus card ap",
      "discard kept Mao-Kwik to take 4 ap instead",
    ]


class PhaseAP(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_ACTION])

  def use_ap(self, state: ExpanseState):
    state.obs_int[..., OBS_INT_PHASE_AP] -= 1
    state.obs_bool[..., OBS_BOOL_PHASE_ACTION] = state.obs_int[..., OBS_INT_PHASE_AP].bool()
    state.obs_bool[..., OBS_BOOL_PHASE_ACTION_DONE] = state.obs_bool[..., OBS_BOOL_PHASE_ACTION].logical_not()


  def enumerate_fleet(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, count, src, dest, active player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1, 1)
    self.use_ap(states)

    states: ExpanseState = states.repeat(1, FLEET_COUNT, ORBITAL_COUNT, ORBITAL_COUNT, PLAYER_COUNT)
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
    
    state_fleets = state.obs_int_fleets()
    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, 1, PLAYER_COUNT))
      # has enough fleets
      & (state_fleets.view(state.batch[0], 1, ORBITAL_COUNT, 1, PLAYER_COUNT) > fleet_indices.view(1, FLEET_COUNT, 1, 1, 1))
      # adjacent
      & (orbital_adjacent.view(1, 1, ORBITAL_COUNT, ORBITAL_COUNT, 1))
    )

    states: ExpanseState = states.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)

    return (states, mask)
  

  def enumerate_influence(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, base, player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    self.use_ap(states)

    states: ExpanseState = states.repeat(1, BASE_COUNT, PLAYER_COUNT)
    states.obs_int_influence()[
      :,
      base_indices.view(BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT),
      base_indices.view(BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT)
    ] += 1

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT))
      # has presence
      & (state.obs_int_fleets()[:, base_orbital.view(BASE_COUNT, 1), player_indices.view(1, PLAYER_COUNT)] != 0)
    )

    states: ExpanseState = states.view(state.batch[0], BASE_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], BASE_COUNT * PLAYER_COUNT)

    return (states, mask)
  

  def enumerate_build(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, player)
    states: ExpanseState = state.clone().view(state.batch[0], 1).repeat(1, PLAYER_COUNT)
    self.use_ap(states)
    states.obs_int_fleets()[:, player_indices, player_home_orbital, player_indices] += 1

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT])
      # fleet in reserve
      & (state.obs_int_fleets().sum(dim=1) < FLEET_COUNT)
    )

    # enumerations already flat

    return (states, mask)


  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_int[:, :, OBS_INT_PHASE_AP] = 1
    self.use_ap(states)

    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    # enumerations already flat (1)

    return (states, mask)


  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_fleet, 
      self.enumerate_influence,
      self.enumerate_build,
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
      *[
        f"Place on {b}"
        for b in base_name
        for p in range(PLAYER_COUNT)
      ],
      *[
        f"Build a fleet"
        for p in range(PLAYER_COUNT)
      ],
      "Pass remaining points"
    ]


CARD_MILLER = 1

class PhaseAPTurn_Miller(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_AP_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_EVENT].logical_not()
      & state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE]
    )
  
  def enumerate_actions(self, state, card_embeds):
    # (batch, use miller)
    states: ExpanseState = state.clone().view(state.batch[0], 1).repeat(1, 2)
    states.obs_bool[:, :, OBS_BOOL_PHASE_ACTION_DONE] = False
    
    states.obs_bool[:, 0, OBS_BOOL_PHASE_CHOOSE_EVENT] = True
    states.obs_bool[:, 0, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not()
    states.obs_bool[:, 0, OBS_BOOL_PLAYER_PERSPECTIVE:OBS_BOOL_PLAYER_PERSPECTIVE+PLAYER_COUNT] = state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not()

    states.obs_bool[:, 1, OBS_BOOL_PHASE_AP_TURN] = False
    states.obs_bool[:, 1, OBS_BOOL_PHASE_EVENT_TURN] = True
    states.obs_bool[:, 1, OBS_BOOL_PHASE_EVENT] = True
    states.obs_bool[:, 1, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
    states.obs_pile_present[:, 1, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MILLER] = False
    states.obs_pile_cached_embed[:, 1, OBS_PILE_CACHED_EMBED_KEPT:OBS_PILE_CACHED_EMBED_KEPT+PLAYER_COUNT, :] -= (
      card_embeds.weight[CARD_MILLER, :].view(1, 1, state.CARD_EMBED_LENGTH)
      * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.float32).view(state.batch[0], PLAYER_COUNT, 1)
    )

    mask = torch.concat([
      torch.ones((state.batch[0], 1), dtype=torch.bool, device=gpu_device),
      (
        # action player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
        # has miller kept
        & state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MILLER]
        # focus event eligible
        & card_factions[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS], :]
        # exists
      ).any(dim=1).view(state.batch[0], 1),
    ], dim=1)

    return (states, mask)

  def action_str(self):
    return [
      "pass initiative",
      "discard kept Miller to use focused event as well",
    ]

