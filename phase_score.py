from phase_rule import *
from game import *

import torch
import torch.nn


class PhaseScore_Sector(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_SECTOR])
  
  def enumerate_actions(self, state, card_embeds):
    # (batch, sector)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    #assert PLAYER_COUNT == 2
    states.obs_bool[:, :, OBS_BOOL_PHASE_CHOOSE_SECTOR] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_CHOOSE_EVENT] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = \
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
    # perspective switches in next phase rule

    states: ExpanseState = states.repeat(1, SECTOR_COUNT)
    states.hid_bool[:, sector_indices, HID_BOOL_SCORE_SECTOR+sector_indices] = True

    states.obs_bool[:, sector_indices, OBS_BOOL_SCORE_SECTOR+sector_indices] = True
    states.obs_int[:, sector_indices, OBS_INT_BONUS_SECTORS+sector_indices] -= 1

    mask = (state.obs_int[:, OBS_INT_BONUS_SECTORS+sector_indices] != 0)

    return (states, mask)

  def action_str(self):
    return [f"Choose sector {s}" for s in sector_name]
  

class PhaseScore_SwitchPerspective(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT]
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE])
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    reveal_sectors = (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_TURN]).logical_not()
      | state.obs_bool[:, OBS_BOOL_SCORE_SECTOR_REVEALED_TO_ALL]
    )
    new_state.obs_bool[:, OBS_BOOL_SCORE_SECTOR:OBS_BOOL_SCORE_SECTOR+SECTOR_COUNT] = (
      state.hid_bool[:, HID_BOOL_SCORE_SECTOR:HID_BOOL_SCORE_SECTOR+SECTOR_COUNT]
      & reveal_sectors.view(state.batch[0], 1)
    )
    new_state.obs_int[:, OBS_INT_BONUS_SECTORS:OBS_INT_BONUS_SECTORS+SECTOR_COUNT] = (
      new_state.hid_int[:, HID_INT_BONUS_SECTORS:HID_INT_BONUS_SECTORS+SECTOR_COUNT]
      - new_state.obs_bool[:, OBS_BOOL_SCORE_SECTOR:OBS_BOOL_SCORE_SECTOR+SECTOR_COUNT].to(torch.int8)
    )
    new_state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE:OBS_BOOL_PLAYER_PERSPECTIVE+PLAYER_COUNT] = \
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]

    return new_state

  def action_str(self):
    return ""


class PhaseScore_ChooseEvent(PhaseRule):
  
  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT]
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE]).logical_not()
    )
  
  def enumerate_use_event(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, active player, kept card)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1).repeat(1, PLAYER_COUNT, CARD_COUNT)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_CHOOSE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = True
    states.obs_bool[:, player_indices, :, OBS_BOOL_PLAYER_EVENT+player_indices] = True
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = card_indices.view(1, 1, CARD_COUNT)
    states.obs_pile_cached_embed[:, player_indices, :, OBS_PILE_CACHED_EMBED_KEPT+player_indices, :] -= card_embeds.weight[:CARD_COUNT, :].view(1, 1, CARD_COUNT, state.CARD_EMBED_LENGTH)
    states.obs_pile_present[
      :,
      player_indices.view(PLAYER_COUNT, 1),
      card_indices.view(1, CARD_COUNT),
      OBS_PILE_PRESENT_KEPT+player_indices.view(PLAYER_COUNT, 1),
      card_indices.view(1, CARD_COUNT)
    ] = False

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT, 1))
      # has card
      & (state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, :])
    )

    states: ExpanseState = states.view(state.batch[0], PLAYER_COUNT * CARD_COUNT)
    mask = mask.view(state.batch[0], PLAYER_COUNT * CARD_COUNT)

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    #assert PLAYER_COUNT == 2
    states.obs_bool[:, :, OBS_BOOL_PHASE_CHOOSE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True

    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    # enumerations already flat (1)

    return (states, mask)

  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_use_event,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[
        a
        for active_player in range(PLAYER_COUNT)
        for a in [f"play kept {c}" for c in card_name[:CARD_COUNT]]
      ],
      "skip score event opportunity"
    ]


class PhaseScore_OpponentEventDone(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    #assert PLAYER_COUNT == 2
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE]
      & (state.obs_bool[:, OBS_BOOL_PLAYER_TURN].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_ACTION]))
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    #assert PLAYER_COUNT == 2
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not()
    # perspective switches in phase rule above

    return new_state
  
  def action_str(self):
    return ""


class PhaseScore_TurnPlayerEventDone(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    #assert PLAYER_COUNT == 2
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE]
      & (state.obs_bool[:, OBS_BOOL_PLAYER_TURN].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_ACTION]).logical_not())
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = False
    new_state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE:OBS_BOOL_PLAYER_PERSPECTIVE+PLAYER_COUNT] = False

    return new_state
  
  def action_str(self):
    return ""


class PhaseScore_Score(PhaseRule):
  
  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_DONE]
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN] = False
    new_state.obs_bool[:, OBS_BOOL_SCORE_SECTOR:OBS_BOOL_SCORE_SECTOR+SECTOR_COUNT] = False
    new_state.hid_bool[:, HID_BOOL_SCORE_SECTOR:HID_BOOL_SCORE_SECTOR+SECTOR_COUNT] = False
    new_state.hid_int[:, HID_INT_BONUS_SECTORS:HID_INT_BONUS_SECTORS+SECTOR_COUNT] -= \
      state.hid_bool[:, HID_BOOL_SCORE_SECTOR:HID_BOOL_SCORE_SECTOR+SECTOR_COUNT].to(torch.int8)
    new_state.obs_int[:, OBS_INT_BONUS_SECTORS:OBS_INT_BONUS_SECTORS+SECTOR_COUNT] = new_state.hid_int[:, HID_INT_BONUS_SECTORS:HID_INT_BONUS_SECTORS+SECTOR_COUNT]
    obs_int_fleets = state.obs_int_fleets()
    obs_int_influence = state.obs_int_influence()
    #assert PLAYER_COUNT == 2
    orbital_control = (
      (obs_int_fleets != 0)
      & (obs_int_fleets > obs_int_fleets.flip(dims=[2]))
    )
    base_present = obs_int_influence != 0
    base_orbital_control_power_bonus = base_present & orbital_control[:, base_orbital, :]
    base_power = obs_int_influence + base_orbital_control_power_bonus
    base_control = (
      base_present
      & (base_power > base_power.flip(dims=[2]))
    )

    new_state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT] += (
      (
        base_control.to(torch.int8)
        + (
          state.obs_bool[:, OBS_BOOL_SCORE_SECTOR+base_sector].to(torch.int8).view(state.batch[0], BASE_COUNT, 1)
          * base_present.to(torch.int8)
          * bonus_sector_points[
            (
              (
                (STARTING_BONUS_SECTORS * SECTOR_COUNT)
                - state.obs_int[:, OBS_INT_BONUS_SECTORS:OBS_INT_BONUS_SECTORS+SECTOR_COUNT].sum(dim=1)
              )
              .to(torch.long)
              .view(state.batch[0], 1, 1)
            ),
            base_control.logical_not().to(torch.long)
          ]
        )
      ).sum(dim=1)
    )

    new_state.obs_int_fleets()[:, player_home_orbital[player_indices], player_indices] += \
      (obs_int_fleets.sum(dim=1) < FLEET_COUNT).to(torch.int8)

    return new_state

  def action_str(self):
    return "Score!"
