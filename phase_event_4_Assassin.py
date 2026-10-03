from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 4
class PhaseEventMeta_4_Assassin(PhaseEventMeta):

  def card(self):
    return CARD
  

class PhaseEvent_4_Assassin_Discard(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE]).logical_not()
    )

  def enumerate_discard(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, card)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = \
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
    
    states: ExpanseState = states.repeat(1, CARD_COUNT)
    states.obs_pile_present[:, card_indices, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, card_indices] = False
    states.obs_pile_cached_embed[:, :, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, :] -= (
      card_embeds.weight[:CARD_COUNT, :].view(1, CARD_COUNT, 1, state.CARD_EMBED_LENGTH)
      * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.float32).view(state.batch[0], 1, PLAYER_COUNT, 1)
    )

    mask = (
      (
        # active player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT, 1)
        # has card
        & state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, :]
        # exists
      ).any(dim=1)
    )

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT] = \
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, PLAYER_COUNT)
    
    mask = (
      (
        # active player
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT)
        # no cards
        & state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, :].any(dim=2).logical_not()
        # exists
      ).any(dim=1)
    ).view(state.batch[0], 1)

    return (states, mask)
  
  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_discard,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)

  def action_str(self):
    return [
      *[f"discard {c}" for c in card_name],
      "no kept cards to discard",
    ]


class PhaseEvent_4_Assassin_SwitchPerspectiveToOpponent(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_EVENT])
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE])
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    reveal_sectors = (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & (
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_TURN]).logical_not()
        | state.obs_bool[:, OBS_BOOL_SCORE_SECTOR_REVEALED_TO_ALL]
      )
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


class PhaseEvent_4_Assassin_SwitchPerspectiveBackToEvent(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_EVENT]).logical_not()
      & state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_PERSPECTIVE])
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    reveal_sectors = (
      state.obs_bool[:, OBS_BOOL_PHASE_SCORE_TURN]
      & (
        state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_TURN]).logical_not()
        | state.obs_bool[:, OBS_BOOL_SCORE_SECTOR_REVEALED_TO_ALL]
      )
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


