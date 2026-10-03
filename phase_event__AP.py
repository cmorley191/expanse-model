from phase_rule import *
from game import *

import torch
import torch.nn


CARD_DRUMMER = 0
class PhaseEventMeta_0_Drummer(PhaseEventMeta):

  def card(self):
    return CARD_DRUMMER

CARD_COTYAR = 2
class PhaseEventMeta_2_Cotyar(PhaseEventMeta):

  def card(self):
    return CARD_COTYAR


event_card = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.bool, device=gpu_device)
event_card_ap = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.int8, device=gpu_device)

for (card, ap) in [
  (CARD_DRUMMER, 1),
  (CARD_COTYAR, 2),
]:
  event_card[card] = True
  event_card_ap[card] = ap


class PhaseEvent__AP_Start(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (event_card[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]])
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION].logical_not())
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE].logical_not())
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_ACTION] = True
    new_state.obs_bool[:, OBS_BOOL_SCORE_SECTOR_REVEALED_TO_ALL] |= (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION].logical_xor(state.obs_bool[:, OBS_BOOL_PLAYER_TURN])
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD_COTYAR)
    )
    new_state.obs_int[:, OBS_INT_PHASE_AP] = event_card_ap[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]]

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

    return new_state
  
  def action_str(self):
    return ""
  

class PhaseEvent__AP_Done(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (event_card[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]])
      & (state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE])
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    return new_state
  
  def action_str(self):
    return ""