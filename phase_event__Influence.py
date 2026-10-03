from phase_rule import *
from game import *

import torch
import torch.nn



CARD_TECH = 7
class PhaseEventMeta_7_AntonyDresden(PhaseEventMeta):

  def card(self):
    return CARD_TECH


CARD_EARTH = 8
class PhaseEventMeta_8_FranklinDeGraff(PhaseEventMeta):

  def card(self):
    return CARD_EARTH


CARD_MARS = 9
class PhaseEventMeta_9_Terraforming(PhaseEventMeta):

  def card(self):
    return CARD_MARS


event_card = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.bool, device=gpu_device)
event_card_base = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT, BASE_COUNT), dtype=torch.bool, device=gpu_device)
for (card, base_mask) in [
  (CARD_TECH, (base_resource == RESOURCE_TECH)),
  (CARD_EARTH, (base_orbital == 0)),
  (CARD_MARS, (base_orbital == 1)),
]:
  event_card[card] = True
  event_card_base[card, :] = base_mask

class PhaseEvent__Influence(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (event_card[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]])
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_EVENT_DONE] = True
    new_state.obs_bool[:, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    new_state.obs_int_influence()[:, :, :] += (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT)
      & event_card_base[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]].view(state.batch[0], BASE_COUNT, 1)
    ).to(torch.int8)

    return new_state
  
  def action_str(self):
    return ""

