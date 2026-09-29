from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 10
class PhaseEventMeta_10_VoicesEros(PhaseEventMeta):

  def card(self):
    return CARD


sector_bases = [base_indices[base_sector == s] for s in range(SECTOR_COUNT)]
sector_base_count = [sb.shape[0] for sb in sector_bases]
sector_base_indices = [torch.arange(c, dtype=torch.long, device=gpu_device) for c in sector_base_count]

class PhaseEvent_10_VoicesEros(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, s0, s1, s2)
    states: ExpanseState = state.clone().view(state.batch[0], *([1] * SECTOR_COUNT))
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, *sector_base_count)
    states_influence = states.obs_int_influence()
    states_influence[:, sector_base_indices[0], :, :, sector_bases[0], :] += \
      state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].to(torch.int8).view(1, state.batch[0], 1, 1, PLAYER_COUNT)
    states_influence[:, :, sector_base_indices[1], :, sector_bases[1], :] += \
      state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].to(torch.int8).view(1, state.batch[0], 1, 1, PLAYER_COUNT)
    states_influence[:, :, :, sector_base_indices[2], sector_bases[2], :] += \
      state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].to(torch.int8).view(state.batch[0], 1, 1, 1, PLAYER_COUNT)

    states: ExpanseState = states.view(state.batch[0], sector_base_count[0] * sector_base_count[1] * sector_base_count[2])
    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    return (states, mask)
  
  def action_str(self):
    return [
      f"place {base_name[b0]}, {base_name[b1]}, {base_name[b2]}"
      for b0 in sector_bases[0]
      for b1 in sector_bases[1]
      for b2 in sector_bases[2]
    ]

