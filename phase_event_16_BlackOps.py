from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 16
class PhaseEventMeta_16_BlackOps(PhaseEventMeta):

  def card(self):
    return CARD
  

band_bases = [base_indices[base_band == band] for band in range(BAND_COUNT)]
band_base_count = [bb.shape[0] for bb in band_bases]
band_base_indicies = [torch.arange(c, dtype=torch.long, device=gpu_device) for c in band_base_count]

skip_one_band_mask = torch.zeros((1, *[c + 1 for c in band_base_count]), dtype=torch.bool, device=gpu_device)
skip_one_band_mask[:, band_base_count[0], :, :, :] = True
skip_one_band_mask[:, :, band_base_count[1], :, :] = True
skip_one_band_mask[:, :, :, band_base_count[2], :] = True
skip_one_band_mask[:, :, :, :, band_base_count[3]] = True

class PhaseEvent_16_BlackOps(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, band0, band1, band2, band3)
    states: ExpanseState = state.clone().view(state.batch[0], *([1] * BAND_COUNT))
    states.obs_bool[:, :, :, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_slot_index[:, :, :, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    remove_player = state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not()

    states: ExpanseState = states.repeat(1, *[c + 1 for c in band_base_count])
    states_influence = states.obs_int_influence()
    states_influence[:, band_base_indicies[0], :, :, :, band_bases[0], :] -= remove_player.to(torch.int8).view(1, state.batch[0], 1, 1, 1, PLAYER_COUNT)
    states_influence[:, :, band_base_indicies[1], :, :, band_bases[1], :] -= remove_player.to(torch.int8).view(1, state.batch[0], 1, 1, 1, PLAYER_COUNT)
    states_influence[:, :, :, band_base_indicies[2], :, band_bases[2], :] -= remove_player.to(torch.int8).view(1, state.batch[0], 1, 1, 1, PLAYER_COUNT)
    states_influence[:, :, :, :, band_base_indicies[3], band_bases[3], :] -= remove_player.to(torch.int8).view(state.batch[0], 1, 1, 1, 1, PLAYER_COUNT)

    state_influence = state.obs_int_influence()
    mask = (
      skip_one_band_mask
      & torch.concat([
        (
          remove_player.view(state.batch[0], 1, 1, 1, 1, PLAYER_COUNT)
          # with influence
          & state_influence[:, band_bases[0], :].view(state.batch[0], band_base_count[0], 1, 1, 1, PLAYER_COUNT)
          # exists
        ).any(dim=5),
        # or pass this band
        torch.ones((state.batch[0], 1, 1, 1, 1), dtype=torch.bool, device=gpu_device)
      ], dim=1)
      & torch.concat([
        (
          remove_player.view(state.batch[0], 1, 1, 1, 1, PLAYER_COUNT)
          # with influence
          & state_influence[:, band_bases[1], :].view(state.batch[0], 1, band_base_count[1], 1, 1, PLAYER_COUNT)
          # exists
        ).any(dim=5),
        # or pass this band
        torch.ones((state.batch[0], 1, 1, 1, 1), dtype=torch.bool, device=gpu_device)
      ], dim=2)
      & torch.concat([
        (
          remove_player.view(state.batch[0], 1, 1, 1, 1, PLAYER_COUNT)
          # with influence
          & state_influence[:, band_bases[2], :].view(state.batch[0], 1, 1, band_base_count[2], 1, PLAYER_COUNT)
          # exists
        ).any(dim=5),
        # or pass this band
        torch.ones((state.batch[0], 1, 1, 1, 1), dtype=torch.bool, device=gpu_device)
      ], dim=3)
      & torch.concat([
        (
          remove_player.view(state.batch[0], 1, 1, 1, 1, PLAYER_COUNT)
          # with influence
          & state_influence[:, band_bases[3], :].view(state.batch[0], 1, 1, 1, band_base_count[3], PLAYER_COUNT)
          # exists
        ).any(dim=5),
        # or pass this band
        torch.ones((state.batch[0], 1, 1, 1, 1), dtype=torch.bool, device=gpu_device)
      ], dim=4)
    )

    s = (band_base_count[0] + 1) * (band_base_count[1] + 1) * (band_base_count[2] + 1) * (band_base_count[3] + 1)
    states: ExpanseState = states.view(state.batch[0], s)
    mask = mask.view(state.batch[0], s)

    return (states, mask)
  
  def action_str(self):
    return [
      ", ".join([removal0, removal1, removal2, removal3])
      for removal0 in [*[f"remove {base_name[b]}" for b in band_bases[0]], f"don't remove {band_name[0]}"]
      for removal1 in [*[f"remove {base_name[b]}" for b in band_bases[1]], f"don't remove {band_name[1]}"]
      for removal2 in [*[f"remove {base_name[b]}" for b in band_bases[2]], f"don't remove {band_name[2]}"]
      for removal3 in [*[f"remove {base_name[b]}" for b in band_bases[3]], f"don't remove {band_name[3]}"]
    ]

