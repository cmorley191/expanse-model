from phase_rule import *
from game import *

import torch


INIITATIVE_USE_EVENT = 0
INITIATIVE_USE_NON_FOCUS_START = INIITATIVE_USE_EVENT + 1
INITIATIVE_USE_KEEP = INITIATIVE_USE_NON_FOCUS_START
INITIATIVE_USE_PASS = INITIATIVE_USE_KEEP + 1
INITIATIVE_USE_NON_FOCUS_END = INITIATIVE_USE_PASS + 1
INITIATIVE_USE_COUNT = INITIATIVE_USE_NON_FOCUS_END

class PhaseInitiative(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_AP_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT]
    )
  
  def enumerate_actions(self, state, card_embeds):
    # (batch, initiative use, player)
    states = state.clone()
    states.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT] = False

    focus_card = state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]

    states: ExpanseState = states.view(state.batch[0], 1, 1).repeat(1, INITIATIVE_USE_COUNT, PLAYER_COUNT)
    states.obs_bool[:, INIITATIVE_USE_EVENT, :, OBS_BOOL_PHASE_EVENT] = True
    states.obs_bool[:, INITIATIVE_USE_NON_FOCUS_START:INITIATIVE_USE_NON_FOCUS_END, :, OBS_BOOL_PHASE_AP_TURN] = False
    states.obs_bool[:, INITIATIVE_USE_NON_FOCUS_START:INITIATIVE_USE_NON_FOCUS_END, :, OBS_BOOL_PHASE_DONE] = True
    states.obs_bool[:, INITIATIVE_USE_NON_FOCUS_START:INITIATIVE_USE_NON_FOCUS_END, :, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = False
    states.obs_slot_index[:, INITIATIVE_USE_NON_FOCUS_START:INITIATIVE_USE_NON_FOCUS_END, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS
    states.obs_int[:, INITIATIVE_USE_KEEP, player_indices, OBS_INT_CP+player_indices] -= card_keep_cost[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]].view(state.batch[0], 1)
    states.obs_pile_cached_embed[:, INITIATIVE_USE_KEEP, player_indices, OBS_PILE_CACHED_EMBED_KEPT+player_indices, :] += \
      card_embeds(focus_card).view(state.batch[0], 1, state.CARD_EMBED_LENGTH)
    states.obs_pile_present[
      state.batch_indices[0].view(state.batch[0], 1),
      INITIATIVE_USE_KEEP,
      player_indices.view(1, PLAYER_COUNT),
      OBS_PILE_PRESENT_KEPT+player_indices.view(1, PLAYER_COUNT),
      focus_card.view(state.batch[0], 1)
    ] = True

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT))
      & torch.concat([
        # eligible to use
        card_factions[
          focus_card.view(state.batch[0], 1, 1),
          player_indices.view(1, 1, PLAYER_COUNT)
        ],
        # eligible to keep
        (
          (
            state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT) 
            >= card_keep_cost[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]].view(state.batch[0], 1, 1)
          )
          & card_factions[
            focus_card.view(state.batch[0], 1, 1),
            player_indices.view(1, 1, PLAYER_COUNT)
          ]
        ),
        # anyone can pass
        torch.ones((state.batch[0], 1, PLAYER_COUNT), dtype=torch.bool, device=gpu_device),
      ], dim=1)
    )

    states: ExpanseState = states.view(state.batch[0], INITIATIVE_USE_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], INITIATIVE_USE_COUNT * PLAYER_COUNT)

    return (states, mask)

  def action_str(self):
    return [
      u
      for u in [
        "use focused event",
        "keep focused card",
        "skip initiative opportunity",
      ]
      for p in range(PLAYER_COUNT)
    ]
