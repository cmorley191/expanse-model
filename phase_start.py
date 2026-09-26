from phase_rule import *
from game import *

import torch
import torch.nn


track_advancement_matrix = (
  track_indices.view(1, TRACK_CARD_COUNT).repeat(TRACK_CARD_COUNT, 1)
  + (track_indices.view(1, TRACK_CARD_COUNT) >= track_indices.view(TRACK_CARD_COUNT, 1)).long()
)[:, :TRACK_CARD_COUNT-1]

TRACK_USE_FOCUS_START = 0
TRACK_USE_AP = TRACK_USE_FOCUS_START
TRACK_USE_EVENT = TRACK_USE_AP + 1
TRACK_USE_FOCUS_END = TRACK_USE_EVENT + 1
TRACK_USE_KEEP = TRACK_USE_FOCUS_END
TRACK_USE_SCORE = TRACK_USE_KEEP + 1
TRACK_USE_COUNT = TRACK_USE_SCORE + 1
track_use_indices = torch.arange(TRACK_USE_COUNT, dtype=torch.long, device=gpu_device)
track_use_cost = torch.zeros((TRACK_USE_COUNT,), dtype=torch.uint8, device=gpu_device)
track_use_cost[TRACK_USE_KEEP] = 1

class PhaseStart(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_START])
  
  
  def enumerate_kept_event(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, active player, card kept)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_START] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = True

    states: ExpanseState = states.repeat(1, PLAYER_COUNT, CARD_COUNT)
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = card_indices.view(1, CARD_COUNT)
    states.obs_pile_present[:, :, card_indices, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, card_indices] = False
    states.obs_pile_cached_embed[:, player_indices, :, OBS_PILE_CACHED_EMBED_KEPT+player_indices, :] -= card_embeds.weight[:CARD_COUNT].view(1, CARD_COUNT, state.CARD_EMBED_LENGTH)

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT, 1))
      # has card kept
      & (state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT+player_indices, :])
    )

    states: ExpanseState = states.view(state.batch[0], PLAYER_COUNT * CARD_COUNT)
    mask = mask.view(state.batch[0], PLAYER_COUNT * CARD_COUNT)

    return (states, mask)


  def enumerate_track(self, state: ExpanseState, card_embeds: torch.Tensor):
    # (batch, track use, active player, i_track)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1).repeat(1, TRACK_USE_COUNT, 1, 1)
    states.obs_bool[:, :, :, :, OBS_BOOL_PHASE_START] = False
    states.obs_bool[:, TRACK_USE_AP, :, :, OBS_BOOL_PHASE_AP_TURN] = True
    states.obs_bool[:, TRACK_USE_AP, :, :, OBS_BOOL_PHASE_ACTION] = True
    states.obs_bool[:, TRACK_USE_EVENT, :, :, OBS_BOOL_PHASE_EVENT_TURN] = True
    states.obs_bool[:, TRACK_USE_EVENT, :, :, OBS_BOOL_PHASE_EVENT] = True
    states.obs_bool[:, TRACK_USE_KEEP, :, :, OBS_BOOL_PHASE_EVENT_TURN] = True
    states.obs_bool[:, TRACK_USE_KEEP, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, TRACK_USE_SCORE, :, :, OBS_BOOL_PHASE_SCORE_TURN] = True
    states.obs_bool[:, TRACK_USE_SCORE, :, :, OBS_BOOL_PHASE_CHOOSE_SECTOR] = True

    track_card = state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK:OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT]

    states: ExpanseState = states.repeat(1, 1, PLAYER_COUNT, TRACK_CARD_COUNT)
    states.obs_slot_index[:, TRACK_USE_FOCUS_START:TRACK_USE_FOCUS_END, :, :, OBS_SLOT_INDEX_FOCUS] = \
      track_card.view(state.batch[0], 1, 1, TRACK_CARD_COUNT)
    states.obs_pile_present[
      state.batch_indices[0].view(state.batch[0], 1, 1),
      TRACK_USE_KEEP,
      player_indices.view(1, PLAYER_COUNT, 1),
      track_indices.view(1, 1, TRACK_CARD_COUNT),
      OBS_PILE_PRESENT_KEPT+player_indices.view(1, PLAYER_COUNT, 1),
      (track_card.view(state.batch[0], 1, TRACK_CARD_COUNT) % CARD_COUNT)  # when % CARD_COUNT has effect it'll be masked out anyways
    ] = True
    states.obs_pile_cached_embed[:, TRACK_USE_KEEP, player_indices, :, OBS_PILE_CACHED_EMBED_KEPT+player_indices, :] += \
      card_embeds(track_card).view(1, state.batch[0], TRACK_CARD_COUNT, state.CARD_EMBED_LENGTH)
    states.obs_slot_index[
      :,
      :,
      :,
      track_indices.view(TRACK_CARD_COUNT, 1),
      OBS_SLOT_INDEX_TRACK+track_indices[:TRACK_CARD_COUNT-1].view(1, TRACK_CARD_COUNT-1)
    ] = track_card[:, track_advancement_matrix].view(state.batch[0], 1, 1, TRACK_CARD_COUNT, TRACK_CARD_COUNT-1)
    states.obs_slot_index[:, :, :, :, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1] = CARD_EMPTY_TRACK
    states.obs_int[
      :,
      :,
      player_indices.view(PLAYER_COUNT, 1),
      track_indices.view(1, TRACK_CARD_COUNT),
      OBS_INT_CP+player_indices.view(PLAYER_COUNT, 1)
    ] -= track_cost.view(1, 1, 1, TRACK_CARD_COUNT)
    states.obs_int[:, TRACK_USE_KEEP, player_indices, :, OBS_INT_CP+player_indices] -= 1
    states.obs_int[:, TRACK_USE_AP, :, :, OBS_INT_PHASE_AP] = \
      card_ap[track_card].view(state.batch[0], 1, TRACK_CARD_COUNT)

    track_score = (track_card == CARD_SCORE)
    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT, 1))
      # enough cp to spend
      & (state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT, 1) >= (
        track_cost.view(1, 1, 1, TRACK_CARD_COUNT))
        + track_use_cost.view(1, TRACK_USE_COUNT, 1, 1)
      )
      & torch.concat([
        # use any non-score card for ap
        track_score.logical_not().view(state.batch[0], 1, 1, TRACK_CARD_COUNT).repeat(1, 1, PLAYER_COUNT, 1),
        # player must be eligible to use for event
        card_factions[
          track_card.view(state.batch[0], 1, 1, TRACK_CARD_COUNT),
          player_indices.view(1, 1, PLAYER_COUNT, 1)
        ].repeat(1, 2, 1, 1),
        # score with score card
        track_score.view(state.batch[0], 1, 1, TRACK_CARD_COUNT).repeat(1, 1, PLAYER_COUNT, 1),
      ], dim=1)
    )

    states: ExpanseState = states.view(state.batch[0], TRACK_USE_COUNT * PLAYER_COUNT * TRACK_CARD_COUNT)
    mask = mask.view(state.batch[0], TRACK_USE_COUNT * PLAYER_COUNT * TRACK_CARD_COUNT)

    return (states, mask)


  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [self.enumerate_kept_event, self.enumerate_track]]

    return self.concat_state_masks(enumerations)

  def action_str(self):
    return [
      *[
        f"play kept {c}"
        for p in range(PLAYER_COUNT)
        for c in card_name[:CARD_COUNT]
      ],
      *[
        f"use ap on track {t}"
        for p in range(PLAYER_COUNT)
        for t in range(TRACK_CARD_COUNT)
      ],
      *[
        f"use event on track {t}"
        for p in range(PLAYER_COUNT)
        for t in range(TRACK_CARD_COUNT)
      ],
      *[
        f"keep track {t}"
        for active_player in range(PLAYER_COUNT)
        for t in range(TRACK_CARD_COUNT)
      ],
      *[
        f"score track {t}"
        for active_player in range(PLAYER_COUNT)
        for t in range(TRACK_CARD_COUNT)
      ],
    ]

