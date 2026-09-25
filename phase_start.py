from phase_rule import *
from expanse_game import *


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

class PhaseStart(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_START])
  
  
  def enumerate_kept_event(self, state: ExpanseState, card_embeds: torch.Tensor):
    # (batch, active player, card kept)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_START] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = True

    states: ExpanseState = states.repeat(1, PLAYER_COUNT, CARD_COUNT)
    states.obs_pile_embed[:, player_indices, :, OBS_PILE_EMBED_KEPT+player_indices, :] -= card_embeds[:CARD_COUNT].view(1, CARD_COUNT, state.CARD_EMBED_LENGTH)
    states.hid_pile_present[:, :, card_indices, HID_PILE_PRESENT_KEPT:HID_PILE_PRESENT_KEPT+PLAYER_COUNT, card_indices] = False
    states.hid_pile_index[:, :, :, HID_PILE_INDEX_FOCUS] = card_indices.view(1, CARD_COUNT)

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], PLAYER_COUNT, 1))
      # has card kept
      & (state.hid_pile_present[:, HID_PILE_PRESENT_KEPT+player_indices, :])
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

    track_card = state.hid_pile_index[:, HID_PILE_INDEX_TRACK:HID_PILE_INDEX_TRACK+TRACK_CARD_COUNT]

    states: ExpanseState = states.repeat(1, 1, PLAYER_COUNT, TRACK_CARD_COUNT)
    states.obs_pile_embed[:, TRACK_USE_FOCUS_START:TRACK_USE_FOCUS_END, :, :, OBS_PILE_EMBED_FOCUS, :] = \
      card_embeds[track_card, :].view(state.batch[0], 1, 1, TRACK_CARD_COUNT, state.CARD_EMBED_LENGTH)
    states.hid_pile_index[:, TRACK_USE_FOCUS_START:TRACK_USE_FOCUS_END, :, :, HID_PILE_INDEX_FOCUS] = \
      track_card.view(state.batch[0], 1, 1, TRACK_CARD_COUNT)
    states.obs_pile_embed[:, TRACK_USE_KEEP, player_indices, :, OBS_PILE_EMBED_KEPT+player_indices, :] += \
      card_embeds[track_card, :].view(1, state.batch[0], TRACK_CARD_COUNT, state.CARD_EMBED_LENGTH)
    states.hid_pile_present[
      state.batch_indices[0].view(state.batch[0], 1, 1),
      TRACK_USE_KEEP,
      player_indices.view(1, PLAYER_COUNT, 1),
      track_indices.view(1, 1, TRACK_CARD_COUNT),
      HID_PILE_PRESENT_KEPT+player_indices.view(1, PLAYER_COUNT, 1),
      (track_card.view(state.batch[0], 1, TRACK_CARD_COUNT) % CARD_COUNT)  # when % CARD_COUNT has effect it'll be masked out anyways
    ] = True
    states.hid_pile_index[
      :,
      :,
      :,
      track_indices.view(TRACK_CARD_COUNT, 1),
      HID_PILE_INDEX_TRACK+track_indices[:TRACK_CARD_COUNT-1].view(1, TRACK_CARD_COUNT-1)
    ] = track_card[:, track_advancement_matrix].view(state.batch[0], 1, 1, TRACK_CARD_COUNT, TRACK_CARD_COUNT-1)
    states.hid_pile_index[:, :, :, :, HID_PILE_INDEX_TRACK+TRACK_CARD_COUNT-1] = CARD_EMPTY_TRACK
    states.obs_pile_embed[:, :, :, :, OBS_PILE_EMBED_TRACK:OBS_PILE_EMBED_TRACK+TRACK_CARD_COUNT, :] = \
      card_embeds[states.hid_pile_index[:, :, :, :, HID_PILE_INDEX_TRACK:HID_PILE_INDEX_TRACK+TRACK_CARD_COUNT], :]
    states.obs_int[
      :,
      :,
      player_indices.view(PLAYER_COUNT, 1),
      track_indices.view(1, TRACK_CARD_COUNT),
      OBS_INT_CP+player_indices.view(PLAYER_COUNT, 1)
    ] -= track_cost.view(1, 1, 1, TRACK_CARD_COUNT)
    states.obs_int[:, TRACK_USE_AP, :, :, OBS_INT_PHASE_AP] = \
      card_ap[track_card].view(state.batch[0], 1, TRACK_CARD_COUNT)

    track_score = (track_card == CARD_SCORE)
    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT, 1))
      # enough cp to spend
      & (state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT, 1) >= track_cost.view(1, 1, 1, TRACK_CARD_COUNT))
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


