from phase_rule import *
from game import *

import torch


class PhaseDone(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC

  def is_nature(self):
    return PHASE_TYPE_STOCHASTIC

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_TURN_TYPE_START:OBS_BOOL_PHASE_TURN_TYPE_END].any(dim=1).logical_not()
      & state.obs_bool[:, OBS_BOOL_PHASE_DONE]
    )

  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_START] = True
    #assert PLAYER_COUNT == 2
    new_state.obs_bool[:, OBS_BOOL_TURN:OBS_BOOL_TURN+PLAYER_COUNT] = new_state.obs_bool[:, OBS_BOOL_TURN:OBS_BOOL_TURN+PLAYER_COUNT].logical_not()
    new_state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = new_state.obs_bool[:, OBS_BOOL_TURN:OBS_BOOL_TURN+PLAYER_COUNT]
    #assert NONSCORES_NOT_IN_A_PILE > 0   # to avoid /-by-0
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1] = torch.multinomial(
      torch.concat([
        (
          new_state.obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].float() 
          * (
            state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES].float()
            / (  # deck nonscores
              state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES]
              + (state.obs_int[:, OBS_INT_DECK_PILES] * 8)
              + NONSCORES_NOT_IN_A_PILE
            )
          ).view(state.batch[0], 1)
        ),
        state.obs_int[:, OBS_INT_DECK_PILE_SCORES].float().view(state.batch[0], 1)
      ], dim=1),
      1
    ).view(state.batch[0])
    new_state.obs_pile_cached_embed[:, OBS_PILE_CACHED_EMBED_DECK, :] -= card_embeds(new_state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1])
    new_pile = (new_state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES] + new_state.obs_int[:, OBS_INT_DECK_PILE_SCORES] == 1)
    new_state.obs_int[:, OBS_INT_DECK_PILE_SCORES] = torch.where(
      new_pile,
      SCORES_PER_PILE,
      (
        new_state.obs_int[:, OBS_INT_DECK_PILE_SCORES]
        - (new_state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1] == CARD_SCORE).to(torch.int8)
      )
    )
    new_state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES] = torch.where(
      new_pile,
      NONSCORES_PER_PILE,
      (
        new_state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES]
        - (new_state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1] != CARD_SCORE).to(torch.int8)
      )
    )
    new_state.obs_int[:, OBS_INT_DECK_PILES] -= new_pile.to(torch.int8)

    return new_state


