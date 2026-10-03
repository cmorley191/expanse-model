from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 20
class PhaseEventMeta_20_Hybrid(PhaseEventMeta):

  def card(self):
    return CARD
  

class PhaseEvent_20_Hybrid_Select(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & (state.obs_bool[:, OBS_BOOL_PHASE_ARG_BASE:OBS_BOOL_PHASE_ARG_BASE+BASE_COUNT].any(dim=1).logical_not())
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, base)
    states: ExpanseState = state.clone().view(state.batch[0], 1).repeat(1, BASE_COUNT)
    states.obs_bool[:, base_indices, OBS_BOOL_PHASE_ARG_BASE+base_indices] = True

    mask = torch.ones((state.batch[0], BASE_COUNT), dtype=torch.bool, device=gpu_device)

    return (states, mask)

  def action_str(self):
    return [f"Target {b}" for b in base_name]


class PhaseEvent_20_Hybrid_Draw(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_STOCHASTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & (state.obs_bool[:, OBS_BOOL_PHASE_ARG_BASE:OBS_BOOL_PHASE_ARG_BASE+BASE_COUNT].any(dim=1))
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP] == CARD_EMPTY_REVEALED_DECK_TOP)
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_MAO_KWIK] = True
    new_state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP] = (
      torch.multinomial(
        torch.concat([
          (
            new_state.obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].float() 
            * (
              state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES].float()
              / (  # deck nonscores
                state.obs_int[:, OBS_INT_DECK_PILE_NONSCORES]
                + (state.obs_int[:, OBS_INT_DECK_PILES] * 8)
                + NONSCORES_NOT_IN_A_PILE
              ).float()
            ).view(state.batch[0], 1)
          ),
          state.obs_int[:, OBS_INT_DECK_PILE_SCORES].float().view(state.batch[0], 1)
        ], dim=1),
        1
      ).view(state.batch[0])
    )

    return new_state

  def action_str(self):
    return "reveal top"


CARD_MAO_KWIK = 3

EVENT_EFFECT_AP = 0
EVENT_EFFECT_MAO_KWIK = 1
EVENT_EFFECT_SCORE = 2
EVENT_EFFECT_COUNT = 3

class PhaseEVent_20_Hybrid_Resolve(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
      & (state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_MAO_KWIK])
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, effect)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_CHOOSE_MAO_KWIK] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_ARG_BASE:OBS_BOOL_PHASE_ARG_BASE+BASE_COUNT] = False
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    arg_base = state.obs_bool[:, OBS_BOOL_PHASE_ARG_BASE:OBS_BOOL_PHASE_ARG_BASE+BASE_COUNT].int().argmax(dim=1)
    state_influence = state.obs_int_influence()
    arg_base_influence = state_influence[state.batch_indices[0], arg_base, :]
    remaining_influence = (
      arg_base_influence.sum(dim=1, dtype=torch.int8).view(state.batch[0], 1)
      - torch.concat([
        # ap
        card_ap[state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP]].view(state.batch[0], 1),
        # mao kwik
        torch.tensor([4], dtype=torch.int8, device=gpu_device).view(1, 1).repeat(state.batch[0], 1),
        # score
        torch.tensor([torch.iinfo(torch.int8).max], dtype=torch.int8, device=gpu_device).view(1, 1).repeat(state.batch[0], 1)
      ], dim=1)
    ).clamp_min_(0)
    remaining_action_influence = torch.concat([
      remaining_influence.view(state.batch[0], EVENT_EFFECT_COUNT, 1),
      (
        arg_base_influence
        * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8)
      ).sum(dim=1, dtype=torch.int8).view(state.batch[0], 1, 1).repeat(1, EVENT_EFFECT_COUNT, 1),
    ], dim=2).min(dim=2).values

    states: ExpanseState = states.repeat(1, EVENT_EFFECT_COUNT)
    states.obs_int_influence()[state.batch_indices[0], :, arg_base, :] = torch.where(
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT),
      remaining_action_influence.view(state.batch[0], EVENT_EFFECT_COUNT, 1),
      (remaining_influence - remaining_action_influence).view(state.batch[0], EVENT_EFFECT_COUNT, 1)
    )
    states.obs_pile_present[:, EVENT_EFFECT_MAO_KWIK, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MAO_KWIK] = False
    states.obs_pile_cached_embed[:, EVENT_EFFECT_MAO_KWIK, OBS_PILE_CACHED_EMBED_KEPT:OBS_PILE_CACHED_EMBED_KEPT+PLAYER_COUNT, :] -= (
      card_embeds.weight[CARD_MAO_KWIK, :].view(1, 1, state.CARD_EMBED_LENGTH)
      * state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.float32).view(state.batch[0], PLAYER_COUNT, 1)
    )

    mask = torch.concat([
      (state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP] != CARD_SCORE).view(state.batch[0], 1),
      (
        (state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP] != CARD_SCORE)
        & (
          # action player
          state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
          # has mao-kwik
          & state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, CARD_MAO_KWIK]
          # exists
        ).any(dim=1)
      ).view(state.batch[0], 1),
      (state.obs_slot_index[:, OBS_SLOT_INDEX_REVEALED_DECK_TOP] == CARD_SCORE).view(state.batch[0], 1),
    ], dim=1)

    return (states, mask)

  def action_str(self):
    return [
      "remove top deck ap influence",
      "discard mao-kwik to remove 4 influence",
      "remove all influence",
    ]


