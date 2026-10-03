from phase_rule import *
from game import *

import torch
import torch.nn


CARD = 18
class PhaseEventMeta_18_StarHelix(PhaseEventMeta):

  def card(self):
    return CARD
  

event_single_counts = torch.tensor([1, 2], dtype=torch.uint8, device=gpu_device)

class PhaseEvent_18_StarHelix(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] == CARD)
    )

  def enumerate_separate(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, b0, b1)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, BASE_COUNT, BASE_COUNT)
    remove_player = state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not()
    states_influence = states.obs_int_influence()
    states_influence[:, base_indices, :, base_indices, :] -= remove_player.bool().to(torch.int8).view(1, state.batch[0], 1, PLAYER_COUNT)
    states_influence[:, :, base_indices, base_indices, :] -= remove_player.bool().to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)

    state_influence = state.obs_int_influence()
    #assert PLAYER_COUNT == 2
    both_have_influence = state_influence.bool().all(dim=2)
    mask = (
      both_have_influence.view(state.batch[0], BASE_COUNT, 1)
      & both_have_influence.view(state.batch[0], 1, BASE_COUNT)
      # ordering
      & (
        base_indices.view(1, BASE_COUNT, 1)
        < base_indices.view(1, 1, BASE_COUNT)
      )
    )

    states: ExpanseState = states.view(state.batch[0], BASE_COUNT * BASE_COUNT)
    mask = mask.view(state.batch[0], BASE_COUNT * BASE_COUNT)

    return (states, mask)

  def enumerate_single(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, count, base)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, event_single_counts.shape[0], BASE_COUNT)
    states_influence = states.obs_int_influence()
    states_influence[:, :, base_indices, base_indices, :] -= (
      state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().bool().to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)
      * event_single_counts.view(1, event_single_counts.shape[0], 1, 1)
    )

    state_influence = state.obs_int_influence()
    mask = (
      (
        (
          # turn player
          state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, PLAYER_COUNT)
          # with influence
          & state_influence.bool().view(state.batch[0], 1, BASE_COUNT, PLAYER_COUNT)
        )
        | (
          # opponent
          state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT)
          # with enough influence
          & (
            state_influence.view(state.batch[0], 1, BASE_COUNT, PLAYER_COUNT)
            >= event_single_counts.view(1, event_single_counts.shape[0], 1, 1)
          )
        )
        # both
      ).all(dim=3)
    )

    states: ExpanseState = states.view(state.batch[0], event_single_counts.shape[0] * BASE_COUNT)
    mask = mask.view(state.batch[0], event_single_counts.shape[0] * BASE_COUNT)

    return (states, mask)

  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.nn.Embedding):
    # (batch, 1)
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    mask = torch.ones((state.batch[0], 1), dtype=torch.bool, device=gpu_device)

    return (states, mask)
  
  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_separate,
      self.enumerate_single,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)
  
  def action_str(self):
    return [
      *[
        f"remove {b0} and {b1}"
        for b0 in base_name
        for b1 in base_name
      ],
      *[
        f"remove {c} {b}"
        for c in event_single_counts.tolist()
        for b in base_name
      ],
      "don't remove"
    ]

