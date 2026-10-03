from phase_rule import *
from game import *

import torch
import torch.nn


CARD_EARTH = 12
class PhaseEventMeta_12_BlockadeEarth(PhaseEventMeta):

  def card(self):
    return CARD_EARTH


CARD_MARS = 13
class PhaseEventMeta_13_BlockadeMars(PhaseEventMeta):

  def card(self):
    return CARD_MARS


event_card = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.bool, device=gpu_device)
event_card_orbital = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.long, device=gpu_device)
EVENT_BASE_COUNT = 2
event_card_bases = torch.zeros((CARD_COUNT+EXTRA_CARD_INDEX_COUNT, EVENT_BASE_COUNT), dtype=torch.long, device=gpu_device)
event_card_base_indices = torch.arange(event_card_bases.shape[1], dtype=torch.long, device=gpu_device)
for (card, orbital, bases) in [
  (CARD_EARTH, 0, [0, 1]),
  (CARD_MARS, 1, [2, 3]),
]:
  event_card[card] = True
  event_card_orbital[card] = orbital
  event_card_bases[card, :] = torch.tensor(bases, dtype=torch.long, device=gpu_device)

class PhaseEvent__Blockade(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_EVENT]
      & (event_card[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]])
    )

  def enumerate_actions(self, state, card_embeds):
    # (batch, place base, remove base)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT] = False
    states.obs_bool[:, :, :, OBS_BOOL_PHASE_EVENT_DONE] = True
    states.obs_bool[:, :, :, OBS_BOOL_PLAYER_EVENT:OBS_BOOL_PLAYER_EVENT+PLAYER_COUNT] = False
    states.obs_slot_index[:, :, :, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    states: ExpanseState = states.repeat(1, EVENT_BASE_COUNT, EVENT_BASE_COUNT+1)
    states_influence = states.obs_int_influence()
    states_influence[
      state.batch_indices[0].view(state.batch[0], 1),
      event_card_base_indices.view(1, EVENT_BASE_COUNT),
      :,
      event_card_bases[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]],
      :
    ] += state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)
    states_influence[
      state.batch_indices[0].view(state.batch[0], 1),
      :,
      event_card_base_indices.view(1, EVENT_BASE_COUNT),
      event_card_bases[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]],
      :
    ] -= state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().to(torch.int8).view(state.batch[0], 1, 1, PLAYER_COUNT)
  
    state_event_card_orbital_fleets = state.obs_int_fleets()[
      state.batch_indices[0],
      event_card_orbital[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]],
      :
    ]
    mask = (
      torch.concat([
        (
          (
            # action player
            state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT]
            # with orbital control
            & (state_event_card_orbital_fleets > state_event_card_orbital_fleets.flip(dims=[1]))
            # exists
          ).any(dim=1).view(state.batch[0], 1, 1)
          & (
            # removal player
            state.obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].logical_not().view(state.batch[0], 1, 1, PLAYER_COUNT)
            # with influence on removal base
            & (
              state.obs_int_influence()[
                state.batch_indices[0].view(state.batch[0], 1, 1),
                event_card_bases[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS]].view(state.batch[0], 1, EVENT_BASE_COUNT),
                :
              ]
              != 0
            )
            # exists
          ).any(dim=3)
        ),
        # can always pass on removal
        torch.ones((state.batch[0], 1, 1), dtype=torch.bool, device=gpu_device),
      ], dim=2)
      .repeat(1, EVENT_BASE_COUNT, 1)
    )

    states: ExpanseState = states.view(state.batch[0], EVENT_BASE_COUNT * (EVENT_BASE_COUNT + 1))
    mask = mask.view(state.batch[0], EVENT_BASE_COUNT * (EVENT_BASE_COUNT + 1))

    return (states, mask)

  def action_str(self):
    return [
      f"place {pb}th base; {removal}"
      for pb in range(EVENT_BASE_COUNT)
      for removal in [*[f"remove {rb}th base" for rb in range(EVENT_BASE_COUNT)], "don't remove"]
    ]

