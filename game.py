import sys
import typing

if 'torch' not in sys.modules:
  print(f"Loading torch")
import torch
import torch.nn

cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')
assert torch.cuda.is_available()


PLAYER_COUNT = 2
assert PLAYER_COUNT == 2  # needed by many phase rules
player_indices = torch.arange(PLAYER_COUNT, dtype=torch.long, device=gpu_device)

SECTOR_COUNT = 3
sector_indices = torch.arange(SECTOR_COUNT, dtype=torch.long, device=gpu_device)
bonus_sector_points = torch.tensor([
  [0, 0],
  [1, 1],
  [2, 1],
  [2, 1],
  [3, 2],
  [3, 2],
], dtype=torch.int8, device=gpu_device)
STARTING_BONUS_SECTORS = 2

ORBITAL_COUNT = 8
orbital_indices = torch.arange(ORBITAL_COUNT, dtype=torch.long, device=gpu_device)
orbital_adjacent = torch.tensor([
  [False, True, True, True, True, True, False, False],
  [True, False, True, True, True, True, False, False],
  [True, True, False, True, True, True, True, False],
  [True, True, True, False, True, True, True, False],
  [True, True, True, True, False, True, True, False],
  [True, True, True, True, True, False, True, False],
  [False, False, True, True, True, True, False, True],
  [False, False, False, False, False, False, True, False],
], dtype=torch.bool, device=gpu_device)
player_home_orbital = torch.tensor([0, 1], dtype=torch.long, device=gpu_device)
orbital_sector = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.long, device=gpu_device)

BASE_COUNT = 12
base_indices = torch.arange(BASE_COUNT, dtype=torch.long, device=gpu_device)
base_orbital = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7], dtype=torch.long, device=gpu_device)
base_sector = orbital_sector[base_orbital]

TRACK_PRESENT = 0
TRACK_SCORE = 1
TRACK_POINTS = 2
TRACK_FEATURE_COUNT = 3

TRACK_CARD_COUNT = 5
track_indices = torch.arange(TRACK_CARD_COUNT, dtype=torch.long, device=gpu_device)
track_cost = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int8, device=gpu_device)

FLEET_COUNT = 5
fleet_indices = torch.arange(FLEET_COUNT, dtype=torch.long, device=gpu_device)

CARD_COUNT = 30
CARD_SCORE = CARD_COUNT
CARD_EMPTY_TRACK = CARD_COUNT + 1
CARD_EMPTY_FOCUS = CARD_COUNT + 2
EXTRA_CARD_INDEX_COUNT = 3
card_indices = torch.arange(CARD_COUNT, dtype=torch.long, device=gpu_device)
card_ap = torch.tensor([
  3, 4, 4, 3, 2, 3, 4, 4, 3, 2,
  4, 4, 3, 3, 4, 2, 4, 4, 3, 3,
  4, 2, 3, 4, 3, 2, 3, 3, 2, 4,
  0, 0,
], dtype=torch.int8, device=gpu_device)
card_factions = torch.tensor([
  [True, False],
  [False, True],
  [False, True],
  [True, True],
  [True, False],

  [True, False],
  [False, True],
  [False, True],
  [True, False],
  [True, False],

  [True, True],
  [False, True],
  [True, False],
  [False, True],
  [False, True],

  [True, False],
  [True, True],
  [True, True],
  [True, True],
  [True, False],

  [True, True],
  [True, True],
  [True, False],
  [True, True],
  [True, False],

  [False, True],
  [False, True],
  [True, True],
  [False, True],
  [True, True],
], dtype=torch.bool, device=gpu_device)
card_factions = torch.concat([
  card_factions,
  torch.zeros((EXTRA_CARD_INDEX_COUNT, 2), dtype=torch.bool, device=gpu_device)
], dim=0)
card_keep_cost = torch.ones((CARD_COUNT+EXTRA_CARD_INDEX_COUNT,), dtype=torch.int8, device=gpu_device)
card_keep_cost[1:3] = 0  # miller, cotyar, mao-kwik

STARTING_DECK_PILE_COUNT = 3
SCORES_PER_PILE = 2
NONSCORES_PER_PILE = 8
NONSCORES_NOT_IN_A_PILE = CARD_COUNT - TRACK_CARD_COUNT - (STARTING_DECK_PILE_COUNT * NONSCORES_PER_PILE)
assert NONSCORES_NOT_IN_A_PILE > 0  # needed for phase_done


starting_fleets = [
  [0, 2], # Earth
  [2, 0], # Mars
  [1, 1], # Ceres
  [0, 1], # Tycho
  [0, 0], # Eros
  [0, 0], # Thoth
  [1, 1], # Jupiter
  [1, 0], # Saturn
]

starting_influences = [
  [0, 1], # Eurasia
  [0, 1], # Africa
  [1, 0], # MarinerValley
  [1, 0], # LondresNova
  [0, 0], # Ceres
  [0, 0], # Tycho
  [0, 0], # Eros
  [0, 0], # Thoth
  [1, 0], # Europa
  [0, 0], # Ganymede
  [0, 1], # Rhea
  [0, 0], # Titan
]

template_starting_obs_bool = torch.tensor(
  [True]
  + ([False] * 10)
  + ([False] * PLAYER_COUNT)
  + ([False] * PLAYER_COUNT)
  + ([False] * SECTOR_COUNT)
, dtype=torch.bool, device=gpu_device)
OBS_BOOL_LENGTH = template_starting_obs_bool.shape[0]
OBS_BOOL_PHASE_START = 0
OBS_BOOL_PHASE_TURN_TYPE_START = OBS_BOOL_PHASE_START + 1
OBS_BOOL_PHASE_AP_TURN = OBS_BOOL_PHASE_TURN_TYPE_START
OBS_BOOL_PHASE_EVENT_TURN = OBS_BOOL_PHASE_AP_TURN + 1
OBS_BOOL_PHASE_SCORE_TURN = OBS_BOOL_PHASE_EVENT_TURN + 1
OBS_BOOL_PHASE_TURN_TYPE_END = OBS_BOOL_PHASE_SCORE_TURN + 1
OBS_BOOL_PHASE_ACTION = OBS_BOOL_PHASE_TURN_TYPE_END
OBS_BOOL_PHASE_ACTION_DONE = OBS_BOOL_PHASE_ACTION + 1
OBS_BOOL_PHASE_EVENT = OBS_BOOL_PHASE_ACTION_DONE + 1
OBS_BOOL_PHASE_EVENT_DONE = OBS_BOOL_PHASE_EVENT + 1
OBS_BOOL_PHASE_CHOOSE_SECTOR = OBS_BOOL_PHASE_EVENT_DONE + 1
OBS_BOOL_PHASE_CHOOSE_EVENT = OBS_BOOL_PHASE_CHOOSE_SECTOR + 1
OBS_BOOL_PHASE_DONE = OBS_BOOL_PHASE_CHOOSE_EVENT + 1
OBS_BOOL_TURN = OBS_BOOL_PHASE_DONE + 1
OBS_BOOL_ACTION = OBS_BOOL_TURN + PLAYER_COUNT
OBS_BOOL_SCORE_SECTOR = OBS_BOOL_ACTION + PLAYER_COUNT
OBS_BOOL_END = OBS_BOOL_SCORE_SECTOR + SECTOR_COUNT
assert OBS_BOOL_END == OBS_BOOL_LENGTH


template_starting_obs_int = torch.tensor(
  [0]
  + [0]
  + ([10] * PLAYER_COUNT)
  + [n for o in starting_fleets for n in o]
  + [n for b in starting_influences for n in b]
  + [SCORES_PER_PILE]
  + [NONSCORES_PER_PILE]
  + [STARTING_DECK_PILE_COUNT-1]
  + ([STARTING_BONUS_SECTORS] * SECTOR_COUNT)
, dtype=torch.int8, device=gpu_device)
OBS_INT_LENGTH = template_starting_obs_int.shape[0]
OBS_INT_PHASE_AP = 0
OBS_INT_PHASE_EVENT_ACTIONS = OBS_INT_PHASE_AP + 1
OBS_INT_CP = OBS_INT_PHASE_EVENT_ACTIONS + 1
OBS_INT_FLEETS = OBS_INT_CP + PLAYER_COUNT
OBS_INT_INFLUENCE = OBS_INT_FLEETS + (ORBITAL_COUNT * PLAYER_COUNT)
OBS_INT_DECK_PILE_SCORES = OBS_INT_INFLUENCE + (BASE_COUNT * PLAYER_COUNT)
OBS_INT_DECK_PILE_NONSCORES = OBS_INT_DECK_PILE_SCORES + 1
OBS_INT_DECK_PILES = OBS_INT_DECK_PILE_NONSCORES + 1
OBS_INT_BONUS_SECTORS = OBS_INT_DECK_PILES + 1
OBS_INT_END = OBS_INT_BONUS_SECTORS + SECTOR_COUNT
assert OBS_INT_END == OBS_INT_LENGTH


OBS_SLOT_INDEX_FOCUS = 0
OBS_SLOT_INDEX_TRACK = OBS_SLOT_INDEX_FOCUS + 1
OBS_SLOT_INDEX_COUNT = OBS_SLOT_INDEX_TRACK + TRACK_CARD_COUNT


OBS_PILE_PRESENT_DECK = 0
OBS_PILE_PRESENT_KEPT = OBS_PILE_PRESENT_DECK + 1
OBS_PILE_PRESENT_COUNT = OBS_PILE_PRESENT_KEPT + PLAYER_COUNT


OBS_PILE_CACHED_EMBED_DECK = 0
OBS_PILE_CACHED_EMBED_KEPT = OBS_PILE_CACHED_EMBED_DECK + 1
OBS_PILE_CACHED_EMBED_COUNT = OBS_PILE_CACHED_EMBED_KEPT + PLAYER_COUNT



template_starting_hid_bool = torch.tensor(
  ([False] * SECTOR_COUNT)
, dtype=torch.bool, device=gpu_device)
HID_BOOL_LENGTH = template_starting_hid_bool.shape[0]
HID_BOOL_SCORE_SECTOR = 0
HID_BOOL_END = HID_BOOL_SCORE_SECTOR + SECTOR_COUNT
assert HID_BOOL_END == HID_BOOL_LENGTH



class ExpanseState():
  def __init__(self,
               obs_bool: torch.Tensor,
               obs_int: torch.Tensor,
               obs_slot_index: torch.Tensor,
               obs_pile_present: torch.Tensor,
               obs_pile_cached_embed: torch.Tensor,
               hid_bool: torch.Tensor,
               *,
               batch_indices: torch.Tensor | None = None):
    self.batch = obs_bool.shape[:-1]
    if batch_indices is not None:
      self.batch_indices = batch_indices
    else:
      self.batch_indices = [torch.arange(b, dtype=torch.long, device=gpu_device) for b in self.batch]
    self.obs_bool = obs_bool
    self.obs_int = obs_int
    self.obs_slot_index = obs_slot_index
    self.obs_pile_present = obs_pile_present
    self.obs_pile_cached_embed = obs_pile_cached_embed
    self.CARD_EMBED_LENGTH = obs_pile_cached_embed.shape[-1]
    self.hid_bool = hid_bool


  def obs_int_fleets(self):
    """(*batch, orbital, player)"""
    return self.obs_int[..., OBS_INT_FLEETS:OBS_INT_FLEETS+(ORBITAL_COUNT*PLAYER_COUNT)].view(*self.batch, ORBITAL_COUNT, PLAYER_COUNT)

  def obs_int_influence(self):
    """(*batch, base, player)"""
    return self.obs_int[..., OBS_INT_INFLUENCE:OBS_INT_INFLUENCE+(BASE_COUNT*PLAYER_COUNT)].view(*self.batch, BASE_COUNT, PLAYER_COUNT)


  def concat(elements: list[typing.Self], dim: int) -> typing.Self:
    return ExpanseState(
      obs_bool=torch.concat([e.obs_bool for e in elements], dim=dim),
      obs_int=torch.concat([e.obs_int for e in elements], dim=dim),
      obs_slot_index=torch.concat([e.obs_slot_index for e in elements], dim=dim),
      obs_pile_present=torch.concat([e.obs_pile_present for e in elements], dim=dim),
      obs_pile_cached_embed=torch.concat([e.obs_pile_cached_embed for e in elements], dim=dim),
      hid_bool=torch.concat([e.hid_bool for e in elements], dim=dim)
    )

  def index(self, *index) -> typing.Self:
    return ExpanseState(
      obs_bool=self.obs_bool[*index, :],
      obs_int=self.obs_int[*index, :],
      obs_slot_index=self.obs_slot_index[*index, :],
      obs_pile_present=self.obs_pile_present[*index, :, :],
      obs_pile_cached_embed=self.obs_pile_cached_embed[*index, :, :],
      hid_bool=self.hid_bool[*index, :]
    )

  def clone(self, *args, **kwargs) -> typing.Self:
    return ExpanseState(
      obs_bool=self.obs_bool.clone(*args, **kwargs),
      obs_int=self.obs_int.clone(*args, **kwargs),
      obs_slot_index=self.obs_slot_index.clone(*args, **kwargs),
      obs_pile_present=self.obs_pile_present.clone(*args, **kwargs),
      obs_pile_cached_embed=self.obs_pile_cached_embed.clone(*args, **kwargs),
      hid_bool=self.hid_bool.clone(*args, **kwargs),
      batch_indices=self.batch_indices
    )

  def view(self, *shape) -> typing.Self:
    return ExpanseState(
      obs_bool=self.obs_bool.view(*shape, OBS_BOOL_LENGTH),
      obs_int=self.obs_int.view(*shape, OBS_INT_LENGTH),
      obs_slot_index=self.obs_slot_index.view(*shape, OBS_SLOT_INDEX_COUNT),
      obs_pile_present=self.obs_pile_present.view(*shape, OBS_PILE_PRESENT_COUNT, CARD_COUNT),
      obs_pile_cached_embed=self.obs_pile_cached_embed.view(*shape, OBS_PILE_CACHED_EMBED_COUNT, self.CARD_EMBED_LENGTH),
      hid_bool=self.hid_bool.view(*shape, HID_BOOL_LENGTH)
    )
  
  def repeat(self, *repeats) -> typing.Self:
    return ExpanseState(
      obs_bool=self.obs_bool.repeat(*repeats, 1),
      obs_int=self.obs_int.repeat(*repeats, 1),
      obs_slot_index=self.obs_slot_index.repeat(*repeats, 1),
      obs_pile_present=self.obs_pile_present.repeat(*repeats, 1, 1),
      obs_pile_cached_embed=self.obs_pile_cached_embed.repeat(*repeats, 1, 1),
      hid_bool=self.hid_bool.repeat(*repeats, 1)
    )


  def generate_empty(card_embeds: torch.nn.Embedding) -> typing.Self:
    return ExpanseState(
      obs_bool=torch.zeros((0, OBS_BOOL_LENGTH), dtype=torch.bool, device=gpu_device),
      obs_int=torch.zeros((0, OBS_INT_LENGTH), dtype=torch.int8, device=gpu_device),
      obs_slot_index=torch.zeros((0, OBS_SLOT_INDEX_COUNT), dtype=torch.long, device=gpu_device),
      obs_pile_present=torch.zeros((0, OBS_PILE_PRESENT_COUNT, CARD_COUNT), dtype=torch.bool, device=gpu_device),
      obs_pile_cached_embed=torch.zeros((0, OBS_PILE_CACHED_EMBED_COUNT, card_embeds.embedding_dim), dtype=torch.float16, device=gpu_device),
      hid_bool=torch.zeros((0, HID_BOOL_LENGTH), dtype=torch.bool, device=gpu_device),
      batch_indices=[torch.zeros((0,), dtype=torch.long, device=gpu_device)]
    )

  def generate_starting(BATCH: int, card_embeds: torch.nn.Embedding) -> typing.Self:
    state = ExpanseState(
      obs_bool=template_starting_obs_bool.view(1, OBS_BOOL_LENGTH).repeat(BATCH, 1),
      obs_int=template_starting_obs_int.view(1, OBS_INT_LENGTH).repeat(BATCH, 1),
      obs_slot_index=torch.zeros((BATCH, OBS_SLOT_INDEX_COUNT), dtype=torch.long, device=gpu_device),
      obs_pile_present=torch.ones((BATCH, OBS_PILE_PRESENT_COUNT, CARD_COUNT), dtype=torch.bool, device=gpu_device),
      obs_pile_cached_embed=torch.zeros((BATCH, OBS_PILE_CACHED_EMBED_COUNT, card_embeds.embedding_dim), dtype=torch.float16, device=gpu_device),
      hid_bool=template_starting_hid_bool.view(1, HID_BOOL_LENGTH).repeat(BATCH, 1)
    )

    state.obs_bool[state.batch_indices[0], OBS_BOOL_TURN+torch.randint(0, PLAYER_COUNT, (BATCH,), device=gpu_device)] = True
    state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = state.obs_bool[:, OBS_BOOL_TURN:OBS_BOOL_TURN+PLAYER_COUNT]

    state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS] = CARD_EMPTY_FOCUS

    state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK:OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT] = \
      torch.multinomial(torch.ones((BATCH, CARD_COUNT), dtype=torch.float32, device=gpu_device), TRACK_CARD_COUNT)
    state.obs_pile_present[
      state.batch_indices[0].view(state.batch[0], 1), 
      OBS_PILE_PRESENT_DECK,
      state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK:OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT]
    ] = False

    state.obs_pile_present[:, OBS_PILE_PRESENT_KEPT:OBS_PILE_PRESENT_KEPT+PLAYER_COUNT, :] = False

    state.obs_pile_cached_embed[:, OBS_PILE_CACHED_EMBED_DECK, :] = (
      (card_embeds.weight[:CARD_COUNT, :].sum(dim=0).view(1, state.CARD_EMBED_LENGTH))
      - (card_embeds(state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK:OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT]).sum(dim=1))
    )
    return state



