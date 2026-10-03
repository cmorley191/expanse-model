import time

from game import *
from phase_rule import *

import torch
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')

phase_rules = get_phases()
for rule in phase_rules:
  print(f'{rule.get_type()} {rule}')

card_embeds = torch.nn.Embedding(num_embeddings=CARD_COUNT+EXTRA_CARD_INDEX_COUNT, embedding_dim=8, dtype=torch.float32, device=gpu_device)
card_embeds.weight.requires_grad = False

TOTAL_BATCH = 8192 * 2
unsorted_state = ExpanseState.generate_starting(
  BATCH=TOTAL_BATCH,
  card_embeds=card_embeds
)
sorted_states = [[ExpanseState.generate_empty(card_embeds)] for _ in phase_rules]
ended_states: list[ExpanseState] = []

start_time = time.time()
i_step = -1
while unsorted_state.batch[0] != 0:
  i_step += 1
  print(f'Step {i_step} - ', end='')

  unsorted_state_matches_rules = torch.concat([
    rule.matching(unsorted_state).view(unsorted_state.batch[0], 1)
    for rule in phase_rules
  ], dim=1)
  state_matches_one_rule = (unsorted_state_matches_rules.sum(dim=1) == 1)
  if not state_matches_one_rule.all().item():
    print(f'Non-matching phase rule')
    i_example = state_matches_one_rule.int().argmin(dim=0).item()
    example = unsorted_state.index(i_example)
    print(example.obs_bool)
    print(example.obs_int)
    print(example.hid_bool)
    assert False

  unsorted_state_rule = unsorted_state_matches_rules.int().argmax(dim=1)
  for i_rule in range(len(phase_rules)):
    sorted_states[i_rule].append(unsorted_state.index(unsorted_state_rule == i_rule))
  rule_state_count = torch.tensor([sum([s.batch[0] for s in r]) for r in sorted_states], dtype=torch.long, device=cpu_device)
  largest_rules = rule_state_count.argsort(descending=True)
  largest_rules_cumsum = rule_state_count[largest_rules].cumsum(dim=0)

  rule_state_count = rule_state_count.to(gpu_device)
  largest_rules = largest_rules.to(gpu_device)
  largest_rules_cumsum = largest_rules_cumsum.to(gpu_device)

  largest_rule_excluded = (largest_rules_cumsum > (TOTAL_BATCH // 2))
  included_largest_rules = (
    largest_rules
    if largest_rule_excluded.all() or (not largest_rule_excluded.any())
    else largest_rules[:largest_rule_excluded.int().argmax() + 1]
  )

  next_unsorted_states: list[ExpanseState] = []
  processing_state_count = largest_rules_cumsum[included_largest_rules.shape[0] - 1].item()
  print(f'processing {processing_state_count} games ({processing_state_count * 100 / TOTAL_BATCH:.1f}%) from {included_largest_rules.shape[0]} / {len(phase_rules)} rules')
  for i_rule in included_largest_rules:
    #print(phase_rules[i_rule])
    state: ExpanseState = ExpanseState.concat(sorted_states[i_rule], dim=0)
    sorted_states[i_rule] = [ExpanseState.generate_empty(card_embeds)]

    if phase_rules[i_rule].get_type() == PHASE_TYPE_CHOICE:
      torch.cuda.synchronize()
      ret = phase_rules[i_rule].enumerate_actions(state, card_embeds)
      torch.cuda.synchronize()
      assert type(ret) == type(tuple()), f'bad return {phase_rules[i_rule]}'
      (action_states, action_mask) = ret
      assert action_states.batch == action_mask.shape, f'bad action shapes {phase_rules[i_rule]}'

      assert action_mask.any(dim=1).all(), f'blank action mask {phase_rules[i_rule]}'

      i_selected = torch.multinomial(action_mask.float(), 1).view(state.batch[0])
      assert action_mask[state.batch_indices[0], i_selected].all()
      next_unsorted_states.append(
        action_states.index(
          state.batch_indices[0],
          i_selected
        )
      )

    else:
      torch.cuda.synchronize()
      new_state = phase_rules[i_rule].enumerate_actions(state, card_embeds)
      torch.cuda.synchronize()
      assert new_state is not None, f'bad return {phase_rules[i_rule]}'
      assert state.batch == new_state.batch, f'bad new state shape {phase_rules[i_rule]}'
      next_unsorted_states.append(new_state)

    unsynchronized_deck = (
      next_unsorted_states[-1].obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].int().sum(dim=1)
      != (
        next_unsorted_states[-1].obs_int[:, OBS_INT_DECK_PILE_NONSCORES]
        + (NONSCORES_PER_PILE * next_unsorted_states[-1].obs_int[:, OBS_INT_DECK_PILES])
        + NONSCORES_NOT_IN_A_PILE
      )
    )
    assert not unsynchronized_deck.any(), f'unsynchronized deck after {phase_rules[i_rule]}\n\n' \
      f'before int:\n' \
      f'{state.obs_int[unsynchronized_deck]}\n' \
      f'before present:\n' \
      f'{state.obs_pile_present[unsynchronized_deck, OBS_PILE_PRESENT_DECK, :]}\n' \
      f'after int:\n' \
      f'{next_unsorted_states[-1].obs_int[unsynchronized_deck]}\n' \
      f'after present:\n' \
      f'{next_unsorted_states[-1].obs_pile_present[unsynchronized_deck, OBS_PILE_PRESENT_DECK, :]}\n' \
      f'after index:\n' \
      f'{next_unsorted_states[-1].obs_slot_index[unsynchronized_deck]}\n'
    assert ((
      next_unsorted_states[-1].obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].any(dim=1)
      | next_unsorted_states[-1].obs_int[:, OBS_INT_DECK_PILE_SCORES].bool()
    ).all()), f'Empty deck after {phase_rules[i_rule]}\n\n' \
      f'before:\n' \
      f'{state.obs_pile_present[(
        next_unsorted_states[-1].obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].any(dim=1)
        | next_unsorted_states[-1].obs_int[:, OBS_INT_DECK_PILE_SCORES].bool()
      ).logical_not()]}\n' \
      f'int:\n' \
      f'{state.obs_int[(
        next_unsorted_states[-1].obs_pile_present[:, OBS_PILE_PRESENT_DECK, :].any(dim=1)
        | next_unsorted_states[-1].obs_int[:, OBS_INT_DECK_PILE_SCORES].bool()
      ).logical_not()]}'
    assert (next_unsorted_states[-1].obs_bool[:, OBS_BOOL_PLAYER_TURN:OBS_BOOL_PLAYER_TURN+PLAYER_COUNT].int().sum(dim=1) == 1).all(), f'Bad turn after {phase_rules[i_rule]}'
    assert (next_unsorted_states[-1].obs_bool[:, OBS_BOOL_PLAYER_ACTION:OBS_BOOL_PLAYER_ACTION+PLAYER_COUNT].int().sum(dim=1) <= 1).all(), f'Bad action after {phase_rules[i_rule]}'
    non_piles_mask = torch.ones((OBS_INT_LENGTH,), dtype=torch.bool, device=gpu_device)
    non_piles_mask[OBS_INT_DECK_PILES] = False
    assert (next_unsorted_states[-1].obs_int[:, non_piles_mask] >= 0).all(), f'Negative after rule {phase_rules[i_rule]}\n\n'\
      f'before:\n' \
      f'{state.obs_bool[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1)][:5]}\n' \
      f'{state.obs_int[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1)][:5]}\n' \
      f'action:\n' \
      f'{None if phase_rules[i_rule].get_type() != PHASE_TYPE_CHOICE else i_selected[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1)]}\n' \
      f'[{None if phase_rules[i_rule].get_type() != PHASE_TYPE_CHOICE else '; '.join([phase_rules[i_rule].action_str()[i] for i in i_selected[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1).tolist()]])}\n' \
      f'{None if phase_rules[i_rule].get_type() != PHASE_TYPE_CHOICE else action_mask[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1)]}\n' \
      f'after:\n' \
      f'{next_unsorted_states[-1].obs_int[(next_unsorted_states[-1].obs_int[:, :] < 0).any(dim=1)][:5]}\n'

  next_unsorted_state: ExpanseState = ExpanseState.concat([
    ExpanseState.generate_empty(card_embeds),
    *[s for s in next_unsorted_states if s.batch[0] != 0],
  ], dim=0)
  ended_state_mask: torch.Tensor = (
    (next_unsorted_state.obs_int[:, OBS_INT_DECK_PILES] < 0)
    | (
      (next_unsorted_state.obs_int[:, OBS_INT_DECK_PILES] == 0)
      & (next_unsorted_state.obs_int[:, OBS_INT_DECK_PILE_SCORES] == 0)
    )
  )
  ended_states.append(next_unsorted_state.index(ended_state_mask))
  unsorted_state: ExpanseState = next_unsorted_state.index(ended_state_mask.logical_not())

stop_time = time.time()

ended_state: ExpanseState = ExpanseState.concat(ended_states, dim=0)

print(f'Completed {ended_state.batch[0]} games in {stop_time-start_time:.2f}s ({ended_state.batch[0]/(stop_time-start_time):.2f} games/s)')
print(f'MCR wins: {(ended_state.obs_int[:, OBS_INT_CP] > ended_state.obs_int[:, OBS_INT_CP+1]).sum()}')
print(f'UN wins: {(ended_state.obs_int[:, OBS_INT_CP] < ended_state.obs_int[:, OBS_INT_CP+1]).sum()}')
print(f'Ties: {(ended_state.obs_int[:, OBS_INT_CP] == ended_state.obs_int[:, OBS_INT_CP+1]).sum()}')

