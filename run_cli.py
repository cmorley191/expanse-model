
from game import *
from phase_rule import *

import model

import torch
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')

phase_rules = get_phases()
for rule in phase_rules:
  print(f'{rule.get_type()} {rule}')


LOAD_WEIGHTS_PATH = os.path.join("weights", "0_centauri_3_weights___4858_281674.pth")

make_model = (
  lambda log: (
    model.ExpanseModel_Centauri(
      hidden_lengths=[128, 64],
      card_embed_length=8,
      log=log
    )
    .to(gpu_device)
  )
)
eval_model = make_model(log=True)

assert os.path.exists(LOAD_WEIGHTS_PATH), f'not found: {LOAD_WEIGHTS_PATH}'
eval_model.load_state_dict(torch.load(LOAD_WEIGHTS_PATH))


card_embeds = eval_model.card_embeds
state: ExpanseState = ExpanseState.generate_starting(1, card_embeds)
print("Starting track:")
for i_track in range(TRACK_CARD_COUNT):
  print(card_name[state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+i_track].item()])

user_player = None
while type(user_player) != type('') or len(user_player) != 1 or user_player not in 'MU':
  print(f'User player? M/U: ', end='')
  user_player = input().upper().strip()
user_player = 'MU'.index(user_player)

i_step = -1
while not (
  (state.obs_int[:, OBS_INT_DECK_PILES] < 0)
  | (
    (state.obs_int[:, OBS_INT_DECK_PILES] == 0)
    & (state.obs_int[:, OBS_INT_DECK_PILE_SCORES] == 0)
  )
).item():
  i_step += 1
  print()
  print()
  print(state.obs_bool)
  print(state.obs_int)
  print(f'Step {i_step}')
  print(f'Top track: {card_name[state.obs_slot_index[:, OBS_SLOT_INDEX_TRACK+TRACK_CARD_COUNT-1].item()]}')
  print(f'Focus: {card_name[state.obs_slot_index[:, OBS_SLOT_INDEX_FOCUS].item()]}')
  print(f'CP: {state.obs_int[:, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT].tolist()}')
  print(f'Score Sector: {state.obs_bool[:, OBS_BOOL_SCORE_SECTOR:OBS_BOOL_SCORE_SECTOR+SECTOR_COUNT].tolist()}')
  print(f'Sectors remaining: {state.obs_int[:, OBS_INT_BONUS_SECTORS:OBS_INT_BONUS_SECTORS+SECTOR_COUNT].tolist()}')

  rule = [r for r in phase_rules if r.matching(state).item()]
  print(rule)
  assert len(rule) == 1
  rule = rule[0]

  if rule.get_type() != PHASE_TYPE_CHOICE:
    print(f"Executing: {rule.action_str()}")
    state = rule.enumerate_actions(state, card_embeds)

  else:
    user_plays = state.obs_bool[:, OBS_BOOL_ACTION+user_player].item()

    (action_states, action_masks) = rule.enumerate_actions(state, card_embeds)
    action_str = rule.action_str()
    ACTIONS = len(action_str)

    if user_plays:
      print(f'User turn')
      print(f'Available actions:')
      for action in range(ACTIONS):
        if not action_masks[:, action].item():
          continue
        print(f'{action}. {action_str[action]}')
      print()
      action = None
      while type(action) != type(0) or action < 0 or action >= ACTIONS or not action_masks[:, action].item():
        print(f'Select action: ', end='')
        try:
          action = int(input())
        except ValueError:
          pass
      print(f'Selected {action}. {action_str[action]}')
      state = action_states.index(state.batch_indices[0], action)

    else:
      print(f'Model turn')

      model_value = eval_model.forward_eval(
        action_states.obs_bool.view(ACTIONS, OBS_BOOL_LENGTH),
        action_states.obs_int.view(ACTIONS, OBS_INT_LENGTH),
        action_states.obs_slot_index.view(ACTIONS, OBS_SLOT_INDEX_COUNT),
        action_states.obs_pile_cached_embed.view(ACTIONS, OBS_PILE_CACHED_EMBED_COUNT, card_embeds.embedding_dim)
      ).view(ACTIONS)
      model_action_player_value = (
        model_value * 
        (
          1 
          - (
            state.obs_bool[:, OBS_BOOL_ACTION+1].to(torch.float32).view(1)
            * 2
          )
        )
      )
      model_action_player_value = torch.where(
        action_masks.view(ACTIONS),
        model_action_player_value,
        -torch.inf
      )

      actions_ranked = model_action_player_value.argsort(dim=0, descending=True)
      print(f'Favorite actions:')
      for i_rank in range(min(5, action_masks.sum().item())):
        action = actions_ranked[i_rank].item()
        print(f'{action}. {action_str[action]} ({model_action_player_value[action].item():.2f})')

      action = actions_ranked[0].item()
      state = action_states.index(state.batch_indices[0], action)


print(f'Ended')
print(f'Final score:')
print(f'MCR {'(user): ' if user_player == 0 else '(model)'}: {state.obs_int[:, OBS_INT_CP].item()}')
print(f'UN  {'(user): ' if user_player == 1 else '(model)'}: {state.obs_int[:, OBS_INT_CP+1].item()}')

