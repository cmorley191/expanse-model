import math
import threading


from game import *
from phase_rule import *

import model
import memory as memory_

import torch
import torch.distributions
import torch.nn.functional as F
cpu_device = torch.device('cpu')
gpu_device = torch.device('cuda')

phase_rules = get_phases()
for rule in phase_rules:
  print(f'{rule.get_type()} {rule}')


# SAVE_WEIGHTS_FILENAME: network weights are saved here at intervals (with an automatic number added to the name)
WEIGHTS_DIR = "weights"
SAVE_WEIGHTS_FILENAME = "0_centauri_5_weights___.pth"
SAVE_RATE_EPISODES = 10_000
# LOAD_WEIGHTS_FILENAME: the network will load these weights if this file exists (see pausing/resuming training below)
LOAD_WEIGHTS_FILENAME = "0_centauri_4_weights___11090_936366.pth"
STARTING_I_STEP = -1
STARTING_I_EPISODE = 0

LAMBDA = 0.98

make_model = (
  lambda log: (
    model.ExpanseModel_Centauri(
      hidden_lengths=[128, 64, 64],
      card_embed_length=8,
      log=log
    )
    .to(gpu_device)
  )
)
train_model = make_model(log=True)

if os.path.exists(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS_FILENAME)):
  print(f"LOADING STARTING WEIGHTS: {os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS_FILENAME)}")
  train_model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS_FILENAME)))
else:
  print()
  print("!!!!!!!!!")
  print("STARTING FROM SCRATCH! (no starting weights file was found)")
  print("!!!!!!!!!")
  print()

explore_model = make_model(log=False)
def update_explore_from_train(train_portion=0.005):
  with torch.no_grad():
    for train_param, explore_param in zip(train_model.parameters(), explore_model.parameters()):
      explore_param.data.copy_(train_portion * train_param.data + (1 - train_portion) * explore_param.data)
update_explore_from_train(train_portion=1.0)

memory = memory_.ReplayMemory(capacity=128 * 32768)
optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-5, amsgrad=True, weight_decay=0.01)

def optimize():
  if len(memory) == 0:
    return
  OPTIMIZE_BATCH_SIZE = 8192
  for i_epoch in range(10):
    optimizer.zero_grad()

    (
      obs_bool,
      obs_int,
      obs_slot_index,
      obs_pile_present,
      target_value
    ) = memory.sample(OPTIMIZE_BATCH_SIZE)

    model_value = train_model.forward_train(
      obs_bool,
      obs_int,
      obs_slot_index,
      obs_pile_present
    )

    loss = F.mse_loss(model_value, target_value.to(torch.float32))
    loss.backward()

    optimizer.step()

  optimizer.zero_grad()

  update_explore_from_train()


def main():
  #torch.autograd.detect_anomaly(True)
  with torch.no_grad():
    N = 4096 + 2048
    n_indices = torch.arange(N, dtype=torch.long, device=gpu_device)
    M = 1024
    MAX_C = 300
    actual_max_c = 0
    c_indices = torch.arange(MAX_C, dtype=torch.long, device=gpu_device)

    card_embeds = explore_model.card_embeds

    state: ExpanseState = ExpanseState.generate_starting(
      BATCH=N,
      card_embeds=card_embeds
    )
    series = torch.zeros((N,), dtype=torch.int32, device=gpu_device)
    history: ExpanseState = (
      ExpanseState.generate_empty(card_embeds, BATCH=N)
      .view(N, 1)
      .repeat(1, MAX_C)
    )
    history_values = torch.zeros((N, MAX_C), dtype=torch.float32, device=gpu_device)
    history_length = torch.zeros((N,), dtype=torch.long, device=gpu_device)
    lambda_matrix = (
      (LAMBDA ** (c_indices.view(1, 1, MAX_C) - c_indices.view(1, MAX_C, 1)))
      * (c_indices.view(1, 1, MAX_C) >= c_indices.view(1, MAX_C, 1)).float()
    )

  i_step = STARTING_I_STEP
  i_episode = STARTING_I_EPISODE
  last_save_episode = STARTING_I_EPISODE
  while True:
    i_step += 1
    with torch.no_grad():
      print(f'step {i_step} -- episode {i_episode} -- actual max c {actual_max_c}: ', end='')

      min_series = series.min()

      for rule in phase_rules:
        if rule.get_type() == PHASE_TYPE_CHOICE:
          continue

        rm = rule.matching(state).clone()

        match_n_indices = n_indices[rm]
        match_history_length = history_length[rm]
        history.set_index((match_n_indices, match_history_length), state.index(rm))
        #assert first rule is always a choice / not in this loop
        history_values[match_n_indices, match_history_length] = history_values[match_n_indices, match_history_length-1]
        history_length[rm] += 1
        state.set_index((rm,), rule.enumerate_actions(state.index(rm), card_embeds))

      rule_matches = [rule.matching(state).clone() for rule in phase_rules]
      min_series_available = sum([(series[rm] == min_series).int().sum() for rm in rule_matches])

      state_ended = (
        (state.obs_int[:, OBS_INT_DECK_PILES] < 0)
        | (
          (state.obs_int[:, OBS_INT_DECK_PILES] == 0)
          & (state.obs_int[:, OBS_INT_DECK_PILE_SCORES] == 0)
        )
      )
      ended_n_indices = n_indices[state_ended]
      ended_lengths = history_length[state_ended]
      actual_max_c = max(actual_max_c, history_length.max(dim=0).values.item())
      history_values[ended_n_indices, ended_lengths] = (state.obs_int[:, OBS_INT_CP] - state.obs_int[:, OBS_INT_CP+1])[state_ended].float()
      ended_history_mask = (
        state_ended.view(N, 1)
        & (c_indices.view(1, MAX_C) < history_length.view(N, 1))
      )
      history_values[state_ended, :] = (
        history_values.view(N, 1, MAX_C)
        * lambda_matrix
      ).sum(dim=2)[state_ended, :]
      memory.push_all((
        history.obs_bool[ended_history_mask],
        history.obs_int[ended_history_mask],
        history.obs_slot_index[ended_history_mask],
        history.obs_pile_present[ended_history_mask],
        history_values[ended_history_mask].view(-1, 1)
      ))
      history_values[state_ended, :] = 0
      history_length[state_ended] = 0
      ended_count = state_ended.sum().item()
      i_episode += ended_count
      state.set_index((state_ended,), ExpanseState.generate_starting(ended_count, card_embeds))
      series[state_ended] += 1

      min_series_available = 0
      for (rule, rm) in zip(phase_rules, rule_matches):
        if rule.get_type() != PHASE_TYPE_CHOICE:
          print(f'.', end='')
          continue

        i_match = rm.int().cumsum(dim=0)
        match_count = i_match[-1].item()

        if (match_count < M) and not (
          min_series_available < M
          and (series[rm] == min_series).int().sum() > 0
        ):
          print(f'n', end='')
          continue
        print(f'Y', end='')

        history.set_index((n_indices[rm], history_length[rm]), state.index(rm))

        mini_batch_count = math.ceil(match_count / M)
        mini_batch_masks = (
          rm.view(1, N)
          & ((i_match // M).view(1, N) == torch.arange(mini_batch_count, dtype=torch.long, device=gpu_device).view(mini_batch_count, 1))
        )
        for i_mini_batch in range(mini_batch_count):
          mini_batch = state.index(mini_batch_masks[i_mini_batch, :])
          m = mini_batch.batch[0]

          (action_states, action_mask) = rule.enumerate_actions(mini_batch, card_embeds)
          ACTIONS = action_states.batch[1]
          #if not action_mask.any(dim=1).all():
          #  print(rule)
          #  bad_indices = action_states.batch_indices[0][action_mask.any(dim=1).logical_not()]
          #  print(bad_indices)
          #  print(mini_batch.obs_int[bad_indices, OBS_INT_CP:OBS_INT_CP+PLAYER_COUNT])
          #  assert False

          model_value = explore_model.forward_eval(
            action_states.obs_bool.view(m*ACTIONS, OBS_BOOL_LENGTH),
            action_states.obs_int.view(m*ACTIONS, OBS_INT_LENGTH),
            action_states.obs_slot_index.view(m*ACTIONS, OBS_SLOT_INDEX_COUNT),
            action_states.obs_pile_cached_embed.view(m*ACTIONS, OBS_PILE_CACHED_EMBED_COUNT, card_embeds.embedding_dim)
          ).view(m, ACTIONS)
          
          model_action_player_value = (
            model_value * 
            (
              1 
              - (
                mini_batch.obs_bool[:, OBS_BOOL_ACTION+1].to(torch.float32).view(m, 1)
                * 2
              )
            )
          )

          action_dist = torch.distributions.Categorical(
            logits=torch.where(
              action_mask,
              model_action_player_value,
              -torch.inf
            )
          )
          i_selected = action_dist.sample()

          minibatch_n_indices = n_indices[mini_batch_masks[i_mini_batch, :]]
          minibatch_lengths = history_length[mini_batch_masks[i_mini_batch, :]]
          history_values[minibatch_n_indices, minibatch_lengths] = model_value.max(dim=1).values * (1 - LAMBDA)
          history_length[mini_batch_masks[i_mini_batch, :]] += 1

          state.set_index(
            (mini_batch_masks[i_mini_batch],), 
            action_states.index(action_states.batch_indices[0], i_selected)
          )
        
      print()

    optimize()

    if i_episode >= last_save_episode + SAVE_RATE_EPISODES:
      print()
      print("Saving weights.")
      torch.cuda.empty_cache()
      torch.save(
        train_model.state_dict(), 
        os.path.join(
          WEIGHTS_DIR, 
          (
            f'{SAVE_WEIGHTS_FILENAME[:SAVE_WEIGHTS_FILENAME.rindex(".")]}'
            f'{i_step}_{i_episode}{SAVE_WEIGHTS_FILENAME[SAVE_WEIGHTS_FILENAME.rindex("."):]}'
          )
        )
      )
      torch.cuda.empty_cache()
      last_save_episode = i_episode


if __name__ == '__main__':
  thread = threading.Thread(target=main, daemon=True)
  thread.start()

  try:
    input()
    stop_go = True
    thread.join()

  except:
    print()
    print('Stopping')
    print()
    stop_go = True
    raise



