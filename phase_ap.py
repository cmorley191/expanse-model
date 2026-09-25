from phase_rule import *
from expanse_game import *


class PhaseAP(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_CHOICE

  def matching(self, state):
    return (state.obs_bool[:, OBS_BOOL_PHASE_ACTION])

  def use_ap(self, state: ExpanseState):
    state.obs_int[..., OBS_INT_PHASE_AP] -= 1
    state.obs_bool[..., OBS_BOOL_PHASE_ACTION] = state.obs_int[..., OBS_INT_PHASE_AP].bool()
    state.obs_bool[..., OBS_BOOL_PHASE_ACTION_DONE] = state.obs_bool[..., OBS_BOOL_PHASE_ACTION].logical_not()
    #assert PLAYER_COUNT == 2
    state.obs_bool[..., OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = (
      state.obs_bool[..., OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT]
      .logical_xor(state.obs_bool[..., OBS_BOOL_PHASE_ACTION_DONE].view(*state.batch, 1))
    )


  def enumerate_fleet(self, state: ExpanseState, card_embeds: torch.Tensor):
    # (batch, count, src, dest, active player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1, 1, 1)
    self.use_ap(states)

    states: ExpanseState = states.repeat(1, FLEET_COUNT, ORBITAL_COUNT, ORBITAL_COUNT, PLAYER_COUNT)
    states_fleets = states.obs_int_fleets()
    states_fleets[
      :,
      fleet_indices.view(FLEET_COUNT, 1, 1),
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      :,
      player_indices.view(1, 1, PLAYER_COUNT), 
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT)
    ] -= fleet_indices.view(FLEET_COUNT, 1, 1, 1, 1)
    states_fleets[
      :, 
      fleet_indices.view(FLEET_COUNT, 1, 1),
      :,
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT), 
      orbital_indices.view(1, ORBITAL_COUNT, 1),
      player_indices.view(1, 1, PLAYER_COUNT)
    ] += fleet_indices.view(FLEET_COUNT, 1, 1, 1, 1)
    
    state_fleets = state.obs_int_fleets()
    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, 1, 1, PLAYER_COUNT))
      # has enough fleets
      & (state_fleets.view(state.batch[0], 1, ORBITAL_COUNT, 1, PLAYER_COUNT) >= fleet_indices.view(1, FLEET_COUNT, 1, 1, 1))
      # adjacent
      & (orbital_adjacent.view(1, 1, ORBITAL_COUNT, ORBITAL_COUNT, 1))
    )

    states: ExpanseState = states.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], FLEET_COUNT * ORBITAL_COUNT * ORBITAL_COUNT * PLAYER_COUNT)

    return (states, mask)
  

  def enumerate_influence(self, state: ExpanseState, card_embeds: torch.Tensor):
    # (batch, base, player)
    states: ExpanseState = state.clone().view(state.batch[0], 1, 1)
    self.use_ap(states)

    states: ExpanseState = states.repeat(1, BASE_COUNT, PLAYER_COUNT)
    states.obs_int_influence()[
      :,
      base_indices.view(BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT),
      base_indices.view(BASE_COUNT, 1),
      player_indices.view(1, PLAYER_COUNT)
    ] += 1

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].view(state.batch[0], 1, PLAYER_COUNT))
      # has presence
      & (state.obs_int_fleets()[:, base_orbital.view(BASE_COUNT, 1), player_indices.view(1, PLAYER_COUNT)] != 0)
    )

    states: ExpanseState = states.view(state.batch[0], BASE_COUNT * PLAYER_COUNT)
    mask = mask.view(state.batch[0], BASE_COUNT * PLAYER_COUNT)

    return (states, mask)
  

  def enumerate_build(self, state: ExpanseState, card_embeds: torch.Tensor):
    # (batch, player)
    states: ExpanseState = state.clone().view(state.batch[0], 1).repeat(1, PLAYER_COUNT)
    self.use_ap(states)
    states.obs_int_fleets()[:, player_indices, player_home_orbital, player_indices] += 1

    mask = (
      # active player
      (state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT])
      # fleet in reserve
      & (state.obs_int_fleets().sum(dim=1) < FLEET_COUNT)
    )

    # enumerations already flat

    return (states, mask)


  def enumerate_pass(self, state: ExpanseState, card_embeds: torch.Tensor):
    states: ExpanseState = state.clone().view(state.batch[0], 1)
    states.obs_int[:, :, OBS_INT_PHASE_AP] = 1
    self.use_ap(states)

    mask = torch.ones(states.batch, dtype=torch.bool, device=gpu_device)

    # enumerations already flat (1)

    return (states, mask)


  def enumerate_actions(self, state, card_embeds):
    enumerations = [f(state, card_embeds) for f in [
      self.enumerate_fleet, 
      self.enumerate_influence,
      self.enumerate_build,
      self.enumerate_pass,
    ]]

    return self.concat_state_masks(enumerations)


class PhaseAPTurn_APDone(PhaseRule):

  def get_type(self):
    return PHASE_TYPE_DETERMINISTIC
  
  def matching(self, state):
    return (
      state.obs_bool[:, OBS_BOOL_PHASE_AP_TURN]
      & state.obs_bool[:, OBS_BOOL_PHASE_EVENT].logical_not()
      & state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE]
    )
  
  def enumerate_actions(self, state, card_embeds):
    new_state = state.clone()
    new_state.obs_bool[:, OBS_BOOL_PHASE_ACTION_DONE] = False
    new_state.obs_bool[:, OBS_BOOL_PHASE_CHOOSE_EVENT] = True
    new_state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT] = new_state.obs_bool[:, OBS_BOOL_ACTION:OBS_BOOL_ACTION+PLAYER_COUNT].logical_not()

    return new_state

