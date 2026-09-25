import abc
import importlib
import inspect
import os
import pkgutil

import game as game
import torch
import torch.nn


PHASE_TYPE_CHOICE = 0
PHASE_TYPE_DETERMINISTIC = 1
PHASE_TYPE_STOCHASTIC = 2


class PhaseRule(abc.ABC):

  @abc.abstractmethod
  def get_type(self) -> int:
    pass

  @abc.abstractmethod
  def matching(self, state: game.ExpanseState) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def enumerate_actions(self, state: game.ExpanseState, card_embeds: torch.nn.Embedding) -> tuple[game.ExpanseState, torch.Tensor]:
    pass


  def concat_state_masks(self, state_masks = list[tuple[game.ExpanseState, torch.Tensor]]):
    states = game.ExpanseState.concat([s for (s, m) in state_masks], dim=1)
    mask = torch.concat([m for (s, m) in state_masks], dim=1)

    return (states, mask)


def get_phases() -> list[PhaseRule]:
  instances = []
    
  for _, module_name, _ in pkgutil.iter_modules([os.path.abspath(".")]):
    if (
      (not module_name.startswith('phase_'))
      or (module_name == 'phase_rule')
    ):
      continue

    try:
      module = importlib.import_module(module_name)
      
      for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
          (obj.__module__ != module.__name__)
          or (not name.startswith('Phase'))
          or (name == 'PhaseRule')
          or (not issubclass(obj, PhaseRule))
        ):
          continue

        try:
          instances.append(obj())
        except:
          print(f"Could not instantiate {name}")
          raise
                
    except:
      print(f"Error processing module {module_name}")
      raise
          
  return instances
