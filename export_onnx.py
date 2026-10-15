import os

from game import *
import model

import torch
cpu_device = torch.device('cpu')

LOAD_WEIGHTS_PATH = os.path.join("weights", "0_centauri_7_weights___95156_2950896.pth")
EXPORT_PATH = (
  f'{LOAD_WEIGHTS_PATH[:LOAD_WEIGHTS_PATH.rindex(".")]}'
  f'.onnx'
)

make_model = (
  lambda log: (
    model.ExpanseModel_Centauri(
      hidden_lengths=[128, 128, 64, 64],
      card_embed_length=8,
      log=log
    )
  )
)
trained_model = make_model(log=True).to(cpu_device)

assert os.path.exists(LOAD_WEIGHTS_PATH), f'not found: {LOAD_WEIGHTS_PATH}'
print(f'LOADING WEIGHTS: {LOAD_WEIGHTS_PATH}')
trained_model.load_state_dict(torch.load(LOAD_WEIGHTS_PATH))
trained_model.eval()

class OnnxWrapper(torch.nn.Module):
  def __init__(self, wrapped_model: model.ExpanseModel_Centauri):
    super().__init__()
    self.wrapped_model = wrapped_model

  def forward(
    self,
    obs_bool: torch.Tensor,
    obs_int: torch.Tensor,
    obs_slot_index: torch.Tensor,
    obs_pile_present: torch.Tensor
  ):
    obs_pile_embed = obs_pile_present.to(torch.float32) @ self.wrapped_model.card_embeds.weight[:CARD_COUNT, :]

    return self.wrapped_model.forward_eval(obs_bool, obs_int, obs_slot_index, obs_pile_embed)

export_model = OnnxWrapper(trained_model)
export_model.eval()

dummy_state: ExpanseState = ExpanseState.generate_empty(trained_model.card_embeds, BATCH=1)
dummy_input = (
  dummy_state.obs_bool.to(cpu_device),
  dummy_state.obs_int.to(cpu_device),
  dummy_state.obs_slot_index.to(cpu_device),
  dummy_state.obs_pile_present.to(cpu_device)
)

print(f'Exporting to {EXPORT_PATH}')
torch.onnx.export(
  export_model,
  dummy_input,
  EXPORT_PATH,
  export_params=True,
  opset_version=14,
  input_names=['obs_bool', 'obs_int', 'obs_slot_index', 'obs_pile_present'],
  output_names=['model_value'],
  dynamic_axes={
    "obs_bool": { 0: "batch" },
    "obs_int": { 0: "batch" },
    "obs_slot_index": { 0: "batch" },
    "obs_pile_present": { 0: "batch" },
  }
)
print(f'Complete.')
