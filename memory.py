from game import *


MEMORY_OBS_BOOL = 0
MEMORY_OBS_INT = 1
MEMORY_OBS_SLOT_INDEX = 2
MEMORY_OBS_PILE_PRESENT = 3
MEMORY_TARGET_VALUE = 4
MEMORY_COUNT = 5

MEMORY_SHAPES_DTYPES = [
  ((OBS_BOOL_LENGTH,), torch.bool),
  ((OBS_INT_LENGTH,), torch.int8),
  ((OBS_SLOT_INDEX_COUNT,), torch.float32),
  ((OBS_PILE_PRESENT_COUNT, CARD_COUNT), torch.float32),
  ((1,), torch.int8),
]

class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.reset()

  def __memory_blocks(shape):
    return tuple([
      torch.zeros((*shape, *block_shape), dtype=block_dtype, device=gpu_device)
      for (block_shape, block_dtype) in MEMORY_SHAPES_DTYPES
    ])

  def reset(self):
    self.data = None
    self.data = ReplayMemory.__memory_blocks((self.capacity,))
    self.tail = 0
    self.size = 0

  def __len__(self):
    return self.size

  def push_all(self, data):
    push_count = min(self.capacity, data[0].shape[0])
    i_start = 0
    if push_count != data[0].shape[0]:
      print()
      print(f"Warning: push_all dropped some data. Increase memory capacity or reduce data per push.")
      i_start = data[0].shape[0] - push_count
    self.size = min(self.capacity, self.size + push_count)
    end = self.tail + push_count
    if end > self.capacity:
      portion = self.capacity - self.tail
      for (self_block, block) in zip(self.data, data):
        self_block[self.tail:self.capacity, ...] = block[i_start:i_start+portion, ...]
      end %= self.capacity
      for (self_block, block) in zip(self.data, data):
        self_block[:end, ...] = block[i_start+portion:, ...]
      self.tail = end
    else:
      for (self_block, block) in zip(self.data, data):
        self_block[self.tail:end, :] = block
      self.tail = end % self.capacity

  def sample(self, sample_size):
    i_samples = torch.randperm(self.size, device=gpu_device)[:sample_size]
    return tuple([
      block[i_samples, ...]
      for block in self.data
    ])
  
