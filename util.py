import torch
from torch.utils.data import Dataset

class GoDataset(Dataset):
  # pos_paths: list of paths to position features
  # labels: list of moves played in a given position
  def __init__(self, pos_paths, labels):
        self.labels = labels
        self.pos_paths = pos_paths

  def __len__(self):
        return len(self.pos_paths)

  def __getitem__(self, idx):
        pos = torch.load(pos_paths[idx])
        label = self.labels[idx]

        return pos, label