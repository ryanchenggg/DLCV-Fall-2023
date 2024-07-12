from .blender import BlenderDataset
from .llff import LLFFDataset
from .hw4_dataset import KlevrDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'klevr': KlevrDataset}