from lib2to3.pgen2.token import OP
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
# from Param_Estimation.dataset.ego import EgoDatasetVectorized
# from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
# from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from Param_Estimation.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from Param_Estimation.planning.vectorized.open_loop_model import VectorizedModel
from Param_Estimation.map.rasterizer_builder import build_rasterizer
from l5kit.vectorization.vectorizer_builder import build_vectorizer