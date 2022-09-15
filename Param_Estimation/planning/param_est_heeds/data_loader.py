# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
# from tqdm import tqdm

# from l5kit.configs import load_config_data
# from l5kit.data import LocalDataManager, ChunkedDataset
# from l5kit.dataset import AgentDataset
# #from l5kit.dataset import EgoDataset
# #from l5kit.rasterization import build_rasterizer
# from l5kit.geometry import transform_points, angular_distance, transform_point
# from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
# from l5kit.kinematic import AckermanPerturbation
# from l5kit.random import GaussianRandomGenerator

# from Param_Estimation.map.map_builder import MapBuilder
# from Param_Estimation.driver.DRFModel import DRFModel
# from Param_Estimation.map.rasterizer_builder import build_rasterizer
# from Param_Estimation.dataset.ego import EgoDataset

# import os

# from typing import Dict, List, Optional, Tuple, Union
# from scipy import optimize
# from Param_Estimation.driver.DRFModel import DRFModel


# # set env variable for data
# os.environ["L5KIT_DATA_FOLDER"] = "D:\YURUIDU\DRF_Simulation\Param_Estimation\Param_Estimation\planning"
# dm = LocalDataManager(None)
# # get config
# cfg = load_config_data(os.path.join(os.environ["L5KIT_DATA_FOLDER"], "config.yaml"))


# # ===== INIT DATASET
# eval_cfg = cfg["val_data_loader"]
# rasterizer = build_rasterizer(cfg, dm)
# eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
# eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
# eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
#                              num_workers=eval_cfg["num_workers"])

# # Select lane-keeping scenes
# valid_scene_indices = []
# vels = []
# yaws = []
# dataset = eval_dataset

# for scene_idx in range(100): 
#     indexes = dataset.get_scene_indices(scene_idx)
#     for idx in indexes[:-1]: # throughout each scene
#         # Note: Accessing pose from Egodataset seems to have agent(ego-as-agent) coordinates
#         curr_frame = dataset[idx]
#         curr_road_heading = curr_frame["yaw"]
#         next_frame = dataset[idx + 1]
#         yaws.append(curr_road_heading)
        
#         # compute gt velocity
#         x_curr = curr_frame["centroid"][0]
#         y_curr = curr_frame["centroid"][1]
#         x_next = next_frame["centroid"][0]
#         y_next = next_frame["centroid"][1]
        
#         diff_pos_in_world = np.sqrt((x_next - x_curr)**2 + (y_next - y_curr)**2)
#         gt_vel = diff_pos_in_world / 0.1 # dt = 0.1s
#         vels.append(gt_vel)
    
#     vels = np.array(vels)
#     yaws = np.array(yaws)
#     accs = np.diff(vels) / 0.1 # acc = dv / dt
#     decs = accs[accs < 0]
#     min_decs = decs[decs.argsort()[:10]]

#     if (np.min(vels) > 1.0 and np.mean(vels) > 5 and np.mean(min_decs) > -5 and np.max(yaws) - np.min(yaws) < 15 / 180 * np.pi):
#         valid_scene_indices.append(scene_idx)
#     vels = []
#     yaws = []

valid_scene_indices = [1]
