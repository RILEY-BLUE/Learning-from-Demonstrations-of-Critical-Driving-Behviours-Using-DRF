import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
#from l5kit.dataset import EgoDataset
#from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points, angular_distance, transform_point
from l5kit.visualization import TARGET_POINTS_COLOR, PREDICTED_POINTS_COLOR, draw_trajectory
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

from Param_Estimation.map.map_builder import MapBuilder
from Param_Estimation.driver.DRFModel import DRFModel
from Param_Estimation.map.rasterizer_builder import build_rasterizer
from Param_Estimation.dataset.ego import EgoDataset

import os

from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize, spatial

from Param_Estimation.driver.DRFModel import DRFModel

#### for testing heeds inner only!
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "D:\YURUIDU\DRF_Simulation\Param_Estimation\Param_Estimation\planning"
dm = LocalDataManager(None)
# get config
cfg = load_config_data(os.path.join(os.environ["L5KIT_DATA_FOLDER"], "config.yaml"))

scene = int(scene_idx) # HEEDS does not give int, thus conversion to int is necessary

# ===== INIT DATASET
eval_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
dataset = eval_dataset
#### for testing heeds inner only!

# driver controller here (from paper)
# risk_threshold = 9000
k_h = 0.02 # gain of default heading P-controller
k_v = 0.025 # gain of vehicle's speed-up/down, could be different for normal or sport driving
k_vc = 1.5 * 1e-4 # gain of vehicle's speed-down contributed by the perceived risk (cost)
dt = 0.1 # [s] step time
dvMax = 4 * dt # [m/s^2] Vehicle max decel and acceleration
dvstepMax = 10 * dt # For fminbnd search
dsMax = np.pi / 180 * 1 * dt # [rad/dt] Note: Here assume steer limit is 10 degree/s, be careful with unit!
dstepMax = np.pi / 180 * 50 * dt # [rad/dt] Note: For fminbound search

def driverController(curr_road_heading: float, ego_curr_heading_world: float, curr_risk: float, ego: DRFModel, v_des: float) -> Tuple[float, float]:
    desired_vel = v_des
    curr_vel = ego.v
    ego_heading = ego.phiv # assume ego's ground truth heading is the current road heading
    curr_steering = ego.delta
    next_steering =  curr_steering + k_h * (curr_road_heading - ego_curr_heading_world)
    if (curr_risk <= risk_threshold and curr_vel <= desired_vel):
        # condition 1
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        # steering update
        ds = np.fmin(np.abs(next_steering - curr_steering), dsMax)
        next_steering = curr_steering + np.sign(next_steering - curr_steering) * ds
        #print("condiion1 vel = ", next_vel)
        return next_steering, next_vel
    elif (curr_risk > risk_threshold and curr_vel <= desired_vel):
        # check if changing angular velocity can reduce the cost below the threshold
        opt_s, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeSteering, x1=curr_steering - dstepMax, 
                                                                x2=curr_steering + dstepMax, full_output=True)
        if (minCost > risk_threshold):
            # condition 2a
            # steering angle update
            ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
            next_steering = curr_steering + np.sign(opt_s - curr_steering) * ds
            # velocity update
            dv = np.fmin(dvMax, np.abs(k_vc * (risk_threshold - minCost)))
            next_vel = curr_vel + np.sign(risk_threshold - minCost) * dv
            #print("condiion2a vel = ", next_vel)
            return next_steering, next_vel
            
#             model slows down
#             proportional to Cop − Ck (and not Cop − Ct) since the steering applied = wop is
#             expected to reduce Ck to Cop. This is done so that we do not slow down more than
#             what is required. Hence, w_k+1 = wop 
        elif (minCost <= risk_threshold):
            # condition 2b
            # velocity update
            dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
            next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
            # steering update
            next_steering = curr_steering + k_h * (curr_road_heading - ego_heading)
            opt_s, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeSteeringCt, x1=curr_steering - dstepMax, 
                                                            x2=curr_steering + dstepMax, full_output=True)

            ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
            next_steering = curr_steering + np.sign(opt_s - curr_steering) * ds
            #print("condiion2b vel = ", next_vel)
            return next_steering, next_vel

        else:
            print("Error in stage: condition 2")
        
        # /* In this case the model slows down, while being
        # ** steered by the heading controller since the risk is lower than the threshold and
        # ** speed is higher than what is desired.
        # */
    elif (curr_risk <= risk_threshold and curr_vel > desired_vel):
        # condition 3
        # steering update
        ds = np.fmin(np.abs(next_steering - curr_steering), dsMax)
        next_steering = curr_steering + np.sign(next_steering - curr_steering) * ds
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        #print("condiion3 vel = ", next_vel)
        return next_steering, next_vel
        
        # /* In this case both the speed and risk are over
        # ** the desired limits and hence the model slows down while steering with δop that
        # ** minimises Ck
        # */
    elif (curr_risk > risk_threshold and curr_vel > desired_vel):
        # condition 4
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        # steering update
        opt_s, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeSteeringCt, x1=curr_steering - dstepMax, 
                                                            x2=curr_steering + dstepMax, full_output=True)

        ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
        next_steering = curr_steering + np.sign(opt_s - curr_steering) * ds
        #print("condiion4 vel = ", next_vel)
        return next_steering, next_vel

    else: 
        print("Error at driver controller: no situations match!")

def driverController2(curr_road_heading: float, ego_curr_heading_world: float, curr_risk: float, ego: DRFModel, v_des: float) -> Tuple[float, float]:
    desired_vel = v_des
    curr_vel = ego.v
    curr_steering = ego.delta
    next_steering =  curr_steering #+ k_h * (curr_road_heading - ego_curr_heading_world)
    if (curr_risk <= risk_threshold and curr_vel <= desired_vel):
        # condition 1
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        # steering update
#         ds = np.fmin(np.abs(next_steering - curr_steering), dsMax)
#         next_steering = curr_steering + np.sign(next_steering - curr_steering) * ds
        #print("condition1 vel = ", next_vel)
        return next_steering, next_vel
    elif (curr_risk > risk_threshold and curr_vel <= desired_vel):
        # check if changing velocity alone can reduce the cost below the threshold
        opt_v, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeVelocity, x1=curr_vel - dvstepMax, 
                                                                x2=curr_vel + dvstepMax, full_output=True)
        if (minCost > risk_threshold):
            # condition 2a
            # steering angle update
#             ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
#             next_steering = curr_steering + np.sign(opt_s - curr_steering) * ds
            # velocity update
            dv = np.fmin(dvMax, np.abs(opt_v - curr_vel)) 
            next_vel = curr_vel + np.sign(opt_v - curr_vel) * dv
            #print("condition2a vel = ", next_vel)
            return next_steering, next_vel
            
#             model slows down
#             proportional to Cop − Ck (and not Cop − Ct) since the steering applied = wop is
#             expected to reduce Ck to Cop. This is done so that we do not slow down more than
#             what is required. Hence, w_k+1 = wop 
        elif (minCost <= risk_threshold):
            # condition 2b
            # velocity update
            opt_v, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeVelocityCt, x1=curr_vel - dvstepMax, 
                                                               x2=curr_vel + dvstepMax, full_output=True)
            dv = np.fmin(dvMax, np.abs(opt_v - curr_vel))
            next_vel = curr_vel + np.sign(opt_v - curr_vel) * dv
            # steering update
#             opt_s, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeSteeringCt, x1=curr_steering - dstepMax, 
#                                                             x2=curr_steering + dstepMax, full_output=True)

#             ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
#             next_steering = curr_steering + np.sign(opt_s - curr_steering) * ds
            #print("condition2b vel = ", next_vel)
            return next_steering, next_vel

        else:
            print("Error in stage: condition 2")
        
        # /* In this case the model slows down, while being
        # ** steered by the heading controller since the risk is lower than the threshold and
        # ** speed is higher than what is desired.
        # */
    elif (curr_risk <= risk_threshold and curr_vel > desired_vel):
        # condition 3
        # steering update
        #ds = np.fmin(np.abs(next_steering - curr_steering), dsMax)
#         next_steering = curr_steering #+ np.sign(next_steering - curr_steering) * ds
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        #print("condition3 vel = ", next_vel)
        return next_steering, next_vel
        
        # /* In this case both the speed and risk are over
        # ** the desired limits and hence the model slows down while steering with δop that
        # ** minimises Ck
        # */
    elif (curr_risk > risk_threshold and curr_vel > desired_vel):
        # condition 4
        # velocity update
        opt_v, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeVelocity, x1=curr_vel - dvstepMax, 
                                                                x2=curr_vel + dvstepMax, full_output=True)
        dv = np.fmin(dvMax, np.abs(opt_v - curr_vel) + k_v * (desired_vel - curr_vel))
       
        next_vel = curr_vel + np.sign(opt_v - curr_vel) * dv
        # steering update
#         opt_s, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeSteeringCt, x1=curr_steering - dstepMax, 
#                                                             x2=curr_steering + dstepMax, full_output=True)

        #ds = np.fmin(np.abs(opt_s - curr_steering), dsMax)
#         next_steering = curr_steering #+ np.sign(opt_s - curr_steering) * ds
        #print("condition4 vel = ", next_vel)
        return next_steering, next_vel

    else: 
        print("Error at driver controller: no situations match!")
    print("Next steering = ", next_steering)
    print("Next Velocity = ", next_vel)

# Compute the error in position between gt and pred
def computePosError(preds: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, float]:
    error_arr = np.array(gts) - np.array(preds)
    error_x = error_arr[:, 0]
    error_y = error_arr[:, 1]
    error_arr = np.sqrt(error_x**2 + error_y**2)
    mean_error_pos = np.mean(error_arr)
    return error_arr, mean_error_pos

def dist2Reference(preds: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, float]:
    
    # This solution is optimal when xy2 is very large
    tree = spatial.cKDTree(gts)
    mindist, minid = tree.query(preds)
    mean_error_lat = np.mean(mindist)
    return mindist, mean_error_lat

def computeYawError(preds: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, float]:
    yaw_preds = np.ravel(np.array(preds))
    yaw_gts = np.ravel(np.array(gts))
    error_yaws = yaw_gts - yaw_preds
    mean_error_yaw = np.mean(np.abs(error_yaws))
    return error_yaws, mean_error_yaw

def computeVelError(preds: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, float]:
    yaw_preds = np.ravel(np.array(preds))
    error_vels = yaw_gts - yaw_preds
    mean_error_vel = np.mean(np.abs(error_vels))
    return error_vels, mean_error_vel

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def filterPosition(ego_pos: np.ndarray, k: float, b: float):
    m = ego_pos[0]
    n = ego_pos[1]
    ego_pos[0] = (n + m / k - b) / (k + 1 / k)
    ego_pos[1] = k * ego_pos[0] + b
    return ego_pos


# ===== Parameter Estimation (closed-loop)

""" Pseudo code
for each scene:
    from the first frame:
        read x_0, y_0, phi_0
        assume delta_0 = 0
        by frame1 - frame 0, get v_0
    for each frame:
        obj_map -> DRF -> next x, y, phi, delta, v
        record x, y
        record ground truth

Compare gts to preds
"""

position_preds = []
yaw_preds = []
position_preds_world = []
yaw_preds_world = []
vel_preds = []

position_gts = []
yaw_gts = []
position_gts_world = []
yaw_gts_world = []
vel_gts = []


# number of scenes depends on the size of the dataset, here we use sample.zarr
indexes = dataset.get_scene_indices(scene) # set scene index here!

# Initialisation
# Note: Accessing pose from Egodataset seems to have agent(ego-as-agent) coordinates
first_frame_idx = indexes[0]
first_frame = dataset[first_frame_idx]

egoDRF = DRFModel(p=p, t_la=t_la, c=0.4, m=m, k_1=k_1, k_2=k_2, safe_distance=safe_distance, cost_threshold=risk_threshold)

# world frame initial pose of ego vehicle
x0 = first_frame["centroid"][0]
y0 = first_frame["centroid"][1]
phiv0 = first_frame["yaw"] 

first_heading = first_frame["yaw"] 
ego_curr_heading_world = first_frame["yaw"] 
ego_position_world = first_frame["centroid"]

# TODO: Assume at the start of each scene, the vehicle's heading is the road heading
# Need to implement a method to compute the road heading of the ego vehicle as the road heading can change through the scene

# compute gt velocity, record ground truth velocity, yaw and position
for idx in indexes[:-1]: # all frames of each scene
    # Note: Accessing pose from Egodataset seems to have agent(ego-as-agent) coordinates
    curr_frame = dataset[idx]
    curr_road_heading = curr_frame["yaw"]
    next_frame = dataset[idx + 1]
    
    # compute gt velocity
    x_curr = curr_frame["centroid"][0]
    y_curr = curr_frame["centroid"][1]
    x_next = next_frame["centroid"][0]
    y_next = next_frame["centroid"][1]
    
    position_gts_world.append(curr_frame["centroid"])
    diff_pos_in_world = np.sqrt((x_next - x_curr)**2 + (y_next - y_curr)**2)
    gt_vel = diff_pos_in_world / 0.1 # dt = 0.1s
    vel_gts.append(gt_vel)
    yaw_gts.append(curr_frame["target_yaws"][0])
    yaw_gts_world.append(curr_frame["yaw"])

road_heading = np.mean(np.array(yaw_gts_world))

f_vel = 13.5 #np.max(smooth(np.array(vel_gts), 10))
egoDRF.v = smooth(np.array(vel_gts), 3)[0] # starting with initial speed

# Mitigate lateral deviation from center line due to untrackable lane headings
pworld = np.array(position_gts_world)
px = pworld[:, 0]
py = pworld[:, 1]
coef = np.polyfit(px, py, 1)
k1 = coef[0]
b1 = coef[1]
# Mitigate lateral deviation from center line due to untrackable lane headings

for idx in indexes[:-1]: # all frames of each scene
    # Note: Accessing pose from Egodataset seems to have agent(ego-as-agent) coordinates
    curr_frame = dataset[idx]
    curr_road_heading = curr_frame["yaw"]
    next_frame = dataset[idx + 1]
    # ego_curr_heading_world += egoDRF.phiv
    ego_curr_heading_world = curr_road_heading

    egoDRF.obj_map = dataset.get_image_from_position(frame_idx=idx, position=ego_position_world, yaw=ego_curr_heading_world)
#         print(ego_curr_heading_world)
#         # for debugging
#         p_im = egoDRF.obj_map
#         p_im = plt.imshow(p_im, animated=True)
#         p_ims.append([p_im])
#         # for debugging
#         plt.imshow(egoDRF.obj_map)
#         plt.show() 
    
    egoDRF.x = 50 # or first_frame["centroid"][:2]? (world frame)
    egoDRF.y = 50
    egoDRF.phiv = 0.
    
    p_risk = egoDRF.overallProcess()
    
#         # for debugging
#         sub_map = egoDRF.zOfGaussian
#         p_im = sub_map
#         p_im = plt.imshow(p_im, animated=True)
#         p_ims.append([p_im])
#         # for debugging
#         plt.imshow(egoDRF.zOfGaussian)
#         plt.show() 
    
    egoDRF.delta, egoDRF.v = driverController2(curr_road_heading, ego_curr_heading_world, p_risk, egoDRF, f_vel)
    egoDRF.carKinematics()

    # Record groundtruth path
    position_gts.append(curr_frame["target_positions"][0])
     
    # Record actual path
    # All driver controller computation is done in the raster frame, change prediction to agent frame for evaluation
    pose_in_world = np.array(
        [
            [np.cos(ego_curr_heading_world), -np.sin(ego_curr_heading_world), ego_position_world[0]],
            [np.sin(ego_curr_heading_world), np.cos(ego_curr_heading_world), ego_position_world[1]],
            [0, 0, 1],
        ]
    )
    
    raster_from_world = dataset.rasterizer.render_context.raster_from_local @ np.linalg.inv(pose_in_world)
    ego_position_world = transform_point(np.reshape(np.array([egoDRF.x, egoDRF.y]), 2), np.linalg.inv(raster_from_world))
    ego_position_world = filterPosition(ego_position_world, k1, b1)

    position_preds_world.append(ego_position_world)
    yaw_preds.append(egoDRF.phiv)
    yaw_preds_world.append(ego_curr_heading_world)
    vel_preds.append(egoDRF.v)
  
error_arr, mean_error_pos = computePosError(position_preds_world, position_gts_world) ### TODO!
# vel_error_arr, mean_error_vel = computeVelError(vel_preds, vel_gts)
# lat_error_arr, mean_error_lat = dist2Reference(position_preds_world, position_gts_world)
yaw_error_arr, mean_error_yaw = computeYawError(yaw_preds_world, yaw_gts_world)
