import imp
import copy
from typing import Dict, List, Tuple, Optional
from scipy.spatial import KDTree

import numpy as np

from l5kit.data import AGENT_DTYPE, FRAME_DTYPE, TL_FACE_DTYPE, PERCEPTION_LABEL_TO_INDEX
from l5kit.data import filter_agents_by_distance, filter_agents_by_labels, filter_tl_faces_by_status
from l5kit.data.filter import filter_agents_by_track_id, get_other_agents_ids
from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.geometry.transform import transform_points, transform_point, yaw_as_rotation33, rotation33_as_yaw
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
# from l5kit.sampling.agent_sampling import get_relative_poses
from Param_Estimation.sampling.agent_sampling import _get_relative_poses

from Param_Estimation.kinematic.ackerman_perturbation import _get_trajectory
from Param_Estimation.driver.DRFModel import DRFModel
from Param_Estimation.driver.DRFController import risk_threshold_controller
import torch

from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext


class Vectorizer:
    """Object that processes parts of an input frame, and converts this frame to a vectorized representation - which
    can e.g. be fed as input to a DNN using the corresponding input format.

    """

    def __init__(self, cfg: dict, mapAPI: MapAPI):
        """Instantiates the class.

        Arguments:
            cfg: config to load settings from
            mapAPI: mapAPI to query map information
        """
        self.lane_cfg_params = cfg["data_generation_params"]["lane_params"]
        self.mapAPI = mapAPI
        self.max_agents_distance = cfg["data_generation_params"]["max_agents_distance"]
        self.history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
        self.future_num_frames = cfg["model_params"]["future_num_frames"]
        self.history_num_frames_max = max(cfg["model_params"]["history_num_frames_ego"], self.history_num_frames_agents)
        self.other_agents_num = cfg["data_generation_params"]["other_agents_num"]
        self.perturb_agents = False

    def vectorize(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray, agent_yaw_rad: float,
                  agent_from_world: np.ndarray, history_frames: np.ndarray, history_agents: List[np.ndarray],
                  history_tl_faces: List[np.ndarray], future_tl_faces: List[np.ndarray], history_position_m: np.ndarray, 
                  history_yaws_rad: np.ndarray, history_availability: np.ndarray, future_frames: np.ndarray, 
                  future_agents: List[np.ndarray], future_vels_mps: np.ndarray, rasterizer: Rasterizer, 
                  lag_agent: Optional[List[np.ndarray]] = None, lead_agent: Optional[List[np.ndarray]] = None, 
                  DRF_perturbation: Optional[bool] = False) -> dict:
        """Base function to execute a vectorization process.

        Arguments:
            selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
            predict the future of.
            This agent is centered in the representation and the returned targets are derived from their future states.
            agent_centroid_m: position of the target agent
            agent_yaw_rad: yaw angle of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_frames: historical frames of the target frame
            history_agents: agents appearing in history_frames
            history_tl_faces: traffic light faces in history frames
            history_position_m: historical positions of target agent
            history_yaws_rad: historical yaws of target agent
            history_availability: availability mask of history frames
            future_frames: future frames of the target frame
            future_agents: agents in future_frames
            future_position_m: future positions of the target agent
            lag_agents: agents behind the ego vehicle (for DRF perturbation only)

        Returns:
            dict: a dict containing the vectorized frame representation
        """
        agent_features = self._vectorize_agents(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                                history_frames, history_agents, history_tl_faces, history_position_m, 
                                                history_yaws_rad, history_availability, future_frames, future_agents, 
                                                future_tl_faces, future_vels_mps, rasterizer, lag_agent, 
                                                lead_agent, DRF_perturbation)
        map_features = self._vectorize_map(agent_centroid_m, agent_from_world, history_tl_faces)
        return {**agent_features, **map_features}

    def _vectorize_agents(self, selected_track_id: Optional[int], agent_centroid_m: np.ndarray,
                          agent_yaw_rad: float, agent_from_world: np.ndarray, history_frames: np.ndarray,
                          history_agents: List[np.ndarray], history_tl_faces: List[np.ndarray], history_position_m: np.ndarray,
                          history_yaws_rad: np.ndarray, history_availability: np.ndarray, future_frames: np.ndarray,
                          future_agents: List[np.ndarray], future_tl_faces: List[np.ndarray], future_vels_mps: np.ndarray, 
                          rasterizer: Rasterizer, lag_agent: Optional[List[np.ndarray]] = None, lead_agent: Optional[List[np.ndarray]] = None, 
                          DRF_perturbation: Optional[bool] = False) -> dict:
        """Vectorize agents in a frame.

        Arguments:
            selected_track_id: selected_track_id: Either None for AV, or the ID of an agent that you want to
            predict the future of.
            This agent is centered in the representation and the returned targets are derived from their future states.
            agent_centroid_m: position of the target agent
            agent_yaw_rad: yaw angle of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_frames: historical frames of the target frame
            history_agents: agents appearing in history_frames
            history_tl_faces: traffic light faces in history frames
            history_position_m: historical positions of target agent
            history_yaws_rad: historical yaws of target agent
            history_availability: availability mask of history frames
            future_frames: future frames of the target frame
            future_agents: agents in future_frames
            future_tl_faces: traffic light faces in future frames
            future_position_m: future positions of target agent
            rasterizer: 
            lag_agents: agents behind the ego vehicle (for DRF perturbation only)

        Returns:
            dict: a dict containing the vectorized agent representation of the target frame
        """
        # compute agent features
        # sequence_length x 2 (two being x, y)
        agent_points = history_position_m.copy()
        # sequence_length x 1
        agent_yaws = history_yaws_rad.copy()
        # sequence_length x xy+yaw (3)
        agent_trajectory_polyline = np.concatenate([agent_points, agent_yaws], axis=-1)
        agent_polyline_availability = history_availability.copy()

        # get agents around AoI sorted by distance in a given radius. Give priority to agents in the current time step
        history_agents_flat = filter_agents_by_labels(np.concatenate(history_agents))
        history_agents_flat = filter_agents_by_distance(history_agents_flat, agent_centroid_m, self.max_agents_distance)

        cur_agents = filter_agents_by_labels(history_agents[0])
        cur_agents = filter_agents_by_distance(cur_agents, agent_centroid_m, self.max_agents_distance)

        list_agents_to_take = get_other_agents_ids(
            history_agents_flat["track_id"], cur_agents["track_id"], selected_track_id, self.other_agents_num
        )

        # Loop to grab history and future for all other agents
        all_other_agents_history_positions = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_yaws = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 1), dtype=np.float32)
        all_other_agents_history_yaws_raw = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 1), dtype=np.float32)
        all_other_agents_history_extents = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1, 2), dtype=np.float32)
        all_other_agents_history_availability = np.zeros(
            (self.other_agents_num, self.history_num_frames_max + 1), dtype=np.float32)
        all_other_agents_types = np.zeros((self.other_agents_num,), dtype=np.int64)

        all_other_agents_future_positions = np.zeros(
            (self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_yaws = np.zeros((self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_yaws_raw = np.zeros((self.other_agents_num, self.future_num_frames, 1), dtype=np.float32)
        all_other_agents_future_extents = np.zeros((self.other_agents_num, self.future_num_frames, 2), dtype=np.float32)
        all_other_agents_future_availability = np.zeros(
            (self.other_agents_num, self.future_num_frames), dtype=np.float32)

        ## Read the GT trajectories
        # ego
        (
            original_traj_ego_t0, # ego_future_coords_offset
            original_yaw_ego_t0, # ego_future_yaws_offset
            original_yaw_ego_world, # ego_future_yaws_raw
            ego_future_extent,
            ego_future_availability,
        ) = _get_relative_poses(self.future_num_frames, future_frames, future_agents, 
                                agent_from_world, agent_yaw_rad, selected_track_id=None)

        # agents
        for idx, track_id in enumerate(list_agents_to_take):
            (
                agent_history_coords_offset,
                agent_history_yaws_offset,
                agent_history_yaws_raw,
                agent_history_extent,
                agent_history_availability,
            ) = _get_relative_poses(self.history_num_frames_max + 1, history_frames, history_agents,
                                   agent_from_world, agent_yaw_rad, track_id)

            all_other_agents_history_positions[idx] = agent_history_coords_offset
            all_other_agents_history_yaws[idx] = agent_history_yaws_offset
            all_other_agents_history_yaws_raw[idx] = agent_history_yaws_raw
            all_other_agents_history_extents[idx] = agent_history_extent
            all_other_agents_history_availability[idx] = agent_history_availability
            # NOTE (@lberg): assumption is that an agent doesn't change class (seems reasonable)
            # We look from history backward and choose the most recent time the track_id was available.
            current_other_actor = filter_agents_by_track_id(history_agents_flat, track_id)[0]
            all_other_agents_types[idx] = np.argmax(current_other_actor["label_probabilities"])

            (
                agent_future_coords_offset,
                agent_future_yaws_offset,
                agent_future_yaws_raw,
                agent_future_extent,
                agent_future_availability,
            ) = _get_relative_poses(
                self.future_num_frames, future_frames, future_agents, agent_from_world, agent_yaw_rad, track_id
            )
            all_other_agents_future_positions[idx] = agent_future_coords_offset
            all_other_agents_future_yaws[idx] = agent_future_yaws_offset
            all_other_agents_future_yaws_raw[idx] = agent_future_yaws_raw
            all_other_agents_future_extents[idx] = agent_future_extent
            all_other_agents_future_availability[idx] = agent_future_availability

        if DRF_perturbation: # if DRF perturbation is required
            all_other_agents_future_positions_perturbed = copy.deepcopy(all_other_agents_future_positions)
            all_other_agents_future_yaws_perturbed = copy.deepcopy(all_other_agents_future_yaws)

            # NOTE: We only consider the 1 lead and 1 lag cars closest to the ego
            if lag_agent is not None:
                lag_agent_track_id = lag_agent["track_id"]
            if lead_agent is not None:
                lead_agent_track_id = lead_agent["track_id"]

            interpolated_traj_ego_t0, interpolated_yaw_ego_t0 = self.get_linear_interpolated_trajectory(original_traj_ego_t0, original_yaw_ego_t0)
            original_traj_ego_world = transform_points(original_traj_ego_t0, np.linalg.inv(agent_from_world))
            interpolated_traj_ego_world, interpolated_yaw_ego_world = self.get_linear_interpolated_trajectory(original_traj_ego_world, 
                                                                        original_yaw_ego_world)
            
            original_vel_ego = np.linalg.norm(future_vels_mps, axis=1)
            original_traj_ego_t0_perturbed = copy.deepcopy(original_traj_ego_t0)
            original_yaws_ego_t0_perturbed = copy.deepcopy(original_yaw_ego_t0)
            original_vel_ego_perturbed = copy.deepcopy(original_vel_ego) # for recording ego vel along perturbation

            original_traj_lag_t0_perturbed = np.zeros_like(original_traj_ego_t0_perturbed)
            original_traj_lag_t0 = np.zeros_like(original_traj_ego_t0_perturbed) # For trajectory check only
        ## Read the GT trajectories

        ## Perturbed the GT trajevtories

        # DRF perturbation required AND future frames are enough for perturbation
        if (lag_agent is not None) and DRF_perturbation and len(future_frames) == self.future_num_frames: 
            # First perturbed frame: for initialization
            perturbed_idx = 0
            curr_frame = history_frames[0]
            ego_agent_pos = curr_frame["ego_translation"][:2]
            ego_agent_yaw = rotation33_as_yaw(curr_frame["ego_rotation"])
            ego_agent_vel = original_vel_ego_perturbed[0] # deliberatly make initial speed large

            # lag agent exists
            if lag_agent is not None and len(filter_agents_by_track_id(cur_agents, lag_agent_track_id)) == 1: 
                cur_lag_agent = filter_agents_by_track_id(filter_agents_by_labels(cur_agents),
                                                           lag_agent_track_id)[0]
                lag_position = cur_lag_agent["centroid"]
                lag_distance = np.linalg.norm(ego_agent_pos - lag_position)
                lag_velocity = np.linalg.norm(cur_lag_agent["velocity"])
            else:
                lag_distance = 30 # No lag agent in the raster, 50m is half the raster length
                lag_velocity = 1e-8

            # Perturb ego
            index, world_from_raster = self.get_recommended_index(ego_agent_pos, ego_agent_yaw, ego_agent_vel, rasterizer, 
                                                                  history_frames, history_agents, history_tl_faces, interpolated_traj_ego_world,
                                                                  lag_distance=lag_distance
                                                                  )
            prev_perturbed_index = -1 # First step, no previous perturbed index

            # Read lead/lag agent's positions
            if lead_agent is not None and len(filter_agents_by_track_id(cur_agents, lead_agent_track_id)) == 1: # lead agent exists and included in list_agents_to_take
                cur_lead_agent = filter_agents_by_track_id(filter_agents_by_labels(cur_agents),
                                                           lead_agent_track_id)[0]
                lead_position = cur_lead_agent["centroid"]
                lead_distance = np.linalg.norm(ego_agent_pos - lead_position)
                lead_velocity = np.linalg.norm(cur_lead_agent["velocity"])
            else:
                lead_position_raster = np.array([80, 50]) # No lead agent in the raster, 50m is half the raster length
                lead_position = transform_point(lead_position_raster, world_from_raster)
                lead_distance = 30
                lead_velocity = 15

            index, world_from_raster = self.get_recommended_index_P_controller(ego_agent_pos, ego_agent_yaw, ego_agent_vel, 
                                                rasterizer, lead_distance, lead_velocity, lag_distance, lag_velocity, 
                                                interpolated_traj_ego_world)
            # maintaining safety distance to front vehicle 
            ego2lead_dist = np.linalg.norm(lead_position - interpolated_traj_ego_world, axis=1) 
            dist_minus_SD = ego2lead_dist - 5 # assume SD = 5m
            min_dist_index = np.argmin(dist_minus_SD)

            # Every index between prev_perturbed_index and SD_index should be "safe" perturbed points
            safe_perturbed_dist = dist_minus_SD[:min_dist_index + 1]
            SDist = safe_perturbed_dist[safe_perturbed_dist > 0]
            if SDist.size == 0: # agent too close too its leading car
                SD_index = perturbed_idx # make SD_index = GT_index
            else:
                SD_index = np.argmin(SDist)

            if (SD_index >= 0):
                # SD_index exists
                index = self.perturb_trajectory_rules(0, index, prev_perturbed_index, SD_index)
            else:
                index = 0

            # For perturbing next frame
            selected_next_pos_world = interpolated_traj_ego_world[index]
            selected_next_yaw_world = interpolated_yaw_ego_world[index]

            # record the perturbed trajectory

            # Prepare next frame for update
            perturbed_frame = np.zeros(1, dtype=FRAME_DTYPE)
            next_frame = future_frames[perturbed_idx]

            perturbed_frame["timestamp"] = next_frame["timestamp"]
            perturbed_frame["agent_index_interval"] = next_frame["agent_index_interval"]
            perturbed_frame["traffic_light_faces_index_interval"] = next_frame["traffic_light_faces_index_interval"]
            perturbed_frame["ego_translation"] = next_frame["ego_translation"]
            perturbed_frame["ego_translation"][:, :2] = selected_next_pos_world
            perturbed_frame["ego_rotation"] = yaw_as_rotation33(selected_next_yaw_world)
            
            #self.insert_perturbed_ego(future_frames[perturbed_idx], selected_next_pos_world, selected_next_yaw_world)
            self.insert_perturbed_ego_(perturbed_idx, future_frames, perturbed_frame)

            # For returning the perturbed trajectory
            selected_next_pos_t0 = interpolated_traj_ego_t0[index]
            selected_next_yaw_t0 = interpolated_yaw_ego_t0[index]
            perturbed_vel = np.linalg.norm(selected_next_pos_world - ego_agent_pos) / 0.1

            # record the perturbed trajectory
            original_traj_ego_t0_perturbed[perturbed_idx, :] = selected_next_pos_t0
            original_yaws_ego_t0_perturbed[perturbed_idx] = selected_next_yaw_t0 
            original_vel_ego_perturbed[perturbed_idx] = perturbed_vel

            # Perturb agents
            if self.perturb_agents and lag_agent is not None and len(filter_agents_by_track_id(cur_agents, lag_agent_track_id)) == 1: # We can only perturb lag agents if there are lag agents
                for idx, track_id in enumerate(list_agents_to_take):
                    if track_id == lag_agent_track_id: # if DRF perturbation is required
                        cur_lag_agent = filter_agents_by_track_id(filter_agents_by_labels(cur_agents),
                                        track_id)[0]
                        position = copy.deepcopy(cur_lag_agent["centroid"]) # IMPORTANT
                        yaw = cur_lag_agent["yaw"]
                        velocity = np.linalg.norm(cur_lag_agent["velocity"]) 

                        # Search original trajectory to change the waypoints close to next_agent_pos_world 
                        original_traj_t0 = all_other_agents_future_positions[idx]
                        original_traj_world = transform_points(original_traj_t0, np.linalg.inv(agent_from_world))
                        original_yaw_t0 = all_other_agents_future_yaws[idx]
                        original_yaw_world = all_other_agents_future_yaws_raw[idx]

                        # Interpolated agent trajectories for more accurate perturbation
                        interpolated_traj_t0, interpolated_yaw_t0 = self.get_linear_interpolated_trajectory(
                                                                    original_traj_t0, original_yaw_t0
                        )
                        interpolated_traj_world, interpolated_yaw_world = self.get_linear_interpolated_trajectory(
                                                                    original_traj_world, original_yaw_world
                        )

                        original_traj_lag_t0_perturbed = copy.deepcopy(original_traj_t0) # For traj check
                        original_traj_lag_t0 = copy.deepcopy(original_traj_t0) # For traj check (DO NOT CHANGE!)

                        index, world_from_raster = self.get_recommended_index(position, yaw, velocity, rasterizer, history_frames,
                                                        history_agents, history_tl_faces, interpolated_traj_world, agent=cur_lag_agent)
                        prev_perturbed_index = -1 # First step, no previous perturbed index
                        
                        # while maintaining safety distance to front ego vehicle 

                        ego2agent_dist = np.linalg.norm(ego_agent_pos - interpolated_traj_world, axis=1) 
                        dist_minus_SD = ego2agent_dist - 5 # assume SD = 5m
                        min_dist_index = np.argmin(dist_minus_SD)

                        # Every index between prev_perturbed_index and SD_index should be "safe" perturbed points
                        safe_perturbed_dist = dist_minus_SD[:min_dist_index + 1]
                        SDist = safe_perturbed_dist[safe_perturbed_dist > 0]
                        if SDist.size == 0: # agent too close too its leading car
                            SD_index = perturbed_idx # make SD_index = GT_index
                        else:
                            SD_index = np.argmin(SDist)
                        if (SD_index >= 0):
                            # SD_index exists
                            index = self.perturb_trajectory_rules(0, index, prev_perturbed_index, SD_index) # first frame, so -1 
                        else:
                            index = 0
                        selected_next_pos_world = interpolated_traj_world[index]
                        selected_next_pos_t0 = interpolated_traj_t0[index]
                        selected_next_yaw_t0 = interpolated_yaw_t0[index]
                        selected_next_yaw_world = interpolated_yaw_world[index]

                        # record the perturbed trajectory
                        all_other_agents_future_positions_perturbed[idx, perturbed_idx, :] = selected_next_pos_t0
                        all_other_agents_future_yaws_perturbed[idx, perturbed_idx, :] = selected_next_yaw_t0 
                        original_traj_lag_t0_perturbed[perturbed_idx] = selected_next_pos_t0

                        # Save the perturbed next_pos_world in future_frames

                        # Prepare agent for update
                        perturbed_agent = np.zeros(1, dtype=AGENT_DTYPE)
                        perturbed_vel = (selected_next_pos_world - position) / 0.1
                        perturbed_agent["velocity"] = perturbed_vel if np.linalg.norm(perturbed_vel) < 20 else perturbed_vel / np.linalg.norm(perturbed_vel) * 20
                        perturbed_agent["centroid"][:2] = selected_next_pos_world
                        perturbed_agent["yaw"] = selected_next_yaw_world
                        perturbed_agent["track_id"] = cur_lag_agent["track_id"]
                        perturbed_agent["extent"] = cur_lag_agent["extent"]
                        perturbed_agent["label_probabilities"][:, PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] = 1

                        future_agents[perturbed_idx] = self.insert_perturbed_agent(future_agents[perturbed_idx], perturbed_agent)

            # The rest of perturbed frames: agent is moving into the future

            while (perturbed_idx < self.future_num_frames - 18 + 8): # perturb 8 more frames for IL's coefficient computation
                cur_agents = filter_agents_by_labels(future_agents[perturbed_idx])
                cur_agents_ = []
                cur_agents_.append(cur_agents) # for rasterisation
                curr_frame = future_frames[perturbed_idx] 
                tl_faces = future_tl_faces[perturbed_idx:]
                ego_agent_pos = curr_frame["ego_translation"][:2]
                ego_agent_yaw = rotation33_as_yaw(curr_frame["ego_rotation"])
                ego_agent_vel = original_vel_ego_perturbed[perturbed_idx]

                # Prepare non-void frame for perturbation
                curr_frame_ = np.zeros(1, dtype=FRAME_DTYPE)

                curr_frame_["timestamp"] = curr_frame["timestamp"]
                curr_frame_["agent_index_interval"] = curr_frame["agent_index_interval"]
                curr_frame_["traffic_light_faces_index_interval"] = curr_frame["traffic_light_faces_index_interval"]
                curr_frame_["ego_translation"] = curr_frame["ego_translation"]
                curr_frame_["ego_rotation"] = curr_frame["ego_rotation"]

                # # Prepare non-void tl_faces for perturbation
                # tl_faces_ = np.zeros(1, dtype=TL_FACE_DTYPE)

                # tl_faces_["face_id"] = tl_faces["face_id"]
                # tl_faces_["traffic_light_id"] = tl_faces["traffic_light_id"]
                # tl_faces_["traffic_light_face_status"] = tl_faces["traffic_light_face_status"]

                if lag_agent is not None and len(filter_agents_by_track_id(cur_agents, lag_agent_track_id)) == 1: # lag agent exists
                    cur_lag_agent = filter_agents_by_track_id(filter_agents_by_labels(cur_agents),
                                                            lag_agent_track_id)[0]
                    lag_position = cur_lag_agent["centroid"]
                    lag_distance = np.linalg.norm(ego_agent_pos - lag_position)
                    lag_velocity = np.linalg.norm(cur_lag_agent["velocity"])
                else:
                    lag_distance = 30 # No lag agent in the raster, 50m is half the raster length
                    lag_velocity = 1e-8

                # Perturb ego
                index, world_from_raster = self.get_recommended_index(ego_agent_pos, ego_agent_yaw, ego_agent_vel, rasterizer, curr_frame_,
                                            cur_agents_, tl_faces, interpolated_traj_ego_world, lag_distance=lag_distance)
                          
                prev_dist, prev_perturbed_index = KDTree(interpolated_traj_ego_world).query(ego_agent_pos)

                if lead_agent is not None and len(filter_agents_by_track_id(cur_agents, lead_agent_track_id)) == 1: # lead agent exists
                    cur_lead_agent = filter_agents_by_track_id(cur_agents, lead_agent_track_id)[0]
                    lead_position = cur_lead_agent["centroid"]
                    lead_distance = np.linalg.norm(ego_agent_pos - lead_position)
                    lead_velocity = np.linalg.norm(cur_lead_agent["velocity"])
                else:
                    lead_position_raster = np.array([80, 50]) # No lead agent in the raster, 50m is half the raster length
                    lead_position = transform_point(lead_position_raster, world_from_raster)
                    lead_distance = 30
                    lead_velocity = 15

                index, world_from_raster = self.get_recommended_index_P_controller(ego_agent_pos, ego_agent_yaw, ego_agent_vel, 
                                                rasterizer, lead_distance, lead_velocity, lag_distance, lag_velocity, 
                                                interpolated_traj_ego_world)    
                
                # maintaining safety distance to front vehicle 
                ego2lead_dist = np.linalg.norm(lead_position - interpolated_traj_ego_world, axis=1) 
                dist_minus_SD = ego2lead_dist - 5 # assume SD = 5m
                min_dist_index = np.argmin(dist_minus_SD)

                # Every index between prev_perturbed_index and SD_index should be "safe" perturbed points
                safe_perturbed_dist = dist_minus_SD[:min_dist_index + 1]
                SDist = safe_perturbed_dist[safe_perturbed_dist > 0]
                if SDist.size == 0: # agent too close too its leading car
                    SD_index = 10 * (perturbed_idx+1) # make SD_index = GT_index
                else:
                    SD_index = np.argmin(SDist)

                GT_index = 10 * (perturbed_idx+1)
                index = self.perturb_trajectory_rules(GT_index, index, prev_perturbed_index, SD_index)

                # For perturbing next frame
                selected_next_pos_world = interpolated_traj_ego_world[index]
                selected_next_yaw_world = interpolated_yaw_ego_world[index]

                # record the perturbed trajectory
                # Prepare next frame for update
                perturbed_frame = np.zeros(1, dtype=FRAME_DTYPE)
                next_frame = future_frames[perturbed_idx + 1]

                perturbed_frame["timestamp"] = next_frame["timestamp"]
                perturbed_frame["agent_index_interval"] = next_frame["agent_index_interval"]
                perturbed_frame["traffic_light_faces_index_interval"] = next_frame["traffic_light_faces_index_interval"]
                perturbed_frame["ego_translation"] = next_frame["ego_translation"]
                perturbed_frame["ego_translation"][:, :2] = selected_next_pos_world
                perturbed_frame["ego_rotation"] = yaw_as_rotation33(selected_next_yaw_world)
            
                #self.insert_perturbed_ego(future_frames[perturbed_idx], selected_next_pos_world, selected_next_yaw_world)
                self.insert_perturbed_ego_(perturbed_idx + 1, future_frames, perturbed_frame)
                
                # For returning the perturbed trajectory
                selected_next_pos_t0 = interpolated_traj_ego_t0[index]
                selected_next_yaw_t0 = interpolated_yaw_ego_t0[index]
                perturbed_vel = np.linalg.norm(selected_next_pos_world - ego_agent_pos) / 0.1

                # record the perturbed trajectory
                original_traj_ego_t0_perturbed[perturbed_idx + 1, :] = selected_next_pos_t0
                original_yaws_ego_t0_perturbed[perturbed_idx + 1] = selected_next_yaw_t0 
                original_vel_ego_perturbed[perturbed_idx + 1] = perturbed_vel
                
                # Perturb lag agents
                if self.perturb_agents and lag_agent is not None and len(filter_agents_by_track_id(cur_agents, lag_agent_track_id)) == 1:
                    for idx, track_id in enumerate(list_agents_to_take):
                        if track_id == lag_agent_track_id: # if DRF perturbation is required

                            cur_lag_agent = filter_agents_by_track_id(filter_agents_by_labels(cur_agents), track_id)[0]
                            position = copy.deepcopy(cur_lag_agent["centroid"])
                            yaw = cur_lag_agent["yaw"]
                            velocity = np.linalg.norm(cur_lag_agent["velocity"])

                            # Search original trajectory to change the waypoints close to next_agent_pos_world 
                            original_traj_t0 = all_other_agents_future_positions[idx]
                            original_traj_world = transform_points(original_traj_t0, np.linalg.inv(agent_from_world))
                            original_yaw_t0 = all_other_agents_future_yaws[idx]
                            original_yaw_world = all_other_agents_future_yaws_raw[idx]

                            # Interpolated agent trajectories for more accurate perturbation
                            interpolated_traj_t0, interpolated_yaw_t0 = self.get_linear_interpolated_trajectory(
                                                                        original_traj_t0, original_yaw_t0
                            )
                            interpolated_traj_world, interpolated_yaw_world = self.get_linear_interpolated_trajectory(
                                                                        original_traj_world, original_yaw_world
                            )

                            index, world_from_raster = self.get_recommended_index(position, yaw, velocity, rasterizer, curr_frame_,
                                                        cur_agents_, tl_faces, interpolated_traj_world, agent=cur_lag_agent)
                            prev_dist, prev_perturbed_index = KDTree(interpolated_traj_world).query(position)
                            
                            # while maintaining safety distance to front ego vehicle 
                            # distance between ego and waypoints of current agent's original trajectory
                            ego2agent_dist = np.linalg.norm(ego_agent_pos - original_traj_world, axis=1) 
                            dist_minus_SD = ego2agent_dist - 5 # assume SD = 5m
                            min_dist_index = np.argmin(dist_minus_SD)

                            # Every index between prev_perturbed_index and SD_index should be "safe" perturbed points
                            safe_perturbed_dist = dist_minus_SD[:min_dist_index + 1]
                            SDist = safe_perturbed_dist[safe_perturbed_dist > 0]
                            if SDist.size == 0: # agent too close too its leading car
                                SD_index = 10 * (perturbed_idx+1) # make SD_index = GT_index
                            else:
                                SD_index = np.argmin(SDist)

                            GT_index = 10 * (perturbed_idx + 1)
                            index = self.perturb_trajectory_rules(GT_index, index, prev_perturbed_index, SD_index)
                            
                            selected_next_pos_world = interpolated_traj_world[index]
                            selected_next_pos_t0 = interpolated_traj_t0[index]
                            selected_next_yaw_t0 = interpolated_yaw_t0[index]
                            selected_next_yaw_world = interpolated_yaw_world[index]

                            # record the perturbed trajectory
                            all_other_agents_future_positions_perturbed[idx, perturbed_idx + 1, :] = selected_next_pos_t0
                            all_other_agents_future_yaws_perturbed[idx, perturbed_idx + 1, :] = selected_next_yaw_t0 
                            original_traj_lag_t0_perturbed[perturbed_idx + 1] = selected_next_pos_t0

                            # Save the perturbed next_pos_world in future_frames

                            # Prepare agent for update
                            perturbed_agent = np.zeros(1, dtype=AGENT_DTYPE)
                            perturbed_vel = (selected_next_pos_world - position) / 0.1
                            perturbed_agent["velocity"] = perturbed_vel if np.linalg.norm(perturbed_vel) < 20 else perturbed_vel / np.linalg.norm(perturbed_vel) * 20
                            perturbed_agent["centroid"][:2] = selected_next_pos_world
                            perturbed_agent["yaw"] = selected_next_yaw_world
                            perturbed_agent["track_id"] = cur_lag_agent["track_id"]
                            perturbed_agent["extent"] = cur_lag_agent["extent"]
                            perturbed_agent["label_probabilities"][:, PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]] = 1

                            future_agents[perturbed_idx + 1] = self.insert_perturbed_agent(future_agents[perturbed_idx + 1], perturbed_agent)

                perturbed_idx += 1
                ## Perturbed the GT trajevtories

        # crop similar to ego above
        all_other_agents_history_positions[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_yaws[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_extents[:, self.history_num_frames_agents + 1:] *= 0
        all_other_agents_history_availability[:, self.history_num_frames_agents + 1:] *= 0

        # compute other agents features
        # num_other_agents (M) x sequence_length x 2 (two being x, y)
        agents_points = all_other_agents_history_positions.copy()
        # num_other_agents (M) x sequence_length x 1
        agents_yaws = all_other_agents_history_yaws.copy()
        # agents_extents = all_other_agents_history_extents[:, :-1]
        # num_other_agents (M) x sequence_length x self._vector_length
        other_agents_polyline = np.concatenate([agents_points, agents_yaws], axis=-1)
        other_agents_polyline_availability = all_other_agents_history_availability.copy()

        if DRF_perturbation: # DRF perturbation required
            agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            "all_other_agents_future_positions": all_other_agents_future_positions_perturbed,
            "all_other_agents_future_yaws": all_other_agents_future_yaws_perturbed,
            "all_other_agents_future_extents": all_other_agents_future_extents,
            "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            "all_other_agents_types": all_other_agents_types,
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
            "target_positions": original_traj_ego_t0_perturbed,
            "target_yaws": original_yaws_ego_t0_perturbed,
        }
        else:
            agent_dict = {
            "all_other_agents_history_positions": all_other_agents_history_positions,
            "all_other_agents_history_yaws": all_other_agents_history_yaws,
            "all_other_agents_history_extents": all_other_agents_history_extents,
            "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            "all_other_agents_future_positions": all_other_agents_future_positions,
            "all_other_agents_future_yaws": all_other_agents_future_yaws,
            "all_other_agents_future_extents": all_other_agents_future_extents,
            "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            "all_other_agents_types": all_other_agents_types,
            "agent_trajectory_polyline": agent_trajectory_polyline,
            "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            "other_agents_polyline": other_agents_polyline,
            "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
        }

        return agent_dict

    def _vectorize_map(self, agent_centroid_m: np.ndarray, agent_from_world: np.ndarray,
                       history_tl_faces: List[np.ndarray]) -> dict:
        """Vectorize map elements in a frame.

        Arguments:
            agent_centroid_m: position of the target agent
            agent_from_world: inverted agent pose as 3x3 matrix
            history_tl_faces: traffic light faces in history frames

        Returns:
            dict: a dict containing the vectorized map representation of the target frame
        """
        # START WORKING ON LANES
        MAX_LANES = self.lane_cfg_params["max_num_lanes"]
        MAX_POINTS_LANES = self.lane_cfg_params["max_points_per_lane"]
        MAX_POINTS_CW = self.lane_cfg_params["max_points_per_crosswalk"]

        MAX_LANE_DISTANCE = self.lane_cfg_params["max_retrieval_distance_m"]
        INTERP_METHOD = InterpolationMethod.INTER_ENSURE_LEN  # split lane polyline by fixed number of points
        STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane
        MAX_CROSSWALKS = self.lane_cfg_params["max_num_crosswalks"]

        lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

        lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
        lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
        lanes_tl_feature = np.zeros((MAX_LANES, MAX_POINTS_LANES, 1), dtype=np.float32)

        # 8505 x 2 x 2
        lanes_bounds = self.mapAPI.bounds_info["lanes"]["bounds"]

        # filter first by bounds and then by distance, so that we always take the closest lanes
        lanes_indices = indices_in_bounds(agent_centroid_m, lanes_bounds, MAX_LANE_DISTANCE)
        distances = []
        for lane_idx in lanes_indices:
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)
            lane_dist = np.linalg.norm(lane["xyz_midlane"][:, :2] - agent_centroid_m, axis=-1)
            distances.append(np.min(lane_dist))
        lanes_indices = lanes_indices[np.argsort(distances)]

        # TODO: move below after traffic lights
        crosswalks_bounds = self.mapAPI.bounds_info["crosswalks"]["bounds"]
        crosswalks_indices = indices_in_bounds(agent_centroid_m, crosswalks_bounds, MAX_LANE_DISTANCE)
        crosswalks_points = np.zeros((MAX_CROSSWALKS, MAX_POINTS_CW, 2), dtype=np.float32)
        crosswalks_availabilities = np.zeros_like(crosswalks_points[..., 0])
        for i, xw_idx in enumerate(crosswalks_indices[:MAX_CROSSWALKS]):
            xw_id = self.mapAPI.bounds_info["crosswalks"]["ids"][xw_idx]
            points = self.mapAPI.get_crosswalk_coords(xw_id)["xyz"]
            points = transform_points(points[:MAX_POINTS_CW, :2], agent_from_world)
            n = len(points)
            crosswalks_points[i, :n] = points
            crosswalks_availabilities[i, :n] = True

        active_tl_faces = set(filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"].tolist())
        active_tl_face_to_color: Dict[str, str] = {}
        for face in active_tl_faces:
            try:
                active_tl_face_to_color[face] = self.mapAPI.get_color_for_face(face).lower()  # TODO: why lower()?
            except KeyError:
                continue  # this happens only on KIRBY, 2 TLs have no match in the map

        for out_idx, lane_idx in enumerate(lanes_indices[:MAX_LANES]):
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane = self.mapAPI.get_lane_as_interpolation(lane_id, STEP_INTERPOLATION, INTERP_METHOD)

            xy_left = lane["xyz_left"][:MAX_POINTS_LANES, :2]
            xy_right = lane["xyz_right"][:MAX_POINTS_LANES, :2]
            # convert coordinates into local space
            xy_left = transform_points(xy_left, agent_from_world)
            xy_right = transform_points(xy_right, agent_from_world)

            num_vectors_left = len(xy_left)
            num_vectors_right = len(xy_right)

            lanes_points[out_idx * 2, :num_vectors_left] = xy_left
            lanes_points[out_idx * 2 + 1, :num_vectors_right] = xy_right

            lanes_availabilities[out_idx * 2, :num_vectors_left] = 1
            lanes_availabilities[out_idx * 2 + 1, :num_vectors_right] = 1

            midlane = lane["xyz_midlane"][:MAX_POINTS_LANES, :2]
            midlane = transform_points(midlane, agent_from_world)
            num_vectors_mid = len(midlane)

            lanes_mid_points[out_idx, :num_vectors_mid] = midlane
            lanes_mid_availabilities[out_idx, :num_vectors_mid] = 1

            lanes_tl_feature[out_idx, :num_vectors_mid] = self.mapAPI.get_tl_feature_for_lane(
                lane_id, active_tl_face_to_color)

        # disable all points over the distance threshold
        valid_distances = np.linalg.norm(lanes_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_availabilities *= valid_distances
        valid_mid_distances = np.linalg.norm(lanes_mid_points, axis=-1) < MAX_LANE_DISTANCE
        lanes_mid_availabilities *= valid_mid_distances

        # 2 MAX_LANES x MAX_VECTORS x (XY + TL-feature)
        # -> 2 MAX_LANES for left and right
        lanes = np.concatenate([lanes_points, np.zeros_like(lanes_points[..., [0]])], axis=-1)
        # pad such that length is 3
        crosswalks = np.concatenate([crosswalks_points, np.zeros_like(crosswalks_points[..., [0]])], axis=-1)
        # MAX_LANES x MAX_VECTORS x 3 (XY + 1 TL-feature)
        lanes_mid = np.concatenate([lanes_mid_points, lanes_tl_feature], axis=-1)

        return {
            "lanes": lanes,
            "lanes_availabilities": lanes_availabilities.astype(np.bool),
            "lanes_mid": lanes_mid,
            "lanes_mid_availabilities": lanes_mid_availabilities.astype(np.bool),
            "crosswalks": crosswalks,
            "crosswalks_availabilities": crosswalks_availabilities.astype(np.bool),
        }

    def insert_perturbed_agent(self, frame_agents: np.ndarray, agent: np.ndarray) -> np.ndarray:
        """Insert a perturbed agent and ego in one frame.

        :param frame_agents: the agents in the inserted frame before inserting the agent
        :param agent: the agent to be inserted

        :returns: perturbed agents in the next frame
        """
        if not isinstance(frame_agents, np.ndarray):
            raise ValueError("dataset agents should be an editable np array")
        if not isinstance(agent, np.ndarray):
            raise ValueError("dataset agents should be an editable np array")

        idx_set = np.argwhere(agent["track_id"] == frame_agents["track_id"])
        assert len(idx_set) in [0, 1]

        if len(idx_set):
            # CASE 1
            # the agent is already there and we can just update it
            # we set also label_probabilities from the current one to ensure it is high enough
            idx_set = int(idx_set[0])
            frame_agents[idx_set: idx_set + 1] = agent
            
        else:
            # CASE 2
            # we need to insert the agent and move everything
            frame_agents = np.concatenate([frame_agents, agent], axis=0)
            # frame_agents = np.concatenate(
            #     [frame_agents[0: agents_slice.stop], agent, frame_agents[agents_slice.stop:]], 0
            # )

            # # move end of the current frame and all other frames start and end
            # dataset.frames[frame_idx]["agent_index_interval"] += (0, 1)
            # dataset.frames[frame_idx + 1:]["agent_index_interval"] += 1
            # raise ValueError("perturbed agents not in the next frame!")
        return frame_agents

    def insert_perturbed_ego_(self, idx: int, next_frames: np.ndarray, frame: np.ndarray) -> None:
        """Insert a perturbed agent and ego in one frame.

        :param idx: the index of the frame we are perturbing here
        :param next_frame: the frame to be inserted with the ego info
        :param frame_agents: the agents in the inserted frame before inserting the agent
        :param agent: the agent to be inserted
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("dataset ego should be an editable np array")
        if not isinstance(next_frames, np.ndarray):
            raise ValueError("dataset frames should be an editable np array")

        next_frames[idx] = frame
        next_frames[idx] = frame

    def insert_perturbed_ego(self, next_frame: dict, ego_pos: np.ndarray, ego_yaw: float) -> None:
        """Insert a perturbed agent and ego in one frame.

        :param next_frame: the frame to be inserted with the ego info
        :param frame_agents: the agents in the inserted frame before inserting the agent
        :param agent: the agent to be inserted
        """
        if not isinstance(ego_pos, np.ndarray):
            raise ValueError("dataset ego should be an editable np array")
        # if not isinstance(next_frame, dict):
        #     raise ValueError("dataset frames should be an editable np array")

        next_frame["ego_translation"][:2] = ego_pos
        next_frame["ego_rotation"] = yaw_as_rotation33(ego_yaw) 

    def perturb_trajectory_rules(self, GT_idx: int, curr_idx: int, prev_idx: int, SD_idx: int) -> int:
        """Perturb the trajectory index with hardcoded safety rules.

        :param GT_idx: the lower bound of perturbed idx at this step
        :param curr_idx: the perturbed trajectory index at current step
        :param prev_idx: the perturbed trajectory index at previous step
        :param SD_idx: the perturbed index should not exceed this index

        :Returns: curr_perturbed_idx
        """
        # if curr_idx <= GT_idx + 5 and np.random.rand() > 0.1:
        #     curr_idx += 1 # encourage slight speed-up
        # if (curr_idx > GT_idx + 5):
        #     curr_idx = prev_idx # not encourage large speed-up
        if (curr_idx <= GT_idx):
            #print("current perturbation slower than GT")
            return GT_idx + 1
        if curr_idx > GT_idx + 10:
            #print("current perturbation much faster than GT")
            return GT_idx + 10            
        if curr_idx <= prev_idx:
            #print("current perturbation slower than previous")
            return prev_idx + 1
        if (curr_idx >= SD_idx): # if perturbation not safe
            #print("current perturbation not safe")
            return prev_idx + 1
        return curr_idx
            
    def get_recommended_index(self, position: float, yaw: float, velocity: float, rasterizer: Rasterizer, 
                              frame: np.ndarray, agents: List[np.ndarray], tl_faces: List[np.ndarray], 
                              original_traj_world: np.ndarray, lag_distance: Optional[float] = None, 
                              agent: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
        """Get recommended perturbed index from objective cost map.

        :param position: position of the perturbed agent
        :param yaw: yaw of the perturbed agent
        :param velocity: position of the perturbed agent
        :param lag_distance: P controller adjusting longitudinal distance between rear car
        :param agent: None if ego

        :Returns: recommended_idx, world_from_raster
        """
        agent_img = rasterizer.rasterize(frame, agents, tl_faces, 
                                         agent=agent, position=position, yaw=yaw)
        pose_in_world = np.array([
                                    [np.cos(yaw), -np.sin(yaw), position[0]],
                                    [np.sin(yaw), np.cos(yaw), position[1]],
                                    [0, 0, 1],
                                ])

        raster_from_world = rasterizer.render_context.raster_from_local @ np.linalg.inv(pose_in_world)

        # Obtain next position estimation from agent_img
        next_x_raster, next_y_raster = risk_threshold_controller(agent_img, CT=9000, p_brake=0.6, p_keep=0.002,
                                                                SD=5, v_curr=velocity, lag_dist=lag_distance, v_des=15)
        # Transform raster coordinates to the world frame
        next_pos_world = transform_point(np.reshape(np.array([next_x_raster, next_y_raster]), 2), 
                                         np.linalg.inv(raster_from_world)) 

        # Search original trajectory to change the waypoints close to next_agent_pos_world
        distance, index = KDTree(original_traj_world).query(next_pos_world)

        return index, np.linalg.inv(raster_from_world)

    def get_recommended_index_P_controller(self, position: float, yaw: float, velocity: float, rasterizer: Rasterizer,  
                                           dis_lead: float, vel_lead: float, dis_lag: float, vel_lag: float,
                                           original_traj_world: np.ndarray) -> Tuple[int, np.ndarray]:
        """Get recommended perturbed index from longitudinal velocity P controller.

        :param position: position of the perturbed agent
        :param yaw: yaw of the perturbed agent
        :param velocity: velocity of the perturbed agent
        :param original_traj_world: trajectory of the perturbed agent in the world coordinater frame

        :Returns: recommended_idx, world_from_raster
        """
        dt = 0.1 # s
        dvMax = 2 * dt # m/s^(-2)
        v_max = 15 # m/s, next speed should not exceed this limit
        k_v = 0.025 # gain for speed-up/down
        raster_x = 50
        raster_y = 50
        phiv = 0

        pose_in_world = np.array([
                                    [np.cos(yaw), -np.sin(yaw), position[0]],
                                    [np.sin(yaw), np.cos(yaw), position[1]],
                                    [0, 0, 1],
                                ])

        raster_from_world = rasterizer.render_context.raster_from_local @ np.linalg.inv(pose_in_world)

        # Compute relative position and velocity
        vel2lead = velocity - vel_lead
        vel2lag = vel_lag - velocity

        # Compute time-to-collision
        # time2rear_collision = dis_lag / vel2lag
        # time2front_collision = dis_lead / vel2lead

        # P-controller to keep desired velocity
        if dis_lag < 15: # if rear agent is close, ego should try to speed up
            dv = np.fmin(dvMax, k_v * np.abs(v_max - velocity))
            next_vel = velocity + np.sign(v_max - velocity) * dv
        else: # keep current speed for now
            next_vel = velocity

        # Obtain next position estimation from bike model
        next_x_raster = raster_x + next_vel * np.cos(phiv) * dt
        next_y_raster = raster_y + next_vel * np.sin(phiv) * dt

        # Transform raster coordinates to the world frame
        next_pos_world = transform_point(np.reshape(np.array([next_x_raster, next_y_raster]), 2), 
                                         np.linalg.inv(raster_from_world)) 

        # Search original trajectory to change the waypoints close to next_agent_pos_world
        distance, index = KDTree(original_traj_world).query(next_pos_world)

        return index, np.linalg.inv(raster_from_world)

    def get_linear_interpolated_trajectory(self, original_trajectory: np.ndarray, original_yaws: np.ndarray,
                                           new_sampling_rate: Optional[float] = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear perturbated trajectory with given sampling rate

        :param original_trajectory: the original trajectory requiring interpolation
        :param original_yaws: the original orientation along the trajectory
        :param new_sampling_rate: the new trajectory's sampling rate (Default: 0.01, meaning 10 times more waypoints than the original trajectory)

        :Returns: interpolated trajectory, interpolated orientations
        """
        original_sampling_rate = 0.1 # 0.1s, as used by l5kit
        inserted_points_between_two = (int) (original_sampling_rate / new_sampling_rate)

        original_x = original_trajectory[:, 0]
        original_y = original_trajectory[:, 1]
        original_yaw = original_yaws[:, 0]

        original_time = np.linspace(0, original_trajectory.shape[0], original_trajectory.shape[0])
        interpolated_time = np.linspace(0, original_trajectory.shape[0], inserted_points_between_two * (original_trajectory.shape[0] - 1))

        interpolated_x = np.interp(interpolated_time, original_time, original_x)
        interpolated_y = np.interp(interpolated_time, original_time, original_y)
        interpolated_yaw = np.interp(interpolated_time, original_time, original_yaw)

        interpolated_trajectory = np.vstack((interpolated_x, interpolated_y)).T

        return interpolated_trajectory, interpolated_yaw
