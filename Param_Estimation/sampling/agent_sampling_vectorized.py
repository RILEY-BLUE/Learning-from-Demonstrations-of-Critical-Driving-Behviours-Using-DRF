from typing import Optional

import numpy as np

# from l5kit.vectorization.vectorizer import Vectorizer
from Param_Estimation.vectorization.vectorizer import Vectorizer

from ..data import filter_agents_by_labels, filter_agents_by_frames, PERCEPTION_LABEL_TO_INDEX
from ..data.filter import filter_agents_by_track_id
from ..geometry import compute_agent_pose, rotation33_as_yaw, transform_point
from ..kinematic import Perturbation
from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from ..sampling.agent_sampling import compute_agent_velocity, get_agent_context, get_relative_poses


def generate_agent_sample_vectorized(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    render_context: RenderContext,
    history_num_frames_ego: int,
    history_num_frames_agents: int,
    future_num_frames: int,
    step_time: float,
    filter_agents_threshold: float,
    vectorizer: Vectorizer,
    rasterizer: Rasterizer,
    perturbation: Optional[Perturbation] = None,
    DRF_perturbation: Optional[bool] = False,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model with vectorized inputs.
    A deep prediction model takes as input the state of the world in vectorized form,
    and outputs where that agent will be some seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the representation and the returned targets are derived from
        their future states.
        history_num_frames_ego (int): Amount of ego history frames to include
        history_num_frames_agents (int): Amount of agent history frames to include
        future_num_frames (int): Amount of future frames to include
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict containing e.g. the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask,
        the vectorized input representation features, and (optional) a raster image
    """
    history_num_frames_max = max(history_num_frames_ego, history_num_frames_agents)
    (
        history_frames,
        future_frames,
        history_agents,
        future_agents,
        history_tl_faces,
        future_tl_faces,
    ) = get_agent_context(state_index, frames, agents, tl_faces, history_num_frames_max, future_num_frames,)

    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]

    if DRF_perturbation:
        # filter lagging vehicle from the agents in the current frame
        ego_centroid_m = cur_frame["ego_translation"][:2]
        ego_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        raster_from_world = render_context.raster_from_world(ego_centroid_m, ego_yaw_rad)

        lag_agent = filter_lag_agents(cur_agents, cur_agents, raster_from_world)
        lead_agent = filter_lead_agents(cur_agents, cur_agents, raster_from_world)

    ## Some smart perturbation function using history and future info

    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        agent_type_idx = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
            )[0]
        except IndexError:
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        agent_type_idx = np.argmax(agent["label_probabilities"])
        selected_agent = agent

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_coords_offset, future_yaws_offset, future_extents, future_availability = get_relative_poses(
        future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, agent_yaw_rad
    )

    # For vectorized version we require both ego and agent history to be a Tensor of same length
    # => fetch history_num_frames_max for both, and later zero out frames exceeding the set history length.
    # Use history_num_frames_max + 1 because it also includes the current frame.
    history_coords_offset, history_yaws_offset, history_extents, history_availability = get_relative_poses(
        history_num_frames_max + 1, history_frames, selected_track_id, history_agents, agent_from_world, agent_yaw_rad
    )

    history_coords_offset[history_num_frames_ego + 1:] *= 0
    history_yaws_offset[history_num_frames_ego + 1:] *= 0
    history_extents[history_num_frames_ego + 1:] *= 0
    history_availability[history_num_frames_ego + 1:] *= 0

    input_im = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)

    history_vels_mps, future_vels_mps = compute_agent_velocity(history_coords_offset, future_coords_offset, step_time)

    frame_info = {
        "extent": agent_extent_m,
        "type": agent_type_idx,
        "image": input_im,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "raster_from_world": raster_from_world,
        "target_velocities": future_vels_mps,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_extents": future_extents,
        "target_availabilities": future_availability.astype(np.bool),
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_extents": history_extents,
        "history_availabilities": history_availability.astype(np.bool),
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm(future_vels_mps[0]),
    }

    if selected_agent != None and len(selected_agent["velocity"]) > 0:
        frame_info["curr_speed"] = selected_agent["velocity"][0]

    vectorized_features = vectorizer.vectorize(selected_track_id, agent_centroid_m, agent_yaw_rad, agent_from_world,
                                            history_frames, history_agents, history_tl_faces, future_tl_faces,
                                            history_coords_offset, history_yaws_offset, history_availability, 
                                            future_frames, future_agents, future_vels_mps, rasterizer, lag_agent, 
                                            lead_agent, DRF_perturbation)
    if DRF_perturbation:
        frame_info["target_positions"] = vectorized_features["target_positions"]
        frame_info["target_yaws"] = vectorized_features["target_yaws"]

    return {**frame_info, **vectorized_features}

def filter_lag_agents(scene_idx: int, frame_agents: np.ndarray,
                      raster_from_world: np.ndarray) -> np.ndarray:
    """Filter agents on the same main road as the ego vehicle:

    This is to enhance computing efficiency because we want to consider vehicles on the main road around the ego.

    :param scene_idx: the scene index (used to check for agents_infos)
    :param frame_agents: the agents in this frame
    :param raster_from_world: the transformation from world to ego's raster coordinate system.
    :return: the filtered agents
    """
    # keep only vehicles
    car_index = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
    vehicle_mask = frame_agents["label_probabilities"][:, car_index]

    dt_agents_ths = 0.5 # filter_agent_threshold in config.yaml
    vehicle_mask = vehicle_mask > dt_agents_ths
    frame_agents = frame_agents[vehicle_mask]

    #main_road_mask = np.zeros(len(frame_agents), dtype=np.bool)
    min_dist = 50
    center = np.array([50, 50])
    closest_agent = None
    for idx_agent, agent in enumerate(frame_agents):
        agent_raster_coords = transform_point(agent["centroid"], raster_from_world)

        # filter protocal
        mins = 15, 48   # x_min, y_min (raster coordinates)
        maxs = 50, 52  # x_max, y_max (raster coordinates)
        # filter protocal

        agent_in_range = (agent_raster_coords >= mins).all() & (agent_raster_coords <= maxs).all()
        if agent_in_range:
            agent2ego_dist = np.linalg.norm(agent_raster_coords - center) # default is l2 distance
            if agent2ego_dist < min_dist:
                min_dist = agent2ego_dist 
                closest_agent = agent

    return closest_agent

def filter_lead_agents(scene_idx: int, frame_agents: np.ndarray,
                      raster_from_world: np.ndarray) -> np.ndarray:
    """Filter leading agent on the same main road as the ego vehicle:

    This is to enhance computing efficiency because we want to consider vehicles on the main road around the ego.

    :param scene_idx: the scene index (used to check for agents_infos)
    :param frame_agents: the agents in this frame
    :param raster_from_world: the transformation from world to ego's raster coordinate system.
    :return: the filtered agents
    """
    # keep only vehicles
    car_index = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
    vehicle_mask = frame_agents["label_probabilities"][:, car_index]

    dt_agents_ths = 0.5 # filter_agent_threshold in config.yaml
    vehicle_mask = vehicle_mask > dt_agents_ths
    frame_agents = frame_agents[vehicle_mask]

    #main_road_mask = np.zeros(len(frame_agents), dtype=np.bool)
    min_dist = 50
    center = np.array([50, 50])
    closest_agent = None
    for idx_agent, agent in enumerate(frame_agents):
        agent_raster_coords = transform_point(agent["centroid"], raster_from_world)

        # filter protocal
        mins = 50, 49   # x_min, y_min (raster coordinates)
        maxs = 80, 51  # x_max, y_max (raster coordinates)
        # filter protocal

        agent_in_range = (agent_raster_coords >= mins).all() & (agent_raster_coords <= maxs).all()
        if agent_in_range:
            agent2ego_dist = np.linalg.norm(agent_raster_coords - center)
            if agent2ego_dist < min_dist:
                min_dist = agent2ego_dist 
                closest_agent = agent

    return closest_agent
