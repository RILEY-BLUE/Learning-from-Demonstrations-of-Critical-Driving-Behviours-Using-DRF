from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE
from l5kit.data.filter import filter_tl_faces_by_status, filter_agents_by_labels, filter_agents_by_track_id
from l5kit.data.map_api import InterpolationMethod, MapAPI, TLFacesColors
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points

from ..rasterization.rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from .render_context import RenderContext


# sub-pixel drawing precision constants
CV2_SUB_VALUES = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE = 2 ** CV2_SUB_VALUES["shift"]
INTERPOLATION_POINTS = 20


class RasterEls(IntEnum):  # map elements
    LANE_NOTL = 0
    ROAD = 1
    CROSSWALK = 2


COLORS = {
    TLFacesColors.GREEN.name: (0, 255, 0),
    TLFacesColors.RED.name: (255, 0, 0),
    TLFacesColors.YELLOW.name: (255, 255, 0),
    RasterEls.LANE_NOTL.name: (255, 217, 82),
    RasterEls.ROAD.name: (17, 17, 31),
    RasterEls.CROSSWALK.name: (255, 117, 69),
}


def indices_in_bounds(center: np.ndarray, bounds: np.ndarray, half_extent: float) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds intersects the one defined around
    center (square with side 2*half_side)

    Args:
        center (float): XY of the center
        bounds (np.ndarray): array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]
        half_extent (float): half the side of the bounding box centered around center

    Returns:
        np.ndarray: indices of elements inside radius from center
    """
    x_center, y_center = center

    x_min_in = x_center > bounds[:, 0, 0] - half_extent
    y_min_in = y_center > bounds[:, 0, 1] - half_extent
    x_max_in = x_center < bounds[:, 1, 0] + half_extent
    y_max_in = y_center < bounds[:, 1, 1] + half_extent
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    coords = coords * CV2_SHIFT_VALUE
    coords = coords.astype(np.int)
    return coords

def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:
    """Get a valid agent with information from the AV. Ford Fusion extent is used.

    :param frame: The frame from which the Ego states are extracted
    :return: An agent numpy array of the Ego states
    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent

def draw_boxes(
        raster_size: Tuple[int, int],
        raster_from_world: np.ndarray,
        agents: np.ndarray,
        color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """ Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected on the image space.
    Finally, cv2 draws the boxes.

    :param raster_size: Desired output raster image size
    :param raster_from_world: Transformation matrix to transform from world to image coordinate
    :param agents: An array of agents to be drawn
    :param color: Single int or RGB color
    :return: the image with agents rendered. A RGB image if using RGB color, otherwise a GRAY image
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
        im_flag = np.zeros((raster_size[1], raster_size[0]))
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    box_world_coords = get_box_world_coords(agents)
    box_raster_coords = transform_points(box_world_coords.reshape((-1, 2)), raster_from_world)
    
    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_raster_coords = cv2_subpixel(box_raster_coords.reshape((-1, 4, 2)))
    # print(box_raster_coords)
    cv2.fillPoly(im, box_raster_coords, color, **CV2_SUB_VALUES)
    im_flag[im > 254] = 2500
    #np.set_printoptions(threshold=999999)

    return im_flag

def get_box_world_coords(agents: np.ndarray) -> np.ndarray:
    """Get world coordinates of the 4 corners of the bounding boxes

    :param agents: agents array of size N with centroid (world coord), yaw and extent
    :return: array of shape (N, 4, 2) with the four corners of each agent
    """
    # shape is (1, 4, 2)
    corners_base_coords = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5)[None, :, :]

    # compute the corner in world-space (start in origin, rotate and then translate)
    # extend extent to shape (N, 1, 2) so that op gives (N, 4, 2)
    corners_m = corners_base_coords * np.asarray([[4.5, 1.6, 1.5]])[:, None, :2]  # corners in zero agents["extent"]
    #print(agents["extent"])
    s = np.sin(agents["yaw"])
    c = np.cos(agents["yaw"])
    # note this is clockwise because it's right-multiplied and not left-multiplied later,
    # and therefore we're still rotating counterclockwise.
    rotation_m = np.moveaxis(np.array(((c, s), (-s, c))), 2, 0)
    # extend centroid to shape (N, 1, 2) so that op gives (N, 4, 2)
    box_world_coords = corners_m @ rotation_m + agents["centroid"][:, None, :2]
    return box_world_coords

class MapBuilder(Rasterizer):
    """
    Build objective and subjective map around the ego vehicle using the vectorised semantic map (generally loaded from json files).
    """

    def __init__(
            self,
            render_context: RenderContext,
            filter_agents_threshold: float,
            history_num_frames: int,
            semantic_map_path: str,
            world_to_ecef: np.ndarray,
            render_ego_history: bool = True,
    ):
        self.render_context = render_context
        self.filter_agents_threshold = filter_agents_threshold
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio

        self.world_to_ecef = world_to_ecef

        self.mapAPI = MapAPI(semantic_map_path, world_to_ecef)
        self.history_num_frames = history_num_frames
        self.render_ego_history = render_ego_history

    def rasterize(
            self,
            history_frames: np.ndarray,
            history_agents: List[np.ndarray],
            history_tl_faces: List[np.ndarray],
            agent: Optional[np.ndarray] = None,
            position: Optional[np.ndarray] = None,
            yaw: Optional[float] = None,
            without_agents: Optional[bool] = False,
    ) -> np.ndarray:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]
        if position is not None and yaw is not None:
            ego_translation_m = np.append(position, history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = yaw

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        obj_im = self.render_objective_map(center_in_world_m, raster_from_world, history_tl_faces[0], history_frames, 
                                        history_agents, agent, without_agents=without_agents)
        return obj_im.astype(np.float32) #/ 255

    def render_objective_map(
            self, center_in_world: np.ndarray, raster_from_world: np.ndarray, tl_faces: np.ndarray, history_frames: np.ndarray, 
            history_agents: List[np.ndarray], agent: Optional[np.ndarray] = None, without_agents: Optional[bool] = False
    ) -> np.ndarray:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """
        # Semantic map (static environment)
        # Note: on road -> cost = 0; off road -> cost = 500
        img_sem = 500 * np.ones(shape=(self.raster_size[1], self.raster_size[0])) # 3

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # get all lanes as interpolation so that we can transform them all together

        lane_indices = indices_in_bounds(center_in_world, self.mapAPI.bounds_info["lanes"]["bounds"], raster_radius)
        lanes_mask: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool))
        lanes_area = np.zeros((len(lane_indices) * 2, INTERPOLATION_POINTS, 2))

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.mapAPI.bounds_info["lanes"]["ids"][lane_idx]

            # interpolate over polyline to always have the same number of points
            lane_coords = self.mapAPI.get_lane_as_interpolation(
                lane_idx, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
            lanes_area[idx * 2] = lane_coords["xyz_left"][:, :2]
            lanes_area[idx * 2 + 1] = lane_coords["xyz_right"][::-1, :2]

            lane_type = RasterEls.LANE_NOTL.name
            lane_tl_ids = set(self.mapAPI.get_lane_traffic_control_ids(lane_idx))
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                lane_type = self.mapAPI.get_color_for_face(tl_id)

            lanes_mask[lane_type][idx * 2: idx * 2 + 2] = True

        if len(lanes_area):
            lanes_area = cv2_subpixel(transform_points(lanes_area.reshape((-1, 2)), raster_from_world))

            for lane_area in lanes_area.reshape((-1, INTERPOLATION_POINTS * 2, 2)):
                # need to for-loop otherwise some of them are empty
                cv2.fillPoly(img_sem, [lane_area], 0, **CV2_SUB_VALUES) # COLORS[RasterEls.ROAD.name]

            # lanes_area = lanes_area.reshape((-1, INTERPOLATION_POINTS, 2))
            # for name, mask in lanes_mask.items():  # draw each type of lane with its own color
            #     cv2.polylines(img_sem, lanes_area[mask], False, COLORS[name], **CV2_SUB_VALUES)
        
        # Agent map (dynamic obstacles) # TODO
        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.raster_size[1], self.raster_size[0], self.history_num_frames + 1)

        # if agent -> cost = 2500; else -> cost = 0
        agents_images = np.zeros(out_shape)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        if (without_agents == False):
            for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
                agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
                # note the cast is for legacy support of dataset before April 2020
                av_agent = get_ego_as_agent(frame).astype(agents.dtype)

                if agent is None:
                    ego_agent = av_agent
                else:
                    ego_agent = filter_agents_by_track_id(agents, agent["track_id"])
                    agents = np.append(agents, av_agent)  # add av_agent to agents
                    if len(ego_agent) > 0:  # check if ego_agent is in the frame
                        agents = agents[agents != ego_agent[0]]  # remove ego_agent from agents

                agents_images[..., i] = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                if len(ego_agent) > 0 and (self.render_ego_history or i == 0):
                    ego_images[..., i] = draw_boxes(self.raster_size, raster_from_world, ego_agent, 255)

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        agt_im = agents_images # np.concatenate((agents_images, ego_images), -1)

        # plot crosswalks
        # crosswalks = []
        # for idx in indices_in_bounds(center_in_world, self.mapAPI.bounds_info["crosswalks"]["bounds"], raster_radius):
        #     crosswalk = self.mapAPI.get_crosswalk_coords(self.mapAPI.bounds_info["crosswalks"]["ids"][idx])
        #     xy_cross = cv2_subpixel(transform_points(crosswalk["xyz"][:, :2], raster_from_world))
        #     crosswalks.append(xy_cross)

        # cv2.polylines(img, crosswalks, True, COLORS[RasterEls.CROSSWALK.name], **CV2_SUB_VALUES)

        return np.add(agt_im.reshape(self.raster_size[1], self.raster_size[0]), img_sem)

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        return (in_im * 255).astype(np.uint8)

    def num_channels(self) -> int:
        return 3
