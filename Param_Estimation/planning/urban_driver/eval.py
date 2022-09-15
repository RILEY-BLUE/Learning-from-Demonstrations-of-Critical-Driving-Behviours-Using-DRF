from typing import Optional, List

import torch
from l5kit.cle.composite_metrics import CompositeMetricAggregator
from l5kit.cle.scene_type_agg import compute_cle_scene_type_aggregations
#from l5kit.cle.validators import ValidationCountingAggregator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from Param_Estimation.cle.validators import RangeValidator, ValidationCountingAggregator
from l5kit.dataset import EgoDataset
#from l5kit.environment.callbacks import L5KitEvalCallback
from Param_Estimation.environment.callbacks import L5KitEvalCallback
#from l5kit.environment.gym_metric_set import CLEMetricSet
from Param_Estimation.environment.gym_metric_set import CLEMetricSet
from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from stable_baselines3.common.logger import Logger

def eval_model(model: torch.nn.Module, dataset: EgoDataset, logger: Logger, d_set: str, iter_num: int,
               scenes_to_unroll: List, num_simulation_steps: int = None,
               enable_scene_type_aggregation: Optional[bool] = False,
               scene_id_to_type_path: Optional[str] = None) -> List:
    """Evaluator function for the drivenet model. Evaluate the model using the CLEMetricSet
    of L5Kit. Logging is performed in the Tensorboard logger.
    :param model: the trained model to evaluate
    :param dataset: the dataset on which the models is evaluated
    :param logger: tensorboard logger to log the evaluation results
    :param d_set: the type of dataset being evaluated ("train" or "eval")
    :param iter_num: iteration number of training (to log in tensorboard)
    :param scenes_to_unroll: list of scene indices to evaluate in the dataset
    :param num_simulation_steps: Number of steps to unroll the model for.
    :param enable_scene_type_aggregation: enable evaluation according to scene type
    :param scene_id_to_type_path: path to the csv file mapping scene id to scene type

    :return: A list of violated scenes' frame indices for re-training
    """

    model.eval()
    torch.set_grad_enabled(False)

    # Close Loop Simulation
    sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                               distance_th_far=30, distance_th_close=15, num_simulation_steps=num_simulation_steps,
                               start_frame_index=25, show_info=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_loop = ClosedLoopSimulator(sim_cfg, dataset, device, model_ego=model, model_agents=None)

    # metric set
    metric_set = CLEMetricSet()

    # unroll
    #batch_unroll = 100 # num_scenes 400, num_frames_unroll 40

    sim_outs = sim_loop.unroll(scenes_to_unroll)
    metric_set.evaluator.evaluate(sim_outs)

    # Aggregate metrics (ADE, FDE)
    ade, fde = L5KitEvalCallback.compute_ade_fde(metric_set)
    logger.record(f'{d_set}/ade', round(ade, 3))
    logger.record(f'{d_set}/fde', round(fde, 3))

    # Aggregate validators
    validation_results = metric_set.evaluator.validation_results()
    agg, agg_failed = ValidationCountingAggregator().aggregate(validation_results)
    for k, v in agg.items():
        logger.record(f'{d_set}/{k}', v.item())
    # Add total collisions as well
    tot_collision = agg['collision_front'].item() + agg['collision_side'].item() + agg['collision_rear'].item()
    logger.record(f'{d_set}/total_collision', tot_collision)

    # Extract violation scenes for re-training
    violated_scenes_list = []
    for k, v in agg_failed.items():
        violated_scenes_list.extend(v)
    non_vio_scenes_list = [x for x in scenes_to_unroll if x not in violated_scenes_list]
    # Extract the ranges of frame from all violated scenes for re-training
    # (in order to pass it to a new dataloader)
    train_indices = []
    num_training_scenes = 0
    for scene_id in violated_scenes_list:
        curr_scene_end_frame = dataset.cumulative_sizes[scene_id]
        if scene_id == 0:
            prev_scene_end_frame = 0
        else:
            prev_scene_end_frame = dataset.cumulative_sizes[scene_id - 1]
        train_indices.extend(list(range(prev_scene_end_frame + 10, curr_scene_end_frame - 40)))
        num_training_scenes += 1

    # Self-adjust violated/normal scenes trade-off
    while num_training_scenes < len(scenes_to_unroll) / 10: # condense training focusing on violations
        for scene_id in non_vio_scenes_list:
            curr_scene_end_frame = dataset.cumulative_sizes[scene_id]
            if scene_id == 0:
                prev_scene_end_frame = 0
            else:
                prev_scene_end_frame = dataset.cumulative_sizes[scene_id - 1]
            train_indices.extend(list(range(prev_scene_end_frame + 10, curr_scene_end_frame - 40)))
            num_training_scenes += 1

    # Aggregate composite metrics
    composite_metric_results = metric_set.evaluator.composite_metric_results()
    comp_agg = CompositeMetricAggregator().aggregate(composite_metric_results)
    for k, v in comp_agg.items():
        logger.record(f'{d_set}/{k}', v.item())

    # Dump log so the evaluation results are printed with the correct timestep
    logger.record("time/total timesteps", iter_num, exclude="tensorboard")
    logger.dump(iter_num)

    metric_set.evaluator.reset()
    torch.set_grad_enabled(True)

    return train_indices