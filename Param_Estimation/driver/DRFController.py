import numpy as np

from typing import Dict, List, Optional, Tuple
from scipy import optimize

from Param_Estimation.driver import DRFModel

def leading_distance_extract(obj_map: np.ndarray) -> float:
    """Extract the distance between current agent and its leading car from its objective cost map.

    Arguments: 
        obj_map: the objective map of an agent

    Returns:
        float: the leading distance
    """
    front = obj_map[49:51, 50:] # same lane, behind the agent
    front_obs = np.argmax(front == 2500)
    if (front_obs != 0 and front_obs != 50):
        front_obs_tmp = np.argmax(front == 2500, axis=1)
        front_obs_tmp = front_obs_tmp[front_obs_tmp > 0]
        front_obs = np.min(front_obs_tmp)
    return front_obs

def risk_threshold_controller(obj_map: np.ndarray, CT: float, p_brake: float, p_keep: float, 
                              SD: float, v_curr: float, lag_dist: Optional[float], v_des: Optional[float] = 15) -> Tuple[float, float]:
    """A risk-threshold controller based on driver's perceived risk theory.

    Arguments: 
        obj_map: the objective map of an agent
        CT: cost threshold
        p_brake: parameter p during braking
        p_keep: parameter p during lane keeping
        SD: safe distance
        v_curr: current velocity
        lag_dist: distance between ego and rear agent
        v_des: desired velocity

    Returns:
        x, y position in the raster frame
    """
    dt = 0.1 # s
    dvMax = 2 * dt # m/s^(-2)
    v_max = 15 # m/s, next speed should not exceed this limit
    k_v = 0.025 # gain for speed-up/down

    front_obs = leading_distance_extract(obj_map)

    # if leading distance <= 1.5*SD, use braking parameter
    if (front_obs > 0 and front_obs <= 1.5 * SD): 
        #print("brake")
        egoDRF = DRFModel(p=p_brake, t_la=5, c=0.4, m=0.0001, k_1=0.2, k_2=1.14, safe_distance=SD, cost_threshold=CT)
        
    # else use car folowing parameter
    else:
        #print("follow")
        egoDRF = DRFModel(p=p_keep, t_la=5, c=0.4, m=0.0001, k_1=0.2, k_2=1.14, safe_distance=SD, cost_threshold=CT)
        
    egoDRF.v = v_curr
    egoDRF.obj_map = obj_map
    egoDRF.x = 50 # assume the given agent will be always at the center of the raster frame
    egoDRF.y = 50 #
    egoDRF.phiv = 0.

    p_risk = egoDRF.overallProcess()
    egoDRF.delta, egoDRF.v = driverController(p_risk, egoDRF, v_des, CT)

    if lag_dist is not None and lag_dist < 15: # if rear agent is close, ego should try to speed up
        dv = np.fmin(dvMax, k_v * np.abs(v_max - egoDRF.v))
        egoDRF.v = egoDRF.v + np.sign(v_max - egoDRF.v) * dv

    egoDRF.carKinematics()
    return egoDRF.x, egoDRF.y

def driverController(curr_risk: float, ego: DRFModel, v_des: float, Ct: float) -> Tuple[float, float]:
    """A simple driver controller imterpreting driver's perceived risk and predict next velocity and steering.

    Arguments: 
        curr_risk: the current driver's perceived risk
        ego: the agent DRF model
        v_des: desired velocity
        CT: cost threshold

    Returns:
        float: the leading distance
    """

    ## Controller parameters
    k_h = 0.02 # gain of default heading P-controller
    k_v = 0.025 # gain of vehicle's speed-up/down, could be different for normal or sport driving
    k_vc = 1.5 * 1e-4 # gain of vehicle's speed-down contributed by the perceived risk (cost)
    dt = 0.1 # [s] step time
    dvMax = 4 * dt # [m/s^2] Vehicle max decel and acceleration
    dvstepMax = 20 * dt # For fminbnd search
    dsMax = np.pi / 180 * 1 * dt # [rad/dt] Note: Here assume steer limit is 10 degree/s, be careful with unit!
    dstepMax = np.pi / 180 * 50 * dt # [rad/dt] Note: For fminbound search
    ## Controller parameters

    risk_threshold = Ct
    desired_vel = v_des
    curr_vel = ego.v
    ego_heading = ego.phiv # assume ego's ground truth heading is the current road heading
    curr_steering = ego.delta
    next_steering =  curr_steering #+ k_h * (curr_road_heading - ego_curr_heading_world)
    if (curr_risk <= risk_threshold and curr_vel <= desired_vel):
        # condition 1
        # velocity update
        dv = np.fmin(dvMax, np.abs(k_v * (desired_vel - curr_vel)))
        next_vel = curr_vel + np.sign(desired_vel - curr_vel) * dv
        #print("condition1 vel = ", next_vel)
        return next_steering, next_vel
    elif (curr_risk > risk_threshold and curr_vel <= desired_vel):
        # check if changing velocity alone can reduce the cost below the threshold
        opt_v, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeVelocity, x1=curr_vel - dvstepMax, 
                                                                x2=curr_vel + dvstepMax, full_output=True)
        if (minCost >= risk_threshold):
            # condition 2a
            # velocity update
            dv = np.fmin(dvMax, np.abs(opt_v - curr_vel)) 
            next_vel = curr_vel + np.sign(opt_v - curr_vel) * dv
            if (next_vel < 0):
                next_vel = 0
            #print("condition2a vel = ", next_vel)
            return next_steering, next_vel
            
#         model slows down
#         proportional to Cop − Ck (and not Cop − Ct) since the steering applied = wop is
#         expected to reduce Ck to Cop. This is done so that we do not slow down more than
#         what is required. Hence, w_k+1 = wop 
        elif (minCost < risk_threshold):
            # condition 2b
            # velocity update
            opt_v, minCost, ierr, numfunc = optimize.fminbound(func=ego.optimizeVelocityCt, x1=curr_vel - dvstepMax, 
                                                               x2=curr_vel + dvstepMax, full_output=True)
            dv = np.fmin(dvMax, np.abs(opt_v - curr_vel))
            next_vel = curr_vel + np.sign(opt_v - curr_vel) * dv
            if (next_vel < 0):
                next_vel = 0
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
        #print("condition4 vel = ", next_vel)
        return next_steering, next_vel

    else: 
        raise ValueError("Error at driver controller: no situations match!")
        print("Next steering = ", next_steering)
        print("Next Velocity = ", next_vel)

