import numpy as np
from typing import Tuple
from scipy import optimize

class DRFModel:
    def __init__(
        self,
        p: float,
        t_la: float,
        c: float,
        m: float,
        k_1: float,
        k_2: float,
        safe_distance: float,
        cost_threshold: float
    ):
        """
        Get a DRF model that takes in the objective map and output the vehicle's next states (x, y, heading, v, δ)

        Args:
            obj_map (np.ndarray): a 2d array contains the cost of the environment (road, agents)
            p = 0.0064: "steepness of the parabola" -> Gaussian's height
            t_la = 3.5 // [s] "look-ahead time"
            c = 0.5 // "a quarter of car width, as +- 2sigma takes up 95% of Gaussian" -> Gaussian's width
            m = 0.001 // "slope of widening" -> Gaussian's width
            k_1, k_2 = 0, 1.3823 // "parameter for inner/outer edges for the DRF" -> Gaussian's width
        """
        self.time_step = 0.1 # [s] same as the interval between frames
        self.obj_map = []
        
        self.p = p
        self.t_la = t_la
        self.c = c
        self.m = m
        self.k_1 = k_1
        self.k_2 = k_2
        self.risk_threshold = cost_threshold
        self.safe_distance = safe_distance
        self.x = 1e-8 # TODO: center_in_world_m
        self.y = 1e-8 # TODO: center_in_world_m
        self.xc = 1e-8
        self.yc = 1e-8
        self.xGrid = [] 
        self.yGrid = [] # TODO
        self.zOfGaussian = []
        self.mexp1 = 1e-8
        self.mexp2 = 1e-8
        self.delta = 1e-8 # assuming no steering angle at the beginning
        self.v = 1e-8 # TODO: from dataset obtain ego car's velocity
        self.phiv = 1e-8 # TODO: from dataset obtain ego car's heading [rad]
        self.t_la = 3.5 # [s] DEPRECATED!
        self.d_la = 1e-8 # TODO: * render_context.pixel_size_m
        self.L = 1.6 # [m] distance between chasis TODO: should be the same as bounding box of the vehicle
        self.R = 1e-8
        self.dt = 0.1 # [s] simulation time step

    def gaussian3DTorusDelta(self):
        self.delta = 1e-8 if np.abs(self.delta) < 1e-8 else self.delta

    def gaussian3DTorusPhiv(self):
        numOf2Pi = np.abs(self.phiv / (2 * np.pi))
        self.phiv = np.remainder(numOf2Pi * 2 * np.pi + self.phiv, 2 * np.pi)

    # # Deprecated 
    # def gaussian3DTorusDla(self):
    #     self.d_la = self.v * self.t_la
    # for optimization
    def gaussian3DTorusDla(self, v=None):
        if (v == None):
            self.d_la = self.safe_distance + self.v * self.t_la # 10 = 2 (0.5 car length) + 8 (minimal safety distance)
        else:
            self.d_la = self.safe_distance + v * self.t_la

    def gaussian3DTorusMeshgrid(self):
        X = np.arange(self.obj_map[1].size)
        Y = np.arange(self.obj_map[0].size)
        Y = Y[::-1] # reverse y to be consistent with XOY coordinates convention
        self.xGrid, self.yGrid = np.meshgrid(X, Y)

        self.arc_len = np.zeros(shape=self.obj_map.shape, dtype=np.float32)
        self.hOfGaussian = np.zeros(shape=self.obj_map.shape, dtype=np.float32)
        self.innSigOfGaussian = np.zeros(shape=self.obj_map.shape, dtype=np.float32)
        self.outSigOfGaussian = np.zeros(shape=self.obj_map.shape, dtype=np.float32)
        self.zOfGaussian = np.zeros(shape=self.obj_map.shape, dtype=np.float32)
        self.sub_map = np.zeros(shape=self.obj_map.shape, dtype=np.float32)

    # def gaussian3DTorusR(self):
    #     self.R = np.abs(self.L / np.tan(self.delta))
    # for optimization
    def gaussian3DTorusR(self, delta=None):
        if (delta == None):
            self.R = np.abs(self.L / np.tan(self.delta))
        else:
            delta = 1e-8 if np.abs(delta) < 1e-8 else delta
            self.R = np.abs(self.L / np.tan(delta))

    def gaussian3DTorusXcyc(self):
        # phil is the angle determined by vehicle's rotation direction
        if (self.delta > 0):
            phil = self.phiv + np.pi / 2
        
        else:
            phil = self.phiv - np.pi / 2
        
        self.xc = self.R * np.cos(phil) + self.x
        self.yc = self.R * np.sin(phil) + self.y
    
    def gaussian3DTorusMexp(self):
        self.mexp1 = self.m + self.k_1 * self.delta
        self.mexp2 = self.m + self.k_2 * self.delta

    def gaussian3DTorusArclen(self):
        theta1 = np.remainder(2 * np.pi + np.arctan2(self.y - self.yc, self.x - self.xc), 2 * np.pi)
        theta2 = np.remainder(2 * np.pi + np.arctan2(self.yGrid - self.yc, self.xGrid - self.xc), 2 * np.pi)
        theta = np.sign(self.delta) * (theta2 - theta1)
        self.arc_len = self.R * theta

    def gaussian3DTorusA(self):
        self.hOfGaussian = self.p * (self.arc_len - self.d_la) * (self.arc_len - self.d_la)
        self.hOfGaussian[self.arc_len > self.d_la] = 0
        self.hOfGaussian[self.arc_len < 0.0] = 0

    def gaussian3DTorusSigma(self):
        self.innSigOfGaussian = self.mexp1 * self.arc_len + self.c
        self.outSigOfGaussian = self.mexp2 * self.arc_len + self.c

    def gaussian3DTorusZ(self):
        dist2R = np.sqrt(np.square(self.xGrid - self.xc) + np.square(self.yGrid - self.yc))
        in_flag = (1 - np.sign(dist2R - self.R)) / 2
        out_flag = (1 + np.sign(dist2R - self.R)) / 2

        z_in = self.hOfGaussian * in_flag * np.exp(-np.square(dist2R - self.R) / 2 / np.square(self.innSigOfGaussian))
        z_out = self.hOfGaussian * out_flag * np.exp(-np.square(dist2R - self.R) / 2 / np.square(self.outSigOfGaussian))
        self.zOfGaussian = z_in + z_out

    def perceivedRisk(self) -> float:
        return np.sum(self.obj_map * self.zOfGaussian)

    def overallProcess(self) -> float:
        self.gaussian3DTorusDelta()
        self.gaussian3DTorusPhiv()
        self.gaussian3DTorusDla()
        self.gaussian3DTorusR()
        self.gaussian3DTorusMeshgrid()
        self.gaussian3DTorusXcyc()
        self.gaussian3DTorusMexp()
        self.gaussian3DTorusArclen()
        self.gaussian3DTorusA()
        self.gaussian3DTorusSigma()
        self.gaussian3DTorusZ()
        return self.perceivedRisk()

    def optimizeSteering(self, delta: float) -> float:
        self.gaussian3DTorusDelta()
        self.gaussian3DTorusPhiv()
        self.gaussian3DTorusDla()
        self.gaussian3DTorusR(delta)
        self.gaussian3DTorusMeshgrid()
        self.gaussian3DTorusXcyc()
        self.gaussian3DTorusMexp()
        self.gaussian3DTorusArclen()
        self.gaussian3DTorusA()
        self.gaussian3DTorusSigma()
        self.gaussian3DTorusZ()
        return self.perceivedRisk()

    def optimizeSteeringCt(self, delta: float) -> float:
        temp = self.optimizeSteering(delta)
        return np.square(temp - self.risk_threshold)

    def optimizeVelocity(self, vel: float) -> float:
        self.gaussian3DTorusDelta()
        self.gaussian3DTorusPhiv()
        self.gaussian3DTorusDla(vel)
        self.gaussian3DTorusR()
        self.gaussian3DTorusMeshgrid()
        self.gaussian3DTorusXcyc()
        self.gaussian3DTorusMexp()
        self.gaussian3DTorusArclen()
        self.gaussian3DTorusA()
        self.gaussian3DTorusSigma()
        self.gaussian3DTorusZ()
        return self.perceivedRisk()

    def optimizeVelocityCt(self, vel: float) -> float:
        temp = self.optimizeVelocity(vel)
        return np.square(temp - 1.1 * self.risk_threshold) # 10% above Ct to mitigate ocsillation

    def carKinematics(self):
        self.x = self.x + self.v * np.cos(self.phiv) * self.dt
        self.y = self.y + self.v * np.sin(self.phiv) * self.dt
        self.phiv = self.phiv + self.v / self.L * np.tan(self.delta) * self.dt

# driver controller 2: Lane keeping (car following)
def driverController2(curr_risk: float, ego: DRFModel, v_des: float, Ct: float) -> Tuple[float, float]:

    k_v = 0.025 # gain of vehicle's speed-up/down, could be different for normal or sport driving
    dt = 0.1 # [s] step time
    dvMax = 4 * dt # [m/s^2] Vehicle max decel and acceleration
    dvstepMax = 20 * dt # For fminbnd search
    risk_threshold = Ct

    desired_vel = v_des
    curr_vel = ego.v
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
        print("Error at driver controller: no situations match!")
    print("Next steering = ", next_steering)
    print("Next Velocity = ", next_vel)