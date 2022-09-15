#define _USE_MATH_DEFINES

#include <iostream>
#include <prescan/sim/ISimulationModel.hpp>
#include "DRFSimulationModel.hpp"
#include "DRFModel/DRFModel.h"
#include <matplot/matplot.h>
#include <math.h>
#include "dlib/optimization.h" 
#include <Eigen/Dense>

#include <prescan/sim/ISimulationModel.hpp>
#include <prescan/api/roads/types/Road.hpp>
#include <prescan/api/roads/RoadsFunctions.hpp>
#include "prescan/api/experiment/Experiment.hpp"
#include "prescan/api/types/WorldObject.hpp"
#include <prescan/api/air/AirFunctions.hpp>
#include <prescan/api/air/AirSensor.hpp>
#include "prescan/api/Vehicledynamics.hpp"
#include "prescan/api/vehicledynamics/AmesimPreconfiguredDynamics.hpp"
#include "prescan/api/Lms.hpp"
#include "prescan/api/Trajectory.hpp"
#include "prescan/api/Utils.hpp"

#include "prescan/sim/Simulation.hpp"
#include "prescan/sim/ManualSimulation.hpp"
#include "prescan/sim/StateActuatorUnit.hpp"
#include "prescan/sim/SelfSensorUnit.hpp"
#include "prescan/sim/AirSensorUnit.hpp"
#include "prescan/sim/AmesimPreconfiguredDynamicsUnit.hpp"
#include "prescan/sim/LmsSensorUnit.hpp"
#include "prescan/sim/PathUnit.hpp"
#include "prescan/sim/SpeedProfileUnit.hpp"
#include <prescan/sim/CameraSensorUnit.hpp>
#include <prescan/api/viewer/Viewer.hpp>
#include <prescan/api/viewer/ViewerFunctions.hpp>

#include <Windows.h>
/* Remove if already defined */
typedef long long int64; typedef unsigned long long uint64;
/* A global function for computing code snippets' run time
*/
uint64 GetTimeMs64() {
    /* Windows */
    FILETIME ft;
    LARGE_INTEGER li;

    /* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
    * to a LARGE_INTEGER structure. */
    GetSystemTimeAsFileTime(&ft);
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;

    uint64 ret = li.QuadPart;
    ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
    ret /= 10000; /* From 100 nano seconds (10^-7) to 1 millisecond (10^-3) intervals */

    return ret;
}

DRFModel ego; // A global variable so class: "DRFSimulationModel" 's static member optimization function can access vehicle params

class DRFSimulationModel : public prescan::sim::ISimulationModel {
public:
    void updateTrajectory() const {
        // Copy data between simulation units
        m_egoPathUnit->motionInput() = m_egoSpeedProfileUnit->motionOutput();
        m_egoActuator->stateActuatorInput() = m_egoPathUnit->stateActuatorOutput();
        /*
        m_targetPathUnit->motionInput() = m_targetSpeedProfileUnit->motionOutput();
        m_targetActuator->stateActuatorInput() = m_targetPathUnit->stateActuatorOutput();*/
        
        for (int i = 0; i < targetNameList.size(); i++) {
            //auto target = experiment.getObjectByName<prescan::api::types::WorldObject>(targetNameList[i]);
            if (m_targetPathUnitList[i] != nullptr) {
            m_targetPathUnitList[i]->motionInput() = m_targetSpeedProfileUnitList[i]->motionOutput();
            m_targetActuatorList[i]->stateActuatorInput() = m_targetPathUnitList[i]->stateActuatorOutput();
            }
        }
    }

    void updateTargetTrajectory() const {
        for (int i = 0; i < targetNameList.size(); i++) {
            //auto target = experiment.getObjectByName<prescan::api::types::WorldObject>(targetNameList[i]);
            if (m_targetPathUnitList[i] != nullptr) {
            m_targetPathUnitList[i]->motionInput() = m_targetSpeedProfileUnitList[i]->motionOutput();
            m_targetActuatorList[i]->stateActuatorInput() = m_targetPathUnitList[i]->stateActuatorOutput();
            }
        }
    }

    bool checkTermination(float simTime) {
        /*** Check if the ego has reached a terminal state to stop simulation ***/
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();

        /* If car has stopped */
        if (selfOutput.Velocity <= 0.01) {
            return true;
        }
        /* If max time is elapsed */
        else if (simTime >= 40.0) {
            return true;
        }
        else {
            return false;
        }
    }

    void registerSimulationUnits(const prescan::api::experiment::Experiment& experiment, 
        prescan::sim::ISimulation* simulation) override {

        // Data Model API: find objects by name
        //freopen("output2.txt","w",stdout);
        std::cout << "Register vehicles and trajectory." << std::endl;
        auto roadList = prescan::api::roads::getRoads(experiment);
        std::cout << "Roadlist size =  " << roadList.size() << std::endl;
        road = roadList[0];
        /*
        auto setList = prescan::api::roads::getSettings(experiment);
        std::cout << "Setlist size =  " << setList << std::endl;
        auto obList = experiment.objects();
        std::cout << "objlist size =  " << obList.size() << std::endl;
        */

        auto ego = experiment.getObjectByName<prescan::api::types::WorldObject>("Ego");
        if (prescan::api::trajectory::hasActiveTrajectory(ego)) {
            auto egoTrajectory = prescan::api::trajectory::getActiveTrajectory(ego);
            m_egoSpeedProfileUnit = prescan::sim::registerUnit<prescan::sim::SpeedProfileUnit>(simulation, 
                                                                                            egoTrajectory.speedProfile());
            m_egoPathUnit = prescan::sim::registerUnit<prescan::sim::PathUnit>(simulation, egoTrajectory.path(), ego);
        }
        else {
            m_egoSpeedProfileUnit = { nullptr };
            m_egoPathUnit = { nullptr };
        }
        // Simulation API: register simulation units for the ego car
        m_egoActuator = prescan::sim::registerUnit<prescan::sim::StateActuatorUnit>(simulation, ego);
        m_egoSelfUnit = prescan::sim::registerUnit<prescan::sim::SelfSensorUnit>(simulation, ego);

        // Needs to be filled in according to the experiment file
        // Note: Is there any function to access all vehicles' names directly?
        /*std::vector<std::string> targetNameList = {"Toyota_Previa_1", "BMW_X5_1", "Citroen_C3_1",
                                                   "Toyota_Yaris_1", "Audi_A8_1", "Toyota_Previa_2"
                                                   "Audi_A8_2", "BMW_X5_2", "Citroen_C3_2", 
                                                   "Toyota_Previa_3", "Citroen_C3_3", "Citroen_C3_4",
                                                   "Toyota_Previa_4", "Toyota_Previa_2", "Audi_A8_3"};*/

        // Simulation API: register simulation units for targets
        /*
        auto target = experiment.getObjectByName<prescan::api::types::WorldObject>("Target1");
        auto targetTrajectory = prescan::api::Trajectory::getActiveTrajectory(target);
        m_targetActuator = prescan::sim::registerUnit<prescan::sim::StateActuatorUnit>(simulation, target);
        m_targetSelfUnit = prescan::sim::registerUnit<prescan::sim::SelfSensorUnit>(simulation, target);
        m_targetSpeedProfileUnit = prescan::sim::registerUnit<prescan::sim::SpeedProfileUnit>(simulation, 
                                                                                            targetTrajectory.speedProfile());
        m_targetPathUnit = prescan::sim::registerUnit<prescan::sim::PathUnit>(simulation, targetTrajectory.path(), target);
        */
        m_targetActuatorList.resize(targetNameList.size());
        m_targetSelfUnitList.resize(targetNameList.size());
        //egoDRF.m_targetSelfUnitList_.resize(targetNameList.size());//
        m_targetSpeedProfileUnitList.resize(targetNameList.size());
        m_targetPathUnitList.resize(targetNameList.size());
        for (int i = 0; i < targetNameList.size(); i++) {
            auto target = experiment.getObjectByName<prescan::api::types::WorldObject>(targetNameList[i]);
            m_targetActuatorList[i] = prescan::sim::registerUnit<prescan::sim::StateActuatorUnit>(simulation, target);

            m_targetSelfUnitList[i] = prescan::sim::registerUnit<prescan::sim::SelfSensorUnit>(simulation, target);
            if (prescan::api::trajectory::hasActiveTrajectory(target)) {
                auto targetTrajectory = prescan::api::Trajectory::getActiveTrajectory(target);
                m_targetSpeedProfileUnitList[i] = prescan::sim::registerUnit<prescan::sim::SpeedProfileUnit>(simulation, 
                                                                                            targetTrajectory.speedProfile());
                m_targetPathUnitList[i] = prescan::sim::registerUnit<prescan::sim::PathUnit>(simulation, targetTrajectory.path(), target);
            }
            else {
                 m_targetSpeedProfileUnitList[i] = { nullptr };
                 m_targetPathUnitList[i] = { nullptr };
            }
        }

        // get viewer
        auto viewerList = prescan::api::viewer::getViewers(experiment);
        std::cout<< "viewerList size = "<< viewerList.size()<<std::endl;
        viewer = viewerList[1];
        auto egoCamList = prescan::api::CameraSensor::getAttachedTo(ego);
        std::cout<< "egoCameraList size = "<< egoCamList.size()<<std::endl;
        viewer.assignCamera(egoCamList[0]);

        std::cout << "register complete"  << std::endl;
    }

    void initialize(prescan::sim::ISimulation* simulation) override {
        // call state actuators input to get consistent simulation results
        //freopen("output2.txt","w",stdout);
        std::cout << "initialize"  << std::endl;

        /* Initialize the ego vehicle's initial conditions below
        ** As for the other vehicles ("targets"), they do not need this because
        */
        
        m_egoActuator->stateActuatorInput().VelocityX = 10;

        //m_egoActuator->stateActuatorInput();
        /*
        m_targetActuator->stateActuatorInput();
        */
        for (int i = 0; i < targetNameList.size(); i++) {
            m_targetActuatorList[i]->stateActuatorInput().VelocityX = 8;
        }
        ::ego.setModelParams(m_egoSelfUnit, m_targetSelfUnitList, road, targetNameList);
        egoDRF.setModelParams(m_egoSelfUnit, m_targetSelfUnitList, road, targetNameList);
        std::cout << "initialize complete"  << std::endl;
    }

    double optimizeAngularVelocity(double yawRate) {
        // pull self sensor output
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();
        //DRFModel egoDRF;
        egoDRF.gaussian3DTorusPhiv();
        egoDRF.gaussian3DTorusDla();
        egoDRF.gaussian3DTorusR(); // Trying to optimize yaw rate here
        egoDRF.gaussian3DTorusXcyc2();
        egoDRF.gaussian3DTorusMeshgrid();
        egoDRF.gaussian3DTorusMexp2();
        egoDRF.gaussian3DTorusArclen2();
        egoDRF.gaussian3DTorusA();
        egoDRF.gaussian3DTorusSigma();
        egoDRF.gaussian3DTorusZ();
        egoDRF.generateCircuitRoadSoloCost();

        for (int i = 0; i < targetNameList.size(); i++) {
            auto targetOutput = m_targetSelfUnitList[i]->selfSensorOutput();
            egoDRF.obstacleCost(targetOutput.PositionX, targetOutput.PositionY);
        }
        return egoDRF.environmentCost();
    }

    static double optimizeAngularVelocity2(double delta) {
        // pull self sensor output
        //const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();
        
        ego.gaussian3DTorusDelta(delta); // Trying to optimize steering angle here
        //ego.gaussian3DTorusPhiv();
        //ego.gaussian3DTorusDla();
        ego.gaussian3DTorusR(); 
        ego.gaussian3DTorusXcyc();
        //ego.gaussian3DTorusMeshgrid();
        ::ego.gaussian3DTorusMexp();
        ::ego.gaussian3DTorusArclen();
        ::ego.gaussian3DTorusA();
        ::ego.gaussian3DTorusSigma();
        ::ego.gaussian3DTorusZ();
        /* No need to compute objective map again in the same step. */
        //ego.generateCircuitRoadSoloCost();
        //ego.totalObsCost();
        return ::ego.environmentCost();
    }

    static double optimizeAngularVelocityCt(double delta) {
        double temp = optimizeAngularVelocity2(delta);
        return (temp - ::ego.costThreshold) * (temp - ::ego.costThreshold);
    }

    inline double lagrange_poly_min_extrap (
        double p1, 
        double p2,
        double p3,
        double f1,
        double f2,
        double f3
    )
    {
        DLIB_ASSERT(p1 < p2 && p2 < p3 && f1 >= f2 && f2 <= f3,
                     "   p1: " << p1 
                     << "   p2: " << p2 
                     << "   p3: " << p3  
                     << "   f1: " << f1 
                     << "   f2: " << f2 
                     << "   f3: " << f3);

        // This formula is out of the book Nonlinear Optimization by Andrzej Ruszczynski.  See section 5.2.
        double temp1 =    f1*(p3*p3 - p2*p2) + f2*(p1*p1 - p3*p3) + f3*(p2*p2 - p1*p1);
        double temp2 = 2*(f1*(p3 - p2)       + f2*(p1 - p3)       + f3*(p2 - p1) );

        if (temp2 == 0)
        {
            return p2;
        }

        const double result = temp1/temp2;

        // do a final sanity check to make sure the result is in the right range
        if (p1 <= result && result <= p3)
        {
            return result;
        }
        else
        {
            return std::min(std::max(p1,result),p3);
        }
    }

    double find_min_single_variable (
        double (*f)(double),
        double& starting_point,
        const double begin = -1e200,
        const double end = 1e200,
        const double eps = 1e-3,
        const long max_iter = 100,
        const double initial_search_radius = 1
    )
    {
        DLIB_CASSERT( eps > 0 &&
                      max_iter > 1 &&
                      begin <= starting_point && starting_point <= end && 
                      initial_search_radius > 0,
                      "eps: " << eps
                      << "\n max_iter: "<< max_iter 
                      << "\n begin: "<< begin 
                      << "\n end:   "<< end 
                      << "\n starting_point: "<< starting_point 
                      << "\n initial_search_radius: "<< initial_search_radius 
        );

        //DRFSimulationModel ob;
        double search_radius = initial_search_radius;

        double p1=0, p2=0, p3=0, f1=0, f2=0, f3=0;
        long f_evals = 1;

        if (begin == end)
        {
            return f(starting_point);
        }

        using std::abs;
        using std::min;
        using std::max;

        // find three bracketing points such that f1 > f2 < f3.   Do this by generating a sequence
        // of points expanding away from 0.   Also note that, in the following code, it is always the
        // case that p1 < p2 < p3.



        // The first thing we do is get a starting set of 3 points that are inside the [begin,end] bounds
        p1 = max(starting_point-search_radius, begin);
        p3 = min(starting_point+search_radius, end);
        f1 = f(p1);
        f3 = f(p3);

        if (starting_point == p1 || starting_point == p3)
        {
            p2 = (p1+p3)/2;
            f2 = f(p2);
        }
        else
        {
            p2 = starting_point;
            f2 = f(starting_point);
        }

        f_evals += 2;

        // Now we have 3 points on the function.  Start looking for a bracketing set such that
        // f1 > f2 < f3 is the case.
        while ( !(f1 > f2 && f2 < f3))
        {
            // check for hitting max_iter or if the interval is now too small
            if (f_evals >= max_iter)
            {
                /*throw optimize_single_variable_failure(
                "The max number of iterations of single variable optimization have been reached\n"
                "without converging.");*/
                std::cout<<"The max number of iterations of single variable optimization have been reached\n"
                        "without converging."<<std::endl;
            }
            if (p3-p1 < eps)
            {
                if (f1 < min(f2,f3)) 
                {
                    starting_point = p1;
                    return f1;
                }

                if (f2 < min(f1,f3)) 
                {
                    starting_point = p2;
                    return f2;
                }

                starting_point = p3;
                return f3;
            }
            
            // If the left most points are identical in function value then expand out the
            // left a bit, unless it's already at bound or we would drop that left most
            // point anyway because it's bad.
            if (f1==f2 && f1<f3 && p1!=begin)
            {
                p1 = max(p1 - search_radius, begin);
                f1 = f(p1);
                ++f_evals;
                search_radius *= 2;
                continue;
            }
            if (f2==f3 && f3<f1 && p3!=end)
            {
                p3 = min(p3 + search_radius, end);
                f3 = f(p3);
                ++f_evals;
                search_radius *= 2;
                continue;
            }


            // if f1 is small then take a step to the left
            if (f1 <= f3)
            { 
                // check if the minimum is butting up against the bounds and if so then pick
                // a point between p1 and p2 in the hopes that shrinking the interval will
                // be a good thing to do.  Or if p1 and p2 aren't differentiated then try and
                // get them to obtain different values.
                if (p1 == begin || (f1 == f2 && (end-begin) < search_radius ))
                {
                    p3 = p2;
                    f3 = f2;

                    p2 = (p1+p2)/2.0;
                    f2 = f(p2);
                }
                else
                {
                    // pick a new point to the left of our current bracket
                    p3 = p2;
                    f3 = f2;

                    p2 = p1;
                    f2 = f1;

                    p1 = max(p1 - search_radius, begin);
                    f1 = f(p1);

                    search_radius *= 2;
                }

            }
            // otherwise f3 is small and we should take a step to the right
            else 
            {
                // check if the minimum is butting up against the bounds and if so then pick
                // a point between p2 and p3 in the hopes that shrinking the interval will
                // be a good thing to do.  Or if p2 and p3 aren't differentiated then try and
                // get them to obtain different values.
                if (p3 == end || (f2 == f3 && (end-begin) < search_radius))
                {
                    p1 = p2;
                    f1 = f2;

                    p2 = (p3+p2)/2.0;
                    f2 = f(p2);
                }
                else
                {
                    // pick a new point to the right of our current bracket
                    p1 = p2;
                    f1 = f2;

                    p2 = p3;
                    f2 = f3;

                    p3 = min(p3 + search_radius, end);
                    f3 = f(p3);

                    search_radius *= 2;
                }
            }

            ++f_evals;
        }


        // Loop until we have done the max allowable number of iterations or
        // the bracketing window is smaller than eps.
        // Within this loop we maintain the invariant that: f1 > f2 < f3 and p1 < p2 < p3
        const double tau = 0.1;
        while( f_evals < max_iter && p3-p1 > eps)
        {
            double p_min = lagrange_poly_min_extrap(p1,p2,p3, f1,f2,f3);


            // make sure p_min isn't too close to the three points we already have
            if (p_min < p2)
            {
                const double min_dist = (p2-p1)*tau;
                if (abs(p1-p_min) < min_dist) 
                {
                    p_min = p1 + min_dist;
                }
                else if (abs(p2-p_min) < min_dist)
                {
                    p_min = p2 - min_dist;
                }
            }
            else
            {
                const double min_dist = (p3-p2)*tau;
                if (abs(p2-p_min) < min_dist) 
                {
                    p_min = p2 + min_dist;
                }
                else if (abs(p3-p_min) < min_dist)
                {
                    p_min = p3 - min_dist;
                }
            }

            // make sure one side of the bracket isn't super huge compared to the other
            // side.  If it is then contract it.
            const double bracket_ratio = abs(p1-p2)/abs(p2-p3);
            // Force p_min to be on a reasonable side.  But only if lagrange_poly_min_extrap()
            // didn't put it on a good side already.
            if (bracket_ratio >= 10)
            { 
                if (p_min > p2)
                    p_min = (p1+p2)/2;
            }
            else if (bracket_ratio <= 0.1) 
            {
                if (p_min < p2)
                    p_min = (p2+p3)/2;
            }


            const double f_min = f(p_min);


            // Remove one of the endpoints of our bracket depending on where the new point falls.
            if (p_min < p2)
            {
                if (f1 > f_min && f_min < f2)
                {
                    p3 = p2;
                    f3 = f2;
                    p2 = p_min;
                    f2 = f_min;
                }
                else
                {
                    p1 = p_min;
                    f1 = f_min;
                }
            }
            else
            {
                if (f2 > f_min && f_min < f3)
                {
                    p1 = p2;
                    f1 = f2;
                    p2 = p_min;
                    f2 = f_min;
                }
                else
                {
                    p3 = p_min;
                    f3 = f_min;
                }
            }


            ++f_evals;
        }

        if (f_evals >= max_iter)
        {
            /*throw optimize_single_variable_failure(
                "The max number of iterations of single variable optimization have been reached\n"
                "without converging.");*/
            std::cout<<"The max number of iterations of single variable optimization have been reached\n"
                "without converging."<<std::endl;
        }

        starting_point = p2;
        return f2;
    }

    // TODO: Update steps
    void updateEgoTrajectory(double velocity, double delta) {
        /* An example
        // pull self sensor output
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();
        double* updatedStates = new double[15];
        PRESCAN_STATEACTUATORDATA state_actuator;
        // state 0 - 2: position
        updatedStates[0] = selfOutput.PositionX + velocity * cos(selfOutput.OrientationYaw + yawRate * dt) * dt;
        updatedStates[1] = selfOutput.PositionY + velocity * sin(selfOutput.OrientationYaw + yawRate * dt) * dt;
        updatedStates[2] = selfOutput.PositionZ;
        // state 3 - 5: velocity (TODO)
        updatedStates[3] = velocity * cos(selfOutput.OrientationYaw + yawRate * dt);
        updatedStates[4] = velocity * sin(selfOutput.OrientationYaw + yawRate * dt);
        updatedStates[5] = 0;
        // state 6 - 8: acceleration
        updatedStates[6] = 0;
        updatedStates[7] = 0;
        updatedStates[8] = 0;
        // state 9 - 11: Orientation
        updatedStates[9] = selfOutput.OrientationRoll;
        updatedStates[10] = selfOutput.OrientationPitch;
        updatedStates[11] = selfOutput.OrientationYaw + yawRate * dt;
        // state 12 - 14: angular velocity
        updatedStates[12] = 0;
        updatedStates[13] = 0;
        updatedStates[14] = yawRate;

        state_actuator.PositionX = updatedStates[0];
        state_actuator.PositionY = updatedStates[1];
        state_actuator.PositionZ = updatedStates[2];
    
        state_actuator.VelocityX = updatedStates[3];
        state_actuator.VelocityY = updatedStates[4];
        state_actuator.VelocityZ = updatedStates[5];
    
        state_actuator.AccelerationX = updatedStates[6];
        state_actuator.AccelerationY = updatedStates[7];
        state_actuator.AccelerationZ = updatedStates[8];
    
        state_actuator.OrientationRoll = updatedStates[9]; // Not sure about : /180*M_PI;
        state_actuator.OrientationPitch = updatedStates[10];
        state_actuator.OrientationYaw = updatedStates[11];
    
        state_actuator.AngularVelocityRoll = updatedStates[12];
        state_actuator.AngularVelocityPitch = updatedStates[13];
        state_actuator.AngularVelocityYaw = updatedStates[14];

        m_egoActuator->stateActuatorInput() = state_actuator;
        */
        egoDRF.carKinematics();
        egoDRF.velocity_ = velocity;
        egoDRF.delta_ = delta;
        m_egoActuator->stateActuatorInput().PositionX = egoDRF.x_;
        m_egoActuator->stateActuatorInput().PositionY = egoDRF.y_;
        m_egoActuator->stateActuatorInput().OrientationYaw = egoDRF.phiv_;
        ego = egoDRF;
    }
    
    /* Driver control: optimizing angular/linear velocity for cost smaller than threshold.
    */
    void driverControl(double currentCost) {
        // pull self sensor output
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();
        double currentVelocity = selfOutput.Velocity;
        double nextVelocity = 0;
        auto roadCoordinates = road.getRoadCoordinatesOfPoint(selfOutput.PositionX, selfOutput.PositionY);
        std::cout << "coordinate list size: " << roadCoordinates.size() << std::endl;
        //std::cout << "coordinate list 0: " << roadCoordinates[0].sideOffset() << std::endl;
        //std::cout << "coordinate list 1: " << roadCoordinates[1].sideOffset() << std::endl;

        auto roadPose = road.poseAtDistance(roadCoordinates[0].sOffset(), roadCoordinates[0].roadSide(), roadCoordinates[0].sideOffset());
        auto roadOrient = roadPose.orientation().yaw();
        double currentAngVel = selfOutput.Yaw_rate;
        double nextAngVel = currentAngVel + k_h * (roadOrient - selfOutput.OrientationYaw);

        /* This condition generally occurs when you start the
           journey. The model speeds up at a rate proportional to (Vdes − vk). The parameter
           kv (specific for each driver) represents how aggressively the model accelerates. The
           augular velocity (steering) is determined by the heading controller (in this case, we do not change its
           current velocity).
        */
        if (currentCost <= costThreshold && currentVelocity <= desiredVelocity) {
            std::cout << "condition 1" << std::endl;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            // steering update
            double dw = fmin(abs(nextAngVel - currentAngVel), dwMax);
            nextAngVel = currentAngVel + ((nextAngVel - currentAngVel > 0) - (nextAngVel - currentAngVel < 0)) * dw;
            updateEgoTrajectory(nextVelocity, nextAngVel);
        }
        /* In this condition, the incurred risk is more
           than the threshold (Ct), and the goal of desired speed has also not been achieved. In
           this case, we first check if the steering alone can help the model reduce the risk
           below the threshold. This check is performed by using the fmin_bound function,
           which finds the angular velocity wop (within the bounds of wk − 180∘/s to wk + 180∘/s) that
           minimises the risk (Ck) assuming a speed of vk. It also calculates the risk (Cop) at
           this wop.
        */
        else if (currentCost > costThreshold && currentVelocity <= desiredVelocity) {
            // check if changing angular velocity can reduce the cost below the threshold
            double minCost = find_min_single_variable(optimizeAngularVelocity2, nextAngVel, currentAngVel - dstepMax, currentAngVel + dstepMax, 1e-3, 100, dwMax);
            if (minCost > costThreshold) {
                std::cout << "condition 2a" << std::endl;
                // steering angle update
                double dw = fmin(abs(nextAngVel - currentAngVel), dwMax);
                nextAngVel = currentAngVel + ((nextAngVel - currentAngVel > 0) - (nextAngVel - currentAngVel < 0)) * dw;
                // velocity update
                double dv = fmin(dvMax, abs(k_vc * (costThreshold - minCost))); 
                nextVelocity = currentVelocity + ((costThreshold - minCost > 0) - (costThreshold - minCost < 0)) * dv;
                updateEgoTrajectory(nextVelocity, nextAngVel);
            }
            /*  model slows down
                proportional to Cop − Ck (and not Cop − Ct) since the steering applied = wop is
                expected to reduce Ck to Cop. This is done so that we do not slow down more than
                what is required. Hence, w_k+1 = wop */
            else if (minCost <= costThreshold) {
                std::cout << "condition 2b" << std::endl;
                // velocity update
                double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
                nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
                // steering update
                double nextAngVel = currentAngVel + k_h * (roadOrient - selfOutput.OrientationYaw);
                double costCt = find_min_single_variable(optimizeAngularVelocityCt, nextAngVel, currentAngVel - dstepMax, currentAngVel + dstepMax, 1e-3, 100, dwMax);
                double dw = fmin(abs(nextAngVel - currentAngVel), dwMax);
                nextAngVel = currentAngVel + ((nextAngVel - currentAngVel > 0) - (nextAngVel - currentAngVel < 0)) * dw;
                updateEgoTrajectory(nextVelocity, nextAngVel);
            }
            else {
                std::cout << "Error in stage: condition 2" << std::endl;
            }
        }
        /* In this case the model slows down, while being
        ** steered by the heading controller since the risk is lower than the threshold and
        ** speed is higher than what is desired.
        */
        else if (currentCost <= costThreshold && currentVelocity > desiredVelocity) {
            std::cout << "condition 3" << std::endl;
            // steering update
            double dw = fmin(abs(nextAngVel - currentAngVel), dwMax);
            nextAngVel = currentAngVel + ((nextAngVel - currentAngVel > 0) - (nextAngVel - currentAngVel < 0)) * dw;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            updateEgoTrajectory(nextVelocity, nextAngVel);
        }
        /* In this case both the speed and risk are over
        ** the desired limits and hence the model slows down while steering with δop that
        ** minimises Ck
        */
        else if (currentCost > costThreshold && currentVelocity > desiredVelocity) {
            std::cout << "condition 4" << std::endl;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            // steering update
            double costCt = find_min_single_variable(optimizeAngularVelocityCt, nextAngVel, currentAngVel - dstepMax, currentAngVel + dstepMax, 1e-3, 100, dwMax);
            double dw = fmin(abs(nextAngVel - currentAngVel), dwMax);
            nextAngVel = currentAngVel + ((nextAngVel - currentAngVel > 0) - (nextAngVel - currentAngVel < 0)) * dw;
            updateEgoTrajectory(nextVelocity, nextAngVel);
        }
        else {}
        std::cout << "Next Angvel = " << nextAngVel << std::endl;
        std::cout << "Next Velocity = " << nextVelocity << std::endl;
    }

    /* Driver control 2 : optimizing velocity and steering angle for cost smaller than threshold.
    */
    void driverControl2(double currentCost) {
        // pull self sensor output
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();

        ego = egoDRF;
        double currentVelocity = ego.velocity_;
        double nextVelocity = ego.velocity_;
        std::cout << "Next V= " << nextVelocity << std::endl;
        auto roadCoordinates = road.getRoadCoordinatesOfPoint(ego.x_, ego.y_);
        std::cout << "coordinate list size: " << roadCoordinates.size() << std::endl;
        //std::cout << "coordinate list 0: " << roadCoordinates[0].sideOffset() << std::endl;
        //std::cout << "coordinate list 1: " << roadCoordinates[1].sideOffset() << std::endl;

        auto roadPose = road.poseAtDistance(roadCoordinates[0].sOffset(), roadCoordinates[0].roadSide(), roadCoordinates[0].sideOffset());
        auto roadOrient = roadPose.orientation().yaw();
        std::cout << "road orientation: " << roadOrient << std::endl;
        std::cout << "ego orientation: " << ego.phiv_ << std::endl;
        double currentSteering = ego.delta_;
        double nextSteering = currentSteering + k_h * (roadOrient - ego.phiv_);

        /* This condition generally occurs when you start the
           journey. The model speeds up at a rate proportional to (Vdes − vk). The parameter
           kv (specific for each driver) represents how aggressively the model accelerates. The
           augular velocity (steering) is determined by the heading controller (in this case, we do not change its
           current velocity).
        */
        if (currentCost <= costThreshold && currentVelocity <= desiredVelocity) {
            std::cout << "condition 1" << std::endl;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            // steering update
            double ds = fmin(abs(nextSteering - currentSteering), dsMax);
            nextSteering = currentSteering + ((nextSteering - currentSteering > 0) - (nextSteering - currentSteering < 0)) * ds;
            updateEgoTrajectory(nextVelocity, nextSteering);
        }
        /* In this condition, the incurred risk is more
           than the threshold (Ct), and the goal of desired speed has also not been achieved. In
           this case, we first check if the steering alone can help the model reduce the risk
           below the threshold. This check is performed by using the fmin_bound function,
           which finds the angular velocity wop (within the bounds of wk − 180∘/s to wk + 180∘/s) that
           minimises the risk (Ck) assuming a speed of vk. It also calculates the risk (Cop) at
           this wop.
        */
        else if (currentCost > costThreshold && currentVelocity <= desiredVelocity) {
            // check if changing angular velocity can reduce the cost below the threshold
            double minCost = find_min_single_variable(optimizeAngularVelocity2, nextSteering, currentSteering - dstepMax, currentSteering + dstepMax, 1e-3, 100, dsMax);
            if (minCost > costThreshold) {
                std::cout << "condition 2a" << std::endl;
                // steering angle update
                double ds = fmin(abs(nextSteering - currentSteering), dsMax);
                nextSteering = currentSteering + ((nextSteering - currentSteering > 0) - (nextSteering - currentSteering < 0)) * ds;
                // velocity update
                double dv = fmin(dvMax, abs(k_vc * (costThreshold - minCost))); 
                nextVelocity = currentVelocity + ((costThreshold - minCost > 0) - (costThreshold - minCost < 0)) * dv;
                updateEgoTrajectory(nextVelocity, nextSteering);
            }
            /*  model slows down
                proportional to Cop − Ck (and not Cop − Ct) since the steering applied = wop is
                expected to reduce Ck to Cop. This is done so that we do not slow down more than
                what is required. Hence, w_k+1 = wop */
            else if (minCost <= costThreshold) {
                std::cout << "condition 2b" << std::endl;
                // velocity update
                double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
                nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
                // steering update
                nextSteering = currentSteering + k_h * (roadOrient - ego.phiv_);
                double costCt = find_min_single_variable(optimizeAngularVelocityCt, nextSteering, currentSteering - dstepMax, currentSteering + dstepMax, 1e-3, 100, dsMax);
                double ds = fmin(abs(nextSteering - currentSteering), dsMax);
                nextSteering = currentSteering + ((nextSteering - currentSteering > 0) - (nextSteering - currentSteering < 0)) * ds;
                updateEgoTrajectory(nextVelocity, nextSteering);
            }
            else {
                std::cout << "Error in stage: condition 2" << std::endl;
            }
        }
        /* In this case the model slows down, while being
        ** steered by the heading controller since the risk is lower than the threshold and
        ** speed is higher than what is desired.
        */
        else if (currentCost <= costThreshold && currentVelocity > desiredVelocity) {
            std::cout << "condition 3" << std::endl;
            // steering update
            double ds = fmin(abs(nextSteering - currentSteering), dsMax);
            nextSteering = currentSteering + ((nextSteering - currentSteering > 0) - (nextSteering - currentSteering < 0)) * ds;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            updateEgoTrajectory(nextVelocity, nextSteering);
        }
        /* In this case both the speed and risk are over
        ** the desired limits and hence the model slows down while steering with δop that
        ** minimises Ck
        */
        else if (currentCost > costThreshold && currentVelocity > desiredVelocity) {
            std::cout << "condition 4" << std::endl;
            // velocity update
            double dv = fmin(dvMax, abs(k_v * (desiredVelocity - currentVelocity))); 
            nextVelocity = currentVelocity + ((desiredVelocity - currentVelocity > 0) - (desiredVelocity - currentVelocity < 0)) * dv;
            // steering update
            double costCt = find_min_single_variable(optimizeAngularVelocityCt, nextSteering, currentSteering - dstepMax, currentSteering + dstepMax, 1e-3, 100, dsMax);
            double ds = fmin(abs(nextSteering - currentSteering), dsMax);
            nextSteering = currentSteering + ((nextSteering - currentSteering > 0) - (nextSteering - currentSteering < 0)) * ds;
            updateEgoTrajectory(nextVelocity, nextSteering);
        }
        else {}
        std::cout << "Next steering = " << nextSteering << std::endl;
        std::cout << "Next Velocity = " << nextVelocity << std::endl;
    }
    

    // TODO: Control algorithms: DRF Model
    void step(prescan::sim::ISimulation* simulation) override {
        // pull self sensor output
        const PRESCAN_SELFSENSORDATA selfOutput = m_egoSelfUnit->selfSensorOutput();
        std::cout << "Self Output x =  " << selfOutput.PositionX << std::endl;
        std::cout << "Self Output y =  " << selfOutput.PositionY << std::endl;
        //std::cout << "Self Output v =  " << selfOutput.Velocity << std::endl;

        const PRESCAN_SELFSENSORDATA targetOutput = m_targetSelfUnitList[0]->selfSensorOutput();//
        std::cout << "first target Output x =  " << targetOutput.PositionX << std::endl;
        std::cout << "first target Output y =  " << targetOutput.PositionY << std::endl;
        std::cout << "first target Output v =  " << targetOutput.Velocity << std::endl;
        std::ofstream file("Target1Trajectory.csv", std::ios::app);
        file << targetOutput.PositionX << ',';
        file << targetOutput.PositionY << '\n';

        updateTargetTrajectory();

        uint64 time1 = GetTimeMs64();
        // Compute current cost
        egoDRF.gaussian3DTorusDelta(); 
        egoDRF.gaussian3DTorusPhiv();
        egoDRF.gaussian3DTorusDla();

        egoDRF.gaussian3DTorusR();
        //egoDRF.gaussian3DTorusR2();
        egoDRF.gaussian3DTorusXcyc();
        //egoDRF.gaussian3DTorusXcyc2();

        egoDRF.gaussian3DTorusMeshgrid();

        egoDRF.gaussian3DTorusMexp();
        //egoDRF.gaussian3DTorusMexp2();
        egoDRF.gaussian3DTorusArclen();
        //egoDRF.gaussian3DTorusArclen2();

        egoDRF.gaussian3DTorusA();
        egoDRF.gaussian3DTorusSigma();
        egoDRF.gaussian3DTorusZ();
        uint64 time2 = GetTimeMs64();

        egoDRF.generateCircuitRoadSoloCost();

        for (int i = 0; i < targetNameList.size(); i++) {
            auto targetOutput = m_targetSelfUnitList[i]->selfSensorOutput();
            egoDRF.obstacleCost(targetOutput.PositionX, targetOutput.PositionY);
        }
        
        double currentCost = egoDRF.environmentCost();
        uint64 time3 = GetTimeMs64();
        std::cout << "sub time =  " << time2 - time1 << std::endl;
        std::cout << "obj time =  " << time3 - time2 << std::endl;
        //egoDRF.visualiseCost();
        egoDRF.saveCostMap();
        egoDRF.saveObstacleCost();
        egoDRF.saveRoadCost();
        egoDRF.saveSubjectiveMap();
        egoDRF.saveEgoTrajectory();
        std::cout << "current cost =  " << currentCost << std::endl;
        // Compute optimized velocity and angular velocity for the next step
        driverControl2(currentCost);
    }

    void terminate(prescan::sim::ISimulation* simulation) override {}
public:
    DRFModel egoDRF;
private:
    /* Driver control parameters
    */
    double desiredVelocity = 20; // [m/s] ego car driver's desired velocity
    double costThreshold = 3000; // could be different for normal or sport driving
    double k_v = 0.14; // gain of vehicle's speed-up/down, could be different for normal or sport driving
    double k_vc = 1.5 * 1e-4; // gain of vehicle's speed-down contributed by the perceived risk (cost)
    double k_h = 0.02; // gain of the heading controller
    double dt = 0.05; // [s] Prescan default simulation time step 
    double dvMax = 4 * dt; // [m/s^2] Vehicle max decel and acceleration
    double dwMax = M_PI / 180 * 30 * dt; // [rad/dt] Note: Here assume angVel limit is 30 degree/s, be careful with unit!
    double dsMax = M_PI / 180 * 10 * dt; // [rad/dt] Note: Here assume steer limit is 2 degree/s, be careful with unit!
    double dstepMax = M_PI / 180 * 180 * dt; // [rad/dt] Note: For fminbound search

    // One DRF model for one Prescan vehicle
    prescan::api::roads::types::Road road;
    prescan::api::viewer::Viewer viewer;
    prescan::sim::StateActuatorUnit* m_egoActuator{ nullptr };
    prescan::sim::SelfSensorUnit* m_egoSelfUnit{ nullptr };


    //prescan::sim::StateActuatorUnit* m_targetActuator{ nullptr };//
    std::vector<prescan::sim::StateActuatorUnit*> m_targetActuatorList{ nullptr };
    //prescan::sim::SelfSensorUnit* m_targetSelfUnit{ nullptr };//
    std::vector<prescan::sim::SelfSensorUnit*> m_targetSelfUnitList{ nullptr };

	/* TODO: Only uncomment if applying DRF also to the target cars
    //DRFModel tarDRF; 
    */

    // Needs to be filled in according to the experiment file
    // Note: Is there any function to access all vehicles' names directly?
    const std::vector<std::string> targetNameList = {"Target1"};

    prescan::sim::PathUnit* m_egoPathUnit{ nullptr };

    //prescan::sim::PathUnit* m_targetPathUnit{ nullptr };//
    std::vector<prescan::sim::PathUnit*> m_targetPathUnitList{ nullptr };

    prescan::sim::SpeedProfileUnit* m_egoSpeedProfileUnit{ nullptr };

    //prescan::sim::SpeedProfileUnit* m_targetSpeedProfileUnit{ nullptr };//
    std::vector<prescan::sim::SpeedProfileUnit*> m_targetSpeedProfileUnitList{ nullptr };

    /* TODO: Only uncomment if planning to use sensors to get target cars' pose info.
    //prescan::sim::AirSensorUnit* m_airSensorUnit{ nullptr };
    //prescan::sim::CameraSensorUnit* m_cameraSensorUnit{ nullptr };
    */
};

PRESCAN_MAIN()
{
	const std::string pathToExperiment = path_to_pb_dir;
	const std::string pathToPb = pathToExperiment + "/DRF_Experiment.pb"; //strcur
	printf("Hello, I am DRFSimulationModel! Don't forget to start PrescanStart.exe\n");

	prescan::api::experiment::Experiment experiment(pathToPb);
    experiment.saveToFile(pathToExperiment + "/DRF_Experiment.pb");

    // create viewer
    if (prescan::api::viewer::getViewers(experiment).size() == 0) {
        auto viewer = prescan::api::viewer::createViewer(experiment);
        //viewer.assignFreeCamera();
    }
    
    std::cout << "add viewers"  << std::endl;
    //save pb file to run the simulation
    experiment.saveToFile(pathToExperiment + "/DRF_Experiment.pb");

    // Do not consider vehicle dynamics
	/* if (prescan::api::vehicledynamics::getAmesimPreconfiguredDynamics(experiment).size() == 0) {
		//Create dynamics
		auto ego = experiment.getObjectByName<prescan::api::types::WorldObject>("Audi_A8_Sedan");
		auto dynamics = prescan::api::vehicledynamics::createAmesimPreconfiguredDynamics(ego);
		dynamics.setInitialVelocity(13.9);

		//save pb file to run the simulation
		experiment.saveToFile(pathToExperiment + "/Demo_AmesimPreconfiguredDynamics.pb");
	} */

	DRFSimulationModel model;
    
	prescan::sim::ManualSimulation sim(&model);
	sim.setSimulationPath(pathToExperiment);

	sim.initialize(experiment);
    std::cout << "after init"  << std::endl;
	sim.step();

	do {
        sim.step();
	} while (!model.checkTermination(sim.getSampleTime()));

	sim.terminate();

	return 0;
}
