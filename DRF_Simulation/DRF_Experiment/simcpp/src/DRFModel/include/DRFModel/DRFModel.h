#ifndef DRFMODEL_H
#define DRFMODEL_H

#include <prescan/api/roads/types/Road.hpp>
#include "prescan/api/experiment/Experiment.hpp"
#include "prescan/api/types/WorldObject.hpp"
#include <prescan/api/air/AirFunctions.hpp>
#include <prescan/api/air/AirSensor.hpp>
#include "prescan/api/Vehicledynamics.hpp"
#include "prescan/api/vehicledynamics/AmesimPreconfiguredDynamics.hpp"
#include "prescan/api/Lms.hpp"
#include "prescan/api/Trajectory.hpp"
#include "prescan/api/Utils.hpp"

#include <prescan/sim/ISimulationModel.hpp>
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

#include <tuple>
#include <vector>
#include <matplot/matplot.h>
#include <Eigen/Dense>
#include <valarray>

using namespace Eigen;

class DRFModel { 
private:
    /* DRF Parameters (Default values from paper)
    ** p = 0.0064 // "steepness of the parabola" -> Gaussian's height
    ** t_la = 3.5 // [s] "look-ahead time"
    ** c = 0.5 // "a quarter of car width, as +- 2sigma takes up 95% of Gaussian" -> Gaussian's width
    ** m = 0.001 // "slope of widening" -> Gaussian's width
    ** k_1, k_2 = 0, 1.3823 // "parameter for inner/outer edges for the DRF" -> Gaussian's width
    */
    double p_ = 0.0064;
    double t_la_ = 2.0;
    double c_ = 0.5;
    double m_ = 0.001;
    double k_1_ = 0;
    double k_2_ = 1.3823;

    /* Simulation Parameters (Default values from the MATLAB script)
    */
    double gridResolution = 1.0; // [m]
    double L = 1.9887; // [m] vehicle wheel base
    double vehicleWidth = 2; // [m] vehicle width
    double dt = 0.05; // [s] simulation step time

public:
    /* Intermediate parameters for computing the cost map
    */
    double x_, y_; // [m] vehicle position in global coordinates
    double phiv_; // [rad] vehicle heading (yaw) 
    double delta_; // [rad] steering angle ??? May not have this, maybe use v and phiv ?
    double yawRate_; // [rad/s]
    double velocity_; // [m/s]
    double d_la; // [m] look-ahead distance
    double R; // [m] radius of the vehicle's predicted arc path in the preview time
    double xc_, yc_; // [m] global coordinates of the center of the very arc
    std::vector<double> xGrid, yGrid; // [m] x and y coordinates for the meshgrids
    std::valarray<double> X, Y;

    double mexp1, mexp2; // mexp = m_ + k_i * |delta|
    std::vector<std::vector<double>> arcLength; // 2D vector contains all (arc between vehicle and meshgrids)s' length
    std::valarray<double> arcLen;
    std::vector<std::vector<double>> heightOfGaussian; // 2D vector contains all the height of all grids' cross section
    std::valarray<double> hOfGaussian;
    std::vector<std::vector<double>> innerSigmaOfGaussian; // 2D vector contains all inner sigma of all grids' cross section
    std::valarray<double> innerSigOfGaussian;
    std::vector<std::vector<double>> outerSigmaOfGaussian; // 2D vector contains all outer sigma of all grids' cross section
    std::valarray<double> outerSigOfGaussian;
    std::vector<std::vector<double>> zOfGaussian; // 2D vector contains z(x, y) of all grids
    std::valarray<double> ziOfGaussian;
    std::vector<std::vector<double>> roadCost; // 2D vector contains the road cost of all grids
    std::valarray<double> rCost;
    std::vector<std::vector<double>> obsCost; // 2D vector contains the obstacle cost of all grids
    std::valarray<double> oCost;
    std::vector<std::vector<double>> totalCost; // 2D vector contains the (road + obstacle) cost of all grids
    std::valarray<double> tCost;

    prescan::api::roads::types::Road road_; // road object in Prescan simulation
    prescan::sim::SelfSensorUnit* m_egoSelfUnit_;
    std::vector<prescan::sim::SelfSensorUnit*> m_targetSelfUnitList_;
    std::vector<std::string> targetNameList_;
    double costThreshold = 3000; // cost threshold for control
    double perceivedRisk_;
    DRFModel() = default;
    DRFModel(double p, double t_la, double c, double m, double k_1, double k_2);

    /* Gaussian 3D torus Methods: Drivers' subjective view of probability of ego vehicles' position 
    **                            within the look-ahead time
    */
    // Read all accessible parameters before computing costs.
    void setModelParams(prescan::sim::SelfSensorUnit* m_egoSelfUnit, std::vector<prescan::sim::SelfSensorUnit*> m_targetSelfUnitList, 
                        prescan::api::roads::types::Road road, std::vector<std::string> targetNameList);

    // Car kinematics
    void carKinematics();

    // In MATLAB script, it is used to filter delta (steering angle) below 1e-8
    void gaussian3DTorusDelta(double delta);
    void gaussian3DTorusDelta();

    // In MATLAB script, it is used to keep the heading phiv (yaw) in range 0 -- 2pi
    void gaussian3DTorusPhiv();

    // Calculate look-ahead distance
    void gaussian3DTorusDla();

    // Determine the radius of the arc (R_car) which the car will go in the preview time
    void gaussian3DTorusR();
    void gaussian3DTorusR(double yawRate);

    // Determine the radius of the arc (R_car) which the car will go in the preview time
    // Use velocity and yaw rate
    // Note: Assume constant velocity and yaw rate in the preview time
    void gaussian3DTorusR2();
    void gaussian3DTorusR2(double yawRate);


    // Calculate the global coordinates of the center of the arc in the preview time
    void gaussian3DTorusXcyc(double, double);
    void gaussian3DTorusXcyc();

    // Calculate the global coordinates of the center of the arc in the preview time
    // Use yaw rate to determine the rotational direction
    // Note: Assume yaw rate to be positive when steering is positive
    void gaussian3DTorusXcyc2();

    // Calculate the discrete grid around the vehicle position
    void gaussian3DTorusMeshgrid();

    // Calculate mexp, where sigma = mexp * s + cexp
    void gaussian3DTorusMexp();

    // Calculate mexp, where sigma = mexp * s + cexp
    // Note: This function use yaw rate to compute mexp
    void gaussian3DTorusMexp2();

    // Calculate the arc length between the car's position and meshgrids to get the height of Gaussian in
    // the torus' section later
    void gaussian3DTorusArclen();

    // Calculate the arc length between the car's position and meshgrids to get the height of Gaussian in
    // the torus' section later
    // Note: This function uses yaw rate the determine rotation direction
    void gaussian3DTorusArclen2();

    // Calculate the height of Gaussian in every cross section of the torus
    void gaussian3DTorusA();

    // Calculate the sigma of Gaussian in every section
    void gaussian3DTorusSigma();

    // Separate the meshgrids into two categories: inside/outside the turning circle
    // Then compute the inner/outer Gaussian separately, finally add both together
    // to obtain the Gaussian3Dtorus distribution.
    void gaussian3DTorusZ();


    /* Gaussian 3D torus Methods: Objective view of events' consequences 
    */

    // Generate cost from the road network only (not including obstacles)
    // MATLAB Script:
    // if meshgrids are on the road, 0 cost.
    // else, 500 cost
    void generateCircuitRoadSoloCost();
    bool isPointOnStraightSection(double x, double y);
    bool isPointOnArcSection(double x, double y);

    // Obstacle cost
    // Cost values from paper: if grid in obstacle, cost += 2500
    //                         else, cost += 0
    // Input: All other vehicle (obstacle)'s global coordinates x and y
    void obstacleCost(double obstacleX, double obstacleY);
    void totalObsCost();

    // Environment cost map (final map)= (road cost + obstacle cost) * zOfGaussian
    // Output: Scaler value of driver's perceived risk
    double environmentCost();

    // Visualise the total cost map as a surface plot
    void visualiseCost();

    // Save cost map as csv file
    void saveCostMap();
    void saveSubjectiveMap();
    void saveRoadCost();
    void saveObstacleCost();

    // Save trajectory of ego vehicle in a csv file
    void saveEgoTrajectory();
};

#endif
