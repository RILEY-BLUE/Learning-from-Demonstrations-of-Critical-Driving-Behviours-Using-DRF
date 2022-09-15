/* File name: Initialisation.cpp
** Usage: This file is based on the Data Model API, which can be used to (create&modify) an experiment.
** Note: In order to simulate an experiment, the Simulation API is also required.
** Author: Yurui Du
** Date: 08/18/2021
** Version: 0.0
**
*/
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>

#include <prescan/api/Utils.hpp>
#include <prescan/api/Experiment.hpp>
#include <prescan/api/experiment/Experiment.hpp>
#include <prescan/api/roads/RoadsFunctions.hpp>
#include <prescan/api/roads/types/Road.hpp>
#include <prescan/api/camera/CameraFunctions.hpp>
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


PRESCAN_MAIN() {
    //prescan::api::experiment::Experiment experiment("D:/YURUIDU/DRF_Simulation/DRF_Experiment/DRF_Experiment.pb"); // Load experiment
    prescan::api::experiment::Experiment experiment;
    auto road = prescan::api::roads::createRoad(experiment); // Create new road

    /* Add sections to the road
    */
    double sectionLength = 100; // [m]
    road.addStraightSection(sectionLength);
    sectionLength = 50 * M_PI;
    double arcCurvature = 0.02;
    road.addArcSection(sectionLength, arcCurvature);

    /* Now add lanes to the road
    */
    auto lane1 = road.addLeftLane(3.2, 0.0, INFINITY);
    auto marker = lane1.getLaneMarker(LaneSideTypeOuter, 0.0, INFINITY);
    marker.setType(RoadMarkTypeBroken);
    auto lane2 = road.addLeftLane(3.2, 0.0, INFINITY);
    marker = lane2.getLaneMarker(LaneSideTypeOuter, 0.0, INFINITY);
    marker.setType(RoadMarkTypeSolid);
    auto lane3 = road.addRightLane(3.2, 0.0, INFINITY);
    marker = lane3.getLaneMarker(LaneSideTypeOuter, 0.0, INFINITY);
    marker.setType(RoadMarkTypeBroken);
    auto lane4 = road.addRightLane(3.2, 0.0, INFINITY);
    marker = lane4.getLaneMarker(LaneSideTypeOuter, 0.0, INFINITY);
    marker.setType(RoadMarkTypeSolid);

    double centerOfLane1 = -3.2 / 2; // 2 lanes to the right, just in the middle of one
    double centerOfLane2 = -3.2 - 3.2 / 2; // 2 lanes to the right, just in the middle of one
    double edgeOfRoadLeft = -15; 
    double edgeOfRoadRight = 15; 	

    /* Add objects to the road
    */
    auto vehicleAudi = experiment.createObject("Audi_A8_Sedan");
    vehicleAudi.setName("Ego");
    auto vehicleBMW = experiment.createObject("BMW_X5_SUV");
    vehicleBMW.setName("Target1");
    //auto vehicleFord = experiment.createObject("Ford_Fiesta_Hatchback");
    //vehicleFord.setName("Target2");

    /* Trees
    */
    for (int i = 0; i < 50; i += 1) {
        double r = -3 + (3 + 3) * rand(); // Make it random
        double xtree = 6 * i + r;
        double ytree1 = edgeOfRoadLeft + r;
        double ytree2 = edgeOfRoadRight + r;
        
        auto tree = experiment.createObject("Dogwood20y");
        tree.pose().position().setX(xtree);
        tree.pose().position().setY(ytree1);
        
        tree = experiment.createObject("Dogwood20y");
        tree.pose().position().setX(xtree);
        tree.pose().position().setY(ytree2);
    }

    /* Position the vehicles correctly
    */
    vehicleBMW.pose().position().setY(centerOfLane1);
    vehicleAudi.pose().position().setY(centerOfLane1);
    
    //vehicleFord.pose().position().setY(centerOfLane1);

    vehicleBMW.pose().position().setX(30); // [m] in front
    vehicleAudi.pose().position().setX(15); // [m] in behind

    /* Add a trajectory to the ego vehicle
    */
    std::vector<double> z = {0, 0};
    std::vector<double> y = {vehicleAudi.pose().position().y(), vehicleAudi.pose().position().y()};
    std::vector<double> x = {vehicleAudi.pose().position().x(), 100};
    auto pathAudi = prescan::api::trajectory::createFittedPath(experiment, x, y, z, 0.1);

    std::vector<double> z1;
    std::vector<double> y1;
    std::vector<double> x1;
    for (double x_ = 30; x_ < 100; x_ += 0.1) {
        x1.push_back(x_);
        y1.push_back(-1.6);
        z1.push_back(0);
    }
    /*for (double y_ = -1.7; y_ > -104.8; y_ -= 0.1) {
        double x_ = sqrt(51.6 * 51.6 - (y_ + 53.2) * (y_ + 53.2)) + 100;
        x1.push_back(x_);
        y1.push_back(y_);
        z1.push_back(0);
    }*/
    for (double y_ = -1.5; y_ < 95.2; y_ += 0.1) {
        double x_ = sqrt(48.4 * 48.4 - (y_ - 46.8) * (y_ - 46.8)) + 100;
        x1.push_back(x_);
        y1.push_back(y_);
        z1.push_back(0);
    }
    auto pathBMW = prescan::api::trajectory::createFittedPath(experiment, x1, y1, z1, 0.1);
    /*
    y = [vehicleFord.pose.position.y, vehicleFord.pose.position.y];
    x = [vehicleFord.pose.position.x, 200];
    pathFord = prescan.api.trajectory.createFittedPath(exp, x, y, z);
    */
    auto speedProfAudi = prescan::api::trajectory::createSpeedProfileOfConstantSpeed(experiment, 10);
    auto speedProfBMW = prescan::api::trajectory::createSpeedProfileOfConstantSpeed(experiment, 8);

    auto trajectoryAudi = prescan::api::trajectory::createTrajectory(vehicleAudi, pathAudi, speedProfAudi);
    auto trajectoryBMW = prescan::api::trajectory::createTrajectory(vehicleBMW, pathBMW, speedProfBMW);
    //trajectoryFord = prescan.api.trajectory.createTrajectory(vehicleFord, pathFord, speedProf);
    

    /* Add a camera to the ego vehicle
    */
    auto cameraAudi = prescan::api::camera::createCameraSensor(vehicleAudi);

    /* Change resolution of the camera, for HD
    */
    cameraAudi.setFocalLength(5);

    /* Save the experiment to a file
    */
    experiment.saveToFile("D:/YURUIDU/DRF_Simulation/DRF_Experiment/DRF_Experiment.pb");

    /* Generate the CS
    */
    //prescan::api::simulink::generate();

    /* Run the experiment
    */
    //prescan::api::simulink::run(exp, 'Regenerate', 'off', 'StopTime', '5');

    //auto airSensor = experiment.getObjectByName<prescan::api::air::AirSensor>("AIR_1");
    //std::cout << "Max detectable objects: " << airSensor.maxDetectableObjects() << std::endl;
    return 0;
}
