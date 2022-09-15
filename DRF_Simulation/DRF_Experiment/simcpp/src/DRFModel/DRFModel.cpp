/* Class name: DRFModel
** Description: This DRFModel aims to simulate human driving behaviour.
** Author: Yurui Du
** Date: 12/10/2021
** Version: 3.0
** TODO: 1. Optimize code structure, readability and style.
**       
*/

#define _USE_MATH_DEFINES

#include "DRFModel/DRFModel.h"

#include <prescan_vehicle_control_data.h>
#include <prescan/api/types/Orientation.hpp>
#include <prescan/api/roads/types/Road.hpp>

#include <math.h>
#include <matplot/matplot.h>
#include <iomanip>

using namespace Eigen;

DRFModel::DRFModel(double p, double t_la, double c, double m, double k_1, double k_2) {
    p_ = p;
    t_la_ = t_la;
    c_ = c;
    m_ = m;
    k_1_ = k_1;
    k_2_ = k_2;
}

void DRFModel::setModelParams(prescan::sim::SelfSensorUnit* m_egoSelfUnit, std::vector<prescan::sim::SelfSensorUnit*> m_targetSelfUnitList, 
                              prescan::api::roads::types::Road road, std::vector<std::string> targetNameList) {
    m_egoSelfUnit_ = m_egoSelfUnit;
    m_targetSelfUnitList_ = m_targetSelfUnitList;
    road_ = road;
    targetNameList_ = targetNameList;
    x_ = m_egoSelfUnit_->selfSensorOutput().PositionX;
    y_ = m_egoSelfUnit_->selfSensorOutput().PositionY;
    delta_ = 1e-8; // assume drive straight at beginning
    velocity_ = 10; //m_egoSelfUnit_->selfSensorOutput().Velocity; // set initial speed
    phiv_ = m_egoSelfUnit_->selfSensorOutput().OrientationYaw;
}

/* Filter delta (steering angle) below 1e-8, set vehicle's delta_ to the given value
** Input: (steering angle) delta [rad]
** Note: delta is positive when it's at the clockwise direction of the heading
*/
void DRFModel::gaussian3DTorusDelta(double delta) {
    if (abs(delta) < 1e-8) {
        delta_ = 1e-8;
    }
    else {
        delta_ = delta;
    }
}
void DRFModel::gaussian3DTorusDelta() {
    if (abs(delta_) < 1e-8) {
        delta_ = 1e-8;
    }
}

/* Keep the heading phiv (yaw) in range 0 -- 2pi
** Input: (heading) phiv [rad]
** Note: heading is defined as the angle between velocity and global x axis.
** It is positive when it's at the counter-clockwise direction of x axis.
*/
void DRFModel::gaussian3DTorusPhiv() {
    int numOf2Pi = abs(phiv_ / (2 * M_PI));
    phiv_ = remainder(numOf2Pi * 2 * M_PI + phiv_, 2 * M_PI);
}

/* Calculate look-ahead distance
** Input: (vehicle's velocity) [m/s]
*/
void DRFModel::gaussian3DTorusDla() {
    d_la = velocity_ * t_la_;
}

/* Determine the radius of the arc (R_car) which the car will go in the preview time
** Note: This equation is derived under the assumption of a kinematic vehicle model.
*/
void DRFModel::gaussian3DTorusR() {
    R = abs(L / tan(delta_));
}

// For optimization
void DRFModel::gaussian3DTorusR(double delta) {
    if (abs(delta) < 1e-8) {
        delta = 1e-8;
    }
    R = abs(L / tan(delta));
}

void DRFModel::gaussian3DTorusR2() {
    R = abs(velocity_ / yawRate_);
}

void DRFModel::gaussian3DTorusR2(double yawRate) {
    if (abs(yawRate) < 1e-8) {
        yawRate_ = 1e-8;
    }
    else {
        yawRate_ = yawRate;
    }
    R = velocity_ / yawRate_;
}

/* Calculate the global coordinates of the center of the arc in the preview time
*/
void DRFModel::gaussian3DTorusXcyc(double x, double y) {
    double phil; // phil is the angle determined by vehicle's rotation direction
    if (delta_ > 0) {
        phil = phiv_ + M_PI / 2;
    }
    else {
        phil = phiv_ - M_PI / 2;
    }
    x_ = x;
    y_ = y;
    xc_ = R * cos(phil) + x_;
    yc_ = R * sin(phil) + y_;
}
void DRFModel::gaussian3DTorusXcyc() {
    double phil; // phil is the angle determined by vehicle's rotation direction
    if (delta_ > 0) {
        phil = phiv_ + M_PI / 2;
    }
    else {
        phil = phiv_ - M_PI / 2;
    }
    xc_ = R * cos(phil) + x_;
    yc_ = R * sin(phil) + y_;
}

/* Calculate the global coordinates of the center of the arc in the preview time
** Note: Use yaw rate to determine the car's rotational direction instead of steering
*/
void DRFModel::gaussian3DTorusXcyc2() {
    double phil; // phil is the angle determined by vehicle's rotation direction
    if (yawRate_ > 0) {
        phil = phiv_ + M_PI / 2;
    }
    else {
        phil = phiv_ - M_PI / 2;
    }
    xc_ = R * cos(phil) + x_;
    yc_ = R * sin(phil) + y_;
}

/* Calculate the discrete grid around the vehicle position
** Note: Convention: row number as y axis, column number as x
*/
void DRFModel::gaussian3DTorusMeshgrid() {
    int n = 1; // meshgrid +- n * dla around the vehicle's position
    double xLowerBound = x_ - n * d_la;
    double xUpperBound = x_ + n * d_la;
    double yLowerBound = y_ - n * d_la;
    double yUpperBound = y_ + n * d_la;

    int numOfGrids = 2 * n * d_la / gridResolution + 1;
    std::cout << "numOfGrids =  " << numOfGrids * numOfGrids << std::endl;
    xGrid.resize(numOfGrids);
    yGrid.resize(numOfGrids);
    X.resize(numOfGrids * numOfGrids);
    Y.resize(numOfGrids * numOfGrids);
    
    // Note: For std::valarray, every element is 0 after resizing, perfect.
    //arcLength.resize(numOfGrids);
    arcLen.resize(numOfGrids * numOfGrids);

    //heightOfGaussian.resize(numOfGrids);
    hOfGaussian.resize(numOfGrids * numOfGrids);

    //innerSigmaOfGaussian.resize(numOfGrids);
    innerSigOfGaussian.resize(numOfGrids * numOfGrids);
    //outerSigmaOfGaussian.resize(numOfGrids);
    outerSigOfGaussian.resize(numOfGrids * numOfGrids);

    //zOfGaussian.resize(numOfGrids);
    ziOfGaussian.resize(numOfGrids * numOfGrids);

    //roadCost.resize(numOfGrids);
    rCost.resize(numOfGrids * numOfGrids);

    //obsCost.resize(numOfGrids);
    oCost.resize(numOfGrids * numOfGrids);
    //totalCost.resize(numOfGrids);
    tCost.resize(numOfGrids * numOfGrids);

    for (int i = 0; i < numOfGrids; i++) {
        xGrid[i] = (xLowerBound + i * gridResolution);
        yGrid[i] = (yLowerBound + i * gridResolution);

        /* arcLength[i].resize(numOfGrids);
        heightOfGaussian[i].resize(numOfGrids);
        innerSigmaOfGaussian[i].resize(numOfGrids);
        outerSigmaOfGaussian[i].resize(numOfGrids);
        zOfGaussian[i].resize(numOfGrids);
        
        roadCost[i].resize(numOfGrids);
        
        obsCost[i].resize(numOfGrids);
        
        totalCost[i].resize(numOfGrids); */ 
        for (int j = 0; j < numOfGrids; j++) {
            X[i * numOfGrids + j] = (xLowerBound + j * gridResolution);
            Y[i * numOfGrids + j] = (yLowerBound + i * gridResolution);
        }
    }
    // Set all obscost to 0 to superpose all obstacles'
    // costs later.
/*  for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            obsCost[j][i] = 0;
        }
    } */
}

/* Calculate mexp, where sigma = mexp * s + cexp
** Note: mexp is for computing sigma_1 and sigma_2.
*/
void DRFModel::gaussian3DTorusMexp() {
    mexp1 = m_ + k_1_ * abs(delta_);
    mexp2 = m_ + k_2_ * abs(delta_);
}
/* Calculate mexp, where sigma = mexp * s + cexp
** Note: mexp is for computing sigma_1 and sigma_2.
** TODO: m_, k_1(2)_ here should be different from above.
*/
void DRFModel::gaussian3DTorusMexp2() {
    mexp1 = m_ + k_1_ * abs(yawRate_);
    mexp2 = m_ + k_2_ * abs(yawRate_);
}

/* Calculate the arc length between the car's position and meshgrids to get the height 
** of Gaussian in the torus' section later. 
** Note: All intermediate angles are in radians.
*/
void DRFModel::gaussian3DTorusArclen() {
     //Older version
    // global angle between the vehicle and center of arc
    double theta1 = remainder(2 * M_PI + atan2(y_ - yc_, x_ - xc_), 2 * M_PI); 
    /* for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            // global angle between the meshgrids and center of arc
            double theta2 = remainder(2 * M_PI + atan2(yGrid[j] - yc_, xGrid[i] - xc_), 2 * M_PI);
            // global angle between the meshgrids and vehicle
            double theta = theta2 - theta1;
            // keep theta positive, where sign(delta_) is implemented as:
            theta = ((delta_ > 0) - (delta_ < 0)) * theta;
            arcLength[j][i] = R * theta;
        }
    } */
    //double theta1 = remainder(2 * M_PI + atan2(y_ - yc_, x_ - xc_), 2 * M_PI);
    //std::valarray<double> THETA1(theta1, X.size());
    // mod operation not supported...
    //std::valarray<double> THETA2 = (2 * M_PI + std::atan2(Y - yc_, X - xc_)) % (2 * M_PI);
    std::valarray<double> THETA2(0.0, X.size());
    for (int i = 0; i < X.size(); i++) {
        THETA2[i] = remainder(2 * M_PI + atan2(Y[i] - yc_, X[i] - xc_), 2 * M_PI);
    }
    std::valarray<double> THETA = THETA2 - theta1;
    THETA *= ((delta_ > 0) - (delta_ < 0));
    arcLen = R * THETA;
}

/* Calculate the arc length between the car's position and meshgrids to get the height 
** of Gaussian in the torus' section later. 
** Note: All intermediate angles are in radians.
** Note2: This function uses yaw rate the determine rotation direction.
*/
void DRFModel::gaussian3DTorusArclen2() {
    // global angle between the vehicle and center of arc
    double theta1 = remainder(2 * M_PI + atan2(y_ - yc_, x_ - xc_), 2 * M_PI); 
    for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            // global angle between the meshgrids and center of arc
            double theta2 = remainder(2 * M_PI + atan2(yGrid[j] - yc_, xGrid[i] - xc_), 2 * M_PI);
            // global angle between the meshgrids and vehicle
            double theta = theta2 - theta1;
            // keep theta positive, where sign(yawRate_) is implemented as:
            theta = ((yawRate_ > 0) - (yawRate_ < 0)) * theta;
            arcLength[j][i] = R * theta;
        }
    }
}

/* Calculate the height of Gaussian in every grid's cross section of the torus
*/
void DRFModel::gaussian3DTorusA() {
    // Older version
    /* for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            double arc = arcLength[j][i];
            double a_ji = p_ * (arc - d_la) * (arc - d_la);
            if (d_la < arc || a_ji < 0 || arc < 0) {
                heightOfGaussian[j][i] = 0;
            }
            else {
                heightOfGaussian[j][i] = a_ji;
            }
            // Note: In MATLAB script, it considers the case where 
            // arclen is negative, because it doesn't consider back of the vehicle.
        }
    } */
    std::valarray<double> AIJ = p_ * (arcLen - d_la) * (arcLen - d_la);
    hOfGaussian = AIJ;
    hOfGaussian[arcLen < 0.0 || arcLen > d_la] = 0;
}

/* Calculate the sigma of Gaussian in every grid's cross section
*/
void DRFModel::gaussian3DTorusSigma() {
    // Older version
    /* for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            innerSigmaOfGaussian[j][i] = mexp1 * arcLength[j][i] + c_;
            outerSigmaOfGaussian[j][i] = mexp2 * arcLength[j][i] + c_;
        }
    } */
    innerSigOfGaussian = mexp1 * arcLen + c_;
    outerSigOfGaussian = mexp2 * arcLen + c_;
}

/* Separate the meshgrids into two categories: inside/outside the turning circle
*  Then compute the inner/outer Gaussian separately, finally add both together
*  to obtain the Gaussian3Dtorus distribution.
*/
void DRFModel::gaussian3DTorusZ() {
    // Older version
    /* for (int i = 0; i < xGrid.size(); i++) {
        double x = xGrid[i];
        for (int j = 0; j < yGrid.size(); j++) {
            double y = yGrid[j];
            double dist2Xcyc = sqrt((x - xc_) * (x - xc_) + (y - yc_) * (y - yc_));

            // grid in outer circle
            if (dist2Xcyc > R) {
                double outerSigma = outerSigmaOfGaussian[j][i];
                zOfGaussian[j][i] = heightOfGaussian[j][i] * exp(-(dist2Xcyc - R) * 
                                    (dist2Xcyc - R) / 2 / outerSigma / outerSigma);
            }
            // grid in inner circle
            else {
                double innerSigma = innerSigmaOfGaussian[j][i];
                zOfGaussian[j][i] = heightOfGaussian[j][i] * exp(-(dist2Xcyc - R) * 
                                    (dist2Xcyc - R) / 2 / innerSigma / innerSigma);
            }
        }
    } */
    std::valarray<double> DIST2XCYC = std::sqrt((X - xc_) * (X - xc_) + (Y - yc_) * (Y - yc_));
    std::valarray<double> outerZ = hOfGaussian * std::exp(-(DIST2XCYC - R) * (DIST2XCYC - R) /
                                   2.0 / outerSigOfGaussian / outerSigOfGaussian);
    std::valarray<double> innerZ = hOfGaussian * std::exp(-(DIST2XCYC - R) * (DIST2XCYC - R) /
                                   2.0 / innerSigOfGaussian / innerSigOfGaussian);                               
    for (int i = 0; i < X.size(); i++) {
        if (DIST2XCYC[i] > R) {
            ziOfGaussian[i] = outerZ[i];
        }
        else {
            ziOfGaussian[i] = innerZ[i];
        }
    }
}

/* Generate cost from the road network only (not including obstacles)
** MATLAB Script:
** if meshgrids are on the road, 0 cost.
** else, 500 cost
** Note: the roadCost ventor is initialized to be a zero vector by default. 
*/
void DRFModel::generateCircuitRoadSoloCost() {
    // Older version
    /* for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            if (xGrid[i] < 100) {
                if (isPointOnStraightSection(xGrid[i], yGrid[j])) {
                roadCost[j][i] = 0;
                //std::cout << "on road! " << std::endl;
                }
                else {
                    roadCost[j][i] = 500;
                    //std::cout << "off road! " << std::endl;
                }
            }
            else {
                if (isPointOnArcSection(xGrid[i], yGrid[j])) {
                roadCost[j][i] = 0;
                //std::cout << "on road! " << std::endl;
                }
                else {
                    roadCost[j][i] = 500;
                    //std::cout << "off road! " << std::endl;
                }
            }
        }
    } */
    for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            if (xGrid[j] < 100) {
                if (isPointOnStraightSection(xGrid[i], yGrid[j])) {
                    rCost[j * xGrid.size() + i] = 0;
                }
                else {
                    rCost[j * xGrid.size() + i] = 500;
                }
            }
            else {
                if (isPointOnArcSection(xGrid[i], yGrid[j])) {
                    rCost[j * xGrid.size() + i] = 0;
                }
                else {
                    rCost[j * xGrid.size() + i] = 500;
                }
            }
        }
    }
}

/* These road function should be changed according to the road built in "Initialisaiton.cpp"
** Note: The following road functions are for the "J"-shaped road.
*/
bool DRFModel::isPointOnStraightSection(double x, double y) {
    double lowerBoundY = -6.4;
    double higherBoundY = 0.0;
    double lowerBoundX = 0.0;
    double higherBoundX = 100.0;
    if (x < lowerBoundX || x > higherBoundX || y < lowerBoundY || y > higherBoundY) {
        return false;
    }
    return true;
}
bool DRFModel::isPointOnArcSection(double x, double y) {
    double arcCenterX = 100;
    double arcCenterY = 46.8;
    double higherBoundR = 53.2;
    double lowerBoundR = 46.8;
    double dist2Center = sqrt((x - arcCenterX) * (x - arcCenterX) + (y - arcCenterY) * (y - arcCenterY));
    if (dist2Center < lowerBoundR || dist2Center > higherBoundR) {
        return false;
    }
    return true;
}

/* Obstacle cost
** TODO: Consider different bounding box size for different obstacles.
** HINT: Check the how to access bounding box size.
** Cost values from paper: if grid in obstacle, cost += 2500
**                         else, cost += 0
** Input: All other vehicle (obstacle)'s global coordinates x and y
** TODO: Consider obstacles' shape more accurately, now only consider a bounding box around the obstacle.
*/
void DRFModel::obstacleCost(double obstacleX, double obstacleY) {
    // Older version
    /* for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            // grid in obstacle
            if (xGrid[i] >= obstacleX - L / 2 && xGrid[i] <= obstacleX + L / 2 &&
                yGrid[j] >= obstacleY - L / 2 && yGrid[j] <= obstacleY + L / 2 ) {
                obsCost[j][i] += 2500;
            }
        }
    } */
    oCost[X >= obstacleX - L / 2 && X <= obstacleX + L / 2 && Y >= obstacleY - L / 2 && Y <= obstacleY + L / 2] = 2500;
}

void DRFModel::totalObsCost() {
    for (int i = 0; i < targetNameList_.size(); i++) {
        auto targetOutput = m_targetSelfUnitList_[i]->selfSensorOutput();
        obstacleCost(targetOutput.PositionX, targetOutput.PositionY);
    }
}

/* Environment cost map(total cost) = (road cost + obstacle cost) * zOfGaussian
** Output: Scaler value of driver's perceived risk
*/
double DRFModel::environmentCost() {
    // Older version
    /* double perceivedRisk = 0;
    for (int i = 0; i < xGrid.size(); i++) {
        for (int j = 0; j < yGrid.size(); j++) {
            totalCost[j][i] = (roadCost[j][i] + obsCost[j][i]) * zOfGaussian[j][i]; 
            perceivedRisk += totalCost[j][i];
        }
    }
    perceivedRisk_ = perceivedRisk * gridResolution * gridResolution;
    //return perceivedRisk_; */
    
    tCost = (rCost + oCost) * ziOfGaussian;
    return tCost.sum() * gridResolution * gridResolution;
}

/* Visualise the total cost map as a mesh plot
*/
void DRFModel::visualiseCost() {
    auto [X, Y] = matplot::meshgrid(xGrid, yGrid);
    //auto Y = yGrid.data();
    matplot::tiledlayout(2, 2);

    matplot::nexttile();
    matplot::mesh(X, Y, zOfGaussian);
    matplot::colorbar(); // You can choose which plot to visualise here

    matplot::nexttile();
    matplot::mesh(X, Y, totalCost);
    matplot::colorbar();

    matplot::nexttile();
    matplot::mesh(X, Y, roadCost);
    matplot::colorbar();

    matplot::nexttile();
    matplot::mesh(X, Y, obsCost);
    matplot::colorbar();

    //auto ax = nexttile(2);
    //colormap(ax, palette::cool());

    //matplot::view(90, 0);
    matplot::show();
}

/* Save cost map as csv file
*/
void DRFModel::saveCostMap() {
    //std::ofstream out("costMap.csv");
    std::ofstream file("costMap.csv", std::ios::app);
    //file << "yGrid" << '\n';
    for (int i = 0; i < yGrid.size(); i++) {
        if (i < yGrid.size() - 1) {
            file << yGrid[i] << ',';
        }
        else {
            file << yGrid[i] << '\n';
        }
    }

    //file << "xGrid" << '\n';
    for (int i = 0; i < xGrid.size(); i++) {
        if (i < xGrid.size() - 1) {
            file << xGrid[i] << ',';
        }
        else {
            file << xGrid[i] << '\n';
        }
    }

    //file << "cost" << std::endl;
    /*
    for (auto& row : totalCost) {
        for (int i = 0; i < row.size(); i++) {
            if (i < row.size() - 1) {
                file << row[i] << ',';
            }
            else {
                file << row[i] << '\n';
            }
        }
    }*/
    int rowCount = 0;
    int rowSize = xGrid.size();
    int colSize = rowSize;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            if (j < colSize - 1) {
                file << tCost[i * colSize + j] << ',';
            }
            else {
                file << tCost[i * colSize + j] << '\n';
            } 
        }
    }
}

/* Save subjective map as csv file
*/
void DRFModel::saveSubjectiveMap() {
    //std::ofstream out("costMap.csv");
    std::ofstream file("subMap.csv", std::ios::app);
    //file << "yGrid" << '\n';
    for (int i = 0; i < yGrid.size(); i++) {
        if (i < yGrid.size() - 1) {
            file << yGrid[i] << ',';
        }
        else {
            file << yGrid[i] << '\n';
        }
    }

    //file << "xGrid" << '\n';
    for (int i = 0; i < xGrid.size(); i++) {
        if (i < xGrid.size() - 1) {
            file << xGrid[i] << ',';
        }
        else {
            file << xGrid[i] << '\n';
        }
    }

    int rowCount = 0;
    int rowSize = xGrid.size();
    int colSize = rowSize;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            if (j < colSize - 1) {
                file << ziOfGaussian[i * colSize + j] << ',';
            }
            else {
                file << ziOfGaussian[i * colSize + j] << '\n';
            } 
        }
    }
}

/* Save road cost as csv file
*/
void DRFModel::saveRoadCost() {
    //std::ofstream out("costMap.csv");
    std::ofstream file("roadCost.csv", std::ios::app);
    //file << "yGrid" << '\n';
    for (int i = 0; i < yGrid.size(); i++) {
        if (i < yGrid.size() - 1) {
            file << yGrid[i] << ',';
        }
        else {
            file << yGrid[i] << '\n';
        }
    }

    //file << "xGrid" << '\n';
    for (int i = 0; i < xGrid.size(); i++) {
        if (i < xGrid.size() - 1) {
            file << xGrid[i] << ',';
        }
        else {
            file << xGrid[i] << '\n';
        }
    }

    int rowCount = 0;
    int rowSize = xGrid.size();
    int colSize = rowSize;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            if (j < colSize - 1) {
                file << rCost[i * colSize + j] << ',';
            }
            else {
                file << rCost[i * colSize + j] << '\n';
            } 
        }
    }
}

/* Save obstacle cost as csv file
*/
void DRFModel::saveObstacleCost() {
    //std::ofstream out("costMap.csv");
    std::ofstream file("obsCost.csv", std::ios::app);
    //file << "yGrid" << '\n';
    for (int i = 0; i < yGrid.size(); i++) {
        if (i < yGrid.size() - 1) {
            file << yGrid[i] << ',';
        }
        else {
            file << yGrid[i] << '\n';
        }
    }

    //file << "xGrid" << '\n';
    for (int i = 0; i < xGrid.size(); i++) {
        if (i < xGrid.size() - 1) {
            file << xGrid[i] << ',';
        }
        else {
            file << xGrid[i] << '\n';
        }
    }

    int rowCount = 0;
    int rowSize = xGrid.size();
    int colSize = rowSize;
    for (int i = 0; i < rowSize; i++) {
        for (int j = 0; j < colSize; j++) {
            if (j < colSize - 1) {
                file << oCost[i * colSize + j] << ',';
            }
            else {
                file << oCost[i * colSize + j] << '\n';
            } 
        }
    }
}

/* Save ego trajectory in a csv file
*/
void DRFModel::saveEgoTrajectory() {
    std::ofstream file("EgoTrajectory.csv", std::ios::app);
    file << x_ << ',';
    file << y_ << '\n';
}

/* Kinematic car model
*/
void DRFModel::carKinematics() {
    x_ = x_ + velocity_ * cos(phiv_) * dt;
    y_ = y_ + velocity_ * sin(phiv_) * dt;
    phiv_ = phiv_ + velocity_ / L * tan(delta_) * dt;
}