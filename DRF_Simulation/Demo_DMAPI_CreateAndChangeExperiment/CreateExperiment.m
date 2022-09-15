%% Experiment creation from scratch with DMAPI

%% Create an empty experiment
exp = prescan.api.experiment.Experiment();

%% Create a road
road_1 = prescan.api.roads.createRoad(exp);

%% Add sections to the road
sectionLength = 200; % [m]
road_1.addStraightSection(sectionLength);
sectionLength = 50;
startCurvature = 0.025;
endCurvature = 0.1;
road_1.addSpiralSection(sectionLength, startCurvature, endCurvature);

%% Once the sections have been added, add lanes to the road
lane1 = road_1.addLeftLane(3.2);
marker = lane1.getLaneMarker('Outer');
marker.type = 'Broken';
lane2 = road_1.addLeftLane(3.2);
marker = lane2.getLaneMarker('Outer');
marker.type = 'Solid';
lane3 = road_1.addRightLane(3.2);
marker = lane3.getLaneMarker('Outer');
marker.type = 'Broken';
lane4 = road_1.addRightLane(3.2);
marker = lane4.getLaneMarker('Outer');
marker.type = 'Solid';

centerOfLane1 = - 3.2/2; % 2 lanes to the right, just in the middle of one
centerOfLane2 = -3.2 - 3.2/2; % 2 lanes to the right, just in the middle of one
edgeOfRoadLeft = -15; 
edgeOfRoadRight = 15; 

%% Add objects to the road
vehicleAudi = exp.createObject(exp.objectTypes.Audi_A8_Sedan);
vehicleAudi.name = 'Ego';
vehicleBMW = exp.createObject(exp.objectTypes.BMW_X5_SUV);
vehicleBMW.name = 'Target1';
vehicleFord = exp.createObject(exp.objectTypes.Ford_Fiesta_Hatchback);
vehicleFord.name = 'Target2';

% Trees
for i = 0:50
    r = -3 + (3+3).*rand(3,1); % Make it random
    xtree = 6 * i + r(1);
    ytree1 = edgeOfRoadLeft + r(2);
    ytree2 = edgeOfRoadRight + r(3);
    
    tree = exp.createObject(exp.objectTypes.Dogwood20y);
    tree.pose.position.x = xtree;
    tree.pose.position.y = ytree1;
    
    tree = exp.createObject(exp.objectTypes.Dogwood20y);
    tree.pose.position.x = xtree;
    tree.pose.position.y = ytree2;
end

%% Position the vehicles correctly
vehicleAudi.pose.position.y = centerOfLane2;
vehicleBMW.pose.position.y = centerOfLane2;
vehicleFord.pose.position.y = centerOfLane1;

vehicleBMW.pose.position.x = 30; % [m] in front
vehicleFord.pose.position.x = 15; % [m] in front

%% Add a trajectory to the ego vehicle
z = [0, 0];
y = [vehicleAudi.pose.position.y, vehicleAudi.pose.position.y];
x = [vehicleAudi.pose.position.x, 200];
pathAudi = prescan.api.trajectory.createFittedPath(exp, x, y, z);
y = [vehicleBMW.pose.position.y, vehicleBMW.pose.position.y];
x = [vehicleBMW.pose.position.x, 200];
pathBMW = prescan.api.trajectory.createFittedPath(exp, x, y, z);
y = [vehicleFord.pose.position.y, vehicleFord.pose.position.y];
x = [vehicleFord.pose.position.x, 200];
pathFord = prescan.api.trajectory.createFittedPath(exp, x, y, z);
speedProf = prescan.api.trajectory.createSpeedProfileOfConstantSpeed(exp, 5);

trajectoryAudi = prescan.api.trajectory.createTrajectory(vehicleAudi, pathAudi, speedProf);
trajectoryBMW = prescan.api.trajectory.createTrajectory(vehicleBMW, pathBMW, speedProf);
trajectoryFord = prescan.api.trajectory.createTrajectory(vehicleFord, pathFord, speedProf);

%% Add a camera to the ego vehicle
cameraAudi = prescan.api.camera.createCameraSensor(vehicleAudi);

%% Change resolution of the camera, for HD
cameraAudi.focalLength = 5;

%% Save the experiment to a file
exp.saveToFile('DMAPI_only.pb');

%% Generate the CS
prescan.api.simulink.generate();

%% Run the experiment
prescan.api.simulink.run(exp, 'Regenerate', 'off', 'StopTime', '5')