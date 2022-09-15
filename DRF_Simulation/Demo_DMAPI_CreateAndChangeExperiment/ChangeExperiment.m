%% Experiment creation from scratch with DMAPI

%% Create an empty experiment
exp = prescan.api.experiment.Experiment('DMAPI_only.pb');

%% Find the vehicles and change their models
vehicleNissan = exp.createObject(exp.objectTypes.Nissan_Cabstar_Boxtruck);
vehicleFH16 = exp.createObject(exp.objectTypes.Volvo_FH16);
vehicleYaris = exp.createObject(exp.objectTypes.Toyota_Yaris_Hatchback);

replacedEgo = findobj(exp.objects, 'name','Ego');
replace(vehicleNissan, replacedEgo);
replacedTarget1 = findobj(exp.objects,'name','Target1');
replace(vehicleFH16, replacedTarget1);
replacedTarget2 = findobj(exp.objects,'name','Target2');
replace(vehicleYaris, replacedTarget2);

prescan.api.simulink.run(exp, 'StopTime', '2')

%% Replace every 2 trees with other trees
objects = exp.objects;
for o = 1: length(objects)
    object = exp.objects(o);
    name = object.name;
    if ~isempty(strfind(name,'Dogwood'))
        if rem(o,2) == 0
            tree = exp.createObject(exp.objectTypes.AustrianPine20y);
            replace(tree, object);
        end
    end
end

prescan.api.simulink.run(exp, 'StopTime', '10')

%% Change the sun location and weather
exp.weather.precipitation.setDefaultRain();

prescan.api.simulink.run(exp, 'Regenerate', 'off',  'StopTime', '5')

exp.sky.setPresetDawn();

prescan.api.simulink.run(exp, 'Regenerate', 'off',  'StopTime', '5')

exp.weather.precipitation.setDisabled();

prescan.api.simulink.run(exp, 'Regenerate', 'off',  'StopTime', '5')

exp.sky.setPresetNight();

%% Change the trajectories
trajectorylist = prescan.api.trajectory.getAttachedTrajectories(vehicleNissan);
z = [0, 0];
y = [3.2/2, 3.2/2];
x = [vehicleNissan.pose.position.x, 200];
newPath = prescan.api.trajectory.createFittedPath(exp, x, y, z);

newSpeedProf = prescan.api.trajectory.createSpeedProfileOfConstantSpeed(exp, 10);

newTrajectory = prescan.api.trajectory.createTrajectory(vehicleNissan, newPath, newSpeedProf);
newTrajectory.setActive();

%% Run the simulation
prescan.api.simulink.run(exp, 'Regenerate', 'off',  'StopTime', '10')