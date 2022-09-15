%% Function to replace objects
%%
function replace(obj, targetObj) 
  % objects are in same experiment, no copy needed 
  replacingobject = obj; 
  
  % set pose of replacing object to that of the target object. 
  [tpx, tpy, tpz] = targetObj.pose.position.getXYZ(); 
  [tor, top, toy] = targetObj.pose.orientation.getRPY(); 
  replacingobject.pose.position.setXYZ(tpx, tpy, tpz); 
  replacingobject.pose.orientation.setRPY(tor, top, toy); 
    
  %Copy the sensors over
  for i = 1 : length(targetObj.sensors)
      targetObj.sensors(i).attachToObject(replacingobject);
  end
  
  %Copy trajectories as well
  trajectorylist = prescan.api.trajectory.getAttachedTrajectories(targetObj);
  
  for i = 1 : length(trajectorylist)
      prescan.api.trajectory.createTrajectory(obj, trajectorylist(i).path, trajectorylist(i).speedProfile);
  end
  
  targetObj.remove();
end 