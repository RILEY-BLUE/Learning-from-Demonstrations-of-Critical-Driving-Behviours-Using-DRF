#pragma once

const char path_to_pb_dir[] = "D:/YURUIDU/DRF_Simulation/DRF_Experiment";
/*
#include <prescan/sim/ISimulationModel.hpp>
#include "DRFModel/DRFModel.h"

class DRFSimulationModel : public prescan::sim::ISimulationModel {
public:
  DRFSimulationModel() = default;
  void registerSimulationUnits(const prescan::api::experiment::Experiment& experiment, prescan::sim::ISimulation* simulation) override;
  void initialize(prescan::sim::ISimulation* simulation) override;
  void step(prescan::sim::ISimulation* simulation) override;
  void terminate(prescan::sim::ISimulation* simulation) override {};
  void updateTrajectory() const;
  bool checkTermination(float simTime);
private:
    // One DRF model for one Prescan vehicle
    prescan::api::roads::types::Road road;
    prescan::sim::StateActuatorUnit* m_egoActuator;
    prescan::sim::SelfSensorUnit* m_egoSelfUnit;
    DRFModel egoDRF;

    prescan::sim::StateActuatorUnit* m_targetActuator;
    prescan::sim::SelfSensorUnit* m_targetSelfUnit;
	  DRFModel tarDRF;

    prescan::sim::PathUnit* m_egoPathUnit;
    prescan::sim::PathUnit* m_targetPathUnit;
    prescan::sim::SpeedProfileUnit* m_egoSpeedProfileUnit;
    prescan::sim::SpeedProfileUnit* m_targetSpeedProfileUnit;
    prescan::sim::AirSensorUnit* m_airSensorUnit;
    prescan::sim::CameraSensorUnit* m_cameraSensorUnit;
    
};

#endif*/
