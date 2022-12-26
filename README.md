# Learning from Demonstrations of Critical Driving Behaviours Using Driver's Risk Field

Imitation Learning using a multi-agent traffic simulator with driver risk field (DRF) agents.


## Description

This project has the following objectives:

1. Implement DRF human behavior driver models in C++ and Python. The Python version is validated with [Lyft Prediction Dataset](https://level-5.global/data/prediction/) in lane-keeping, car-following and braking scenarios from urban traffic.

2. Integrate the developed C++ models into a simulation pipeline of Simcenter Prescan.

3. Develop offline RL and validation.

## Getting Started

### Dependencies

* Windows 10
* Microsoft Visual Studio 2019
* Prescan 2021.2
* Cmake 3.17.3
* Git 2.35.1
* HEEDS 2022.2

### Installing

* Please configure the Prescan-OpenAI Gym environment as described [HERE](https://belnsptgitlab01.net.plm.eds.com/ADAS/mimic/prescan-gym)
* In your project folder (in my case D:\YURUIDU\DRF_Simulation), open Git Bash and clone this project:  
```
git clone git@belnsptgitlab01.net.plm.eds.com:ADAS/interns/yurui-du.git
```
* Create a folder called 'Param_Estimation' in your project folder, add this [setup.py](https://belnsptgitlab01.net.plm.eds.com/ADAS/interns/yurui-du/-/blob/master/setup.py) in this newly-created 'Param_Estimation' folder.
* Copy the 'Param_Estimation' folder from the cloned respository to your newly-created 'Param_Estimation' folder.
* In the newly-created 'Param_Estimation' folder, create a virtual environment using the command prompt:
```
python -m venv venv
```
* In the same folder, activate the virtual environment:
```
./venv/Scripts/activate
```
* Go to the root of your project folder, install this project in the editable mode:
```
pip install -e .
```
* Install [l5kit](https://woven-planet.github.io/l5kit/installation.html) to use all the tools provided along with Lyft dataset.

### Executing program

* After previous steps, you should now be able to run the identification files (in the param_est_heeds folder) in HEEDS and validation scripts in Jupyter notebook.
* To run in Jupyter notebook, type the following command in the project's virtual environment:
```
python -m notebook
```

## Authors

[Yurui Du](Y.DU-7@student.tudelft.nl)

<!-- ## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46) -->
