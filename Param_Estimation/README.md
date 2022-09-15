# Customized Packages

These customized packages are adapted from [l5kit](https://github.com/woven-planet/l5kit/tree/master/l5kit/l5kit), which can work with the [Lyft Prediction Dataset](https://level-5.global/data/prediction/) and our customized adaptations on reactive agents.

Note: This README is still under further maintainance.

## Description

The customized packages are arranged in a similar style as l5kit. Here are the packages and a short description on each package's functionalities:

1. data: the foundation implementation of the zarr-based dataset. It is recommended not to change its contents unless you are very certain what you are doing.

2. dataset: the implementation on the egoDataset and agentDataset interfaces. These interfaces are used to extract information from the raw zarr dataset. You can change the contents in this folder according to your needs. 

3. driver: our implementation of the driver risk field ([DRF](https://www.nature.com/articles/s41467-020-18353-4#:~:text=We%20propose%20the%20Driver's%20Risk,of%20the%20driver's%20perceived%20risk.)) model. This model can be applied to agent vehicles to make reactive simulations.

4. geometry: the functions to aid the transformation between different coordinate systems (world, agent, raster). The original implementation is already very thorough.

5. kinematic: the kinematic model to simulate the vehicle kinematics. It is not necessary to use this kinematic model.

6. map: the implementation of environment map as rasterized images. It is mainly used for visualization purposes, though in our case, a customized map_build is created to aid the implementation of DRF models.

7. planning: our implementation of the DRF's parameter identification framework using HEEDS and validation workflow in Jupyter notebook. Please see the detailed description in its folder's README.

8. random: the random generators. It is recommended to not change its contents.

9. rasterization: the l5kit version of making rasterized plots.

10. sampling: the agent sampling functions. It is often used to select or filter out agents according to your needs.

## Getting Started

### Dependencies

* Windows 10
* Microsoft Visual Studio 2019
* Prescan 2021.2
* Cmake 3.17.3
* Git 2.35.1
* HEEDS 2022.2

### Installing

* Please follow the README instructions in our project's main page.

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