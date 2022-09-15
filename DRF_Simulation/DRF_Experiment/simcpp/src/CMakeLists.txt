cmake_minimum_required(VERSION 3.15)
project(DRF_Simulation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#[[list(APPEND CMAKE_MODULE_PATH C:\\Program Files\\Simcenter Prescan\\Prescan_2021.2)
list(APPEND CMAKE_MODULE_PATH C:\\Program Files\\matplotplusplus 1.1.0)

set(Prescan_DIR "C:/Program Files/Simcenter Prescan/Prescan_2021.2/lib/cmake")
set(Matplot++_DIR "C:/Program Files/matplotplusplus 1.1.0/lib/cmake/Matplot++")]]

add_subdirectory(matplotplusplus)
add_subdirectory(dlib dlib_build)

#[[ 
include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.18  
)
FetchContent_MakeAvailable(dlib)]]

find_package(Prescan 2021.2 REQUIRED 
             HINTS $ENV{Prescan_DIR})
             
#[[             
find_package(Matplot++ REQUIRED 
             HINTS $ENV{MATPLOTPP_DIR})]]
             
set(PATH_TO_PB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
configure_file(DRFSimulationModel.hpp.in DRFSimulationModel.hpp)

include_directories(DRFModel)
include_directories(eigen)

add_executable(Environment_Initialisation Initialisation.cpp)
add_executable(DRFSimulationModel DRFSimulationModel.cpp DRFModel/DRFModel.cpp)

target_include_directories(Environment_Initialisation
PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(DRFSimulationModel
PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/DRFModel/include
${CMAKE_CURRENT_SOURCE_DIR}/eigen
)

target_link_libraries(Environment_Initialisation ${Prescan_LIBS})
target_link_libraries(DRFSimulationModel 
                      PUBLIC matplot
                      ${Prescan_LIBS}
                      dlib::dlib)
                                  