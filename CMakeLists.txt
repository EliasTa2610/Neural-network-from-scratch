cmake_minimum_required(VERSION 3.12)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)  # Set the desired C++ standard

set(SOURCES 
    mlp.cpp
)

set(HEADERS
    utilities/types.h
    utilities/paral.h
    utilities/softmax.h
    utilities/traits_concepts.h
    input.h
    labels.h
    layers.h
    net.h
    loss.h
)

include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(NeuralNet ${SOURCES} ${HEADERS})

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(NeuralNet PUBLIC Eigen3::Eigen)
