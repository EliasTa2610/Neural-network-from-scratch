cmake_minimum_required(VERSION 3.12)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES 
    src/mlp.cpp
)

set(HEADERS
    utilities/types.h
    utilities/paral.h
    utilities/softmax.h
    utilities/traits_concepts.h
    include/input.h
    include/labels.h
    include/layers.h
    include/net.h
    include/loss.h
)

include_directories(${EIGEN_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_executable(NeuralNet ${SOURCES} ${HEADERS})

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(NeuralNet PUBLIC Eigen3::Eigen)
