# Neural-network-from-scratch

Implementation of a feedforward neural network in C++, including the backpropagation algorithm. Backend code is headers only. Uses the [iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) to showcase code.

## Description

This project was meant to refresh my memory on C++ and at the same time get a more intimate handle on the celebrated backpropagation algorithm (and also learn to work with Eigen's many quirks!). Initially, I was going to follow this [blog post](http://www.code-spot.co.za/2009/10/08/15-steps-to-implemented-a-neural-net/) but ended up taking a different direction. Namely, I use a softmax output layer and built my pipeline to be indefinitely extendible. 

Backend code is entirely templated, thus headers only. Neural network pipeline uses curiously recurring template pattern (CRTP) for compile-time polymorphism. Implements the backward propagation alagorithm for feedforward neural networks and does not implement general computational graphs. 

## Requirements and Dependencies

Versions of packages correspond to my local installations; may also work with other versions.

- CMake (3.12)
- Eigen C++ library (3.4.0)
- GNU Make (4.3) if using command line-based compiler **or** Microsoft Visual Studio 2022 (17.6.2)
- C++ compiler with support for C++20 and `pragma once`

### Tested compilers

- g++ (11.3.5)
- clang++ (14.0.0)
- MSVC (19.36.32532 for x64)

## Build and Run

Instructions are for Bash shell. Adapt as needed. On Windows you may use [git bash](https://git-scm.com/downloads) to emulate Bash shell. Instructions assume some level of relevant acumen.

1. Install the dependencies using your OS's package manager (e.g. `apt-get` for Ubuntu, `dnf` for RHEL, `vcpkg` for Windows). You may want to refer to [Eigen's documentation](https://eigen.tuxfamily.org/dox/GettingStarted.html).

2. Clone the repository:

   `$ git clone https://github.com/eliasta2610/Neural-network-from-scratch`

   and `cd` into the created `Neural-network-from-scratch/` directory.

3. Download the data set from [here](http://www.code-spot.co.za/downloads/neural_net/iris_data_files.zip). Unzip `iris_data_files` directory under`Neural-network-from-scratch/data/`:

   `$ wget http://www.code-spot.co.za/downloads/neural_net/iris_data_files.zip ./data`\
   `$ unzip ./data/iris_data_files.zip .`

4. Run `cmake`:

   `$ cmake .`

   If you wish to override your system's default C++ compiler, run with

   `$ cmake -DCMAKE_CXX_COMPILER=<path-to-C++-compiler> .` 

   Note that `cmake` needs to be made aware of the directory in which Eigen resides. If this isn't the case out of the box, you may need to modify `Neural-network-from-scratch/CMakeLists.txt` to set the environment variable `Eigen3_DIR` or `CMAKE_PREFIX_PATH`.


5. If using a command line compiler run

   `$ make -f Makefile`

   If using Microsoft Visual Studio, open `./build/NeuralNet.sln` and from the tool bar run `Build > Build Solution`.

5. Run the executable.
   
   If using command line compiler:

   `$ cd ./build`\
   `$ ./NeuralNet`

   If using Microsoft Visual Studio:

   `$ cd ./build/Debug`\
   `$ ./NeuralNet`

## Possible Roadmap

- Implement more layer types (different activation functions)
- Support symbolic differentiation so that activation and differentiation functions do not have to be defined separately
- Implement more loss functions (e.g. mse, exponential loss)
- Show training progress of neural network via a plotting library
- Add support for regularization
