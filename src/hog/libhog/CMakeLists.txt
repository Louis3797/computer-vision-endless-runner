cmake_minimum_required(VERSION 3.26)
project(cv_hog LANGUAGES CXX)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)
set(pybind11_DIR "/opt/homebrew/lib/python3.11/site-packages/pybind11")
set(CMAKE_PREFIX_PATH "/opt/homebrew/lib/python3.11/site-packages/pybind11")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

pybind11_add_module(cv_hog src/hog_binding.cpp   src/draw_new.cpp
        src/hoghistogram.cpp
        src/hog.cpp
        include/draw_new.h
        include/hoghistogram.h
        include/hog.h)

target_link_libraries(cv_hog PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
