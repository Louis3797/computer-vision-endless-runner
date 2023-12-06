cmake_minimum_required(VERSION 3.26)
project(hog LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

cmake_policy(SET CMP0148 OLD)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

set(pybind11_DIR "C:/Users/janra/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11")
set(CMAKE_PREFIX_PATH "C:/Users/janra/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11")
set(OpenCV_DIR "C:/tools/opencv/build")
set(Eigen3_DIR "C:/ProgramData/Eigen3/share/cmake")


find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

pybind11_add_module(hog
    src/hog_binding.cpp
    include/Gradient.h
    include/Line.h
    include/HOG.h
)



target_link_libraries(hog PRIVATE ${OpenCV_LIBS})



