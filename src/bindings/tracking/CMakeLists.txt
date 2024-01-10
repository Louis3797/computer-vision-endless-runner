cmake_minimum_required(VERSION 3.26)
project(tracking LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

# cmake_policy(SET CMP0148 OLD)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

if(WIN32)
    set(pybind11_DIR "C:/Users/janra/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11")
    set(CMAKE_PREFIX_PATH "C:/Users/janra/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11")
    set(OpenCV_DIR "C:/tools/opencv/build")
    set(Eigen3_DIR "C:/ProgramData/Eigen3/share/cmake")
else()
    set(pybind11_DIR "/opt/homebrew/lib/python3.11/site-packages/pybind11")
    set(CMAKE_PREFIX_PATH "/opt/homebrew/lib/python3.11/site-packages/pybind11")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

pybind11_add_module(tracking
    src/bindings.cpp
    include/Gradient.h
    include/Line.h
    include/HOG.h
    include/SlidingWindow.h
    include/NonMaxSuppression.h
    include/ImagePyramid.h
    include/HOGDescriptor.h
    include/PersonDetector.h
)


target_link_libraries(tracking PRIVATE ${OpenCV_LIBS})



