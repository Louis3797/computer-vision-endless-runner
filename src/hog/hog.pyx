# hog.pyx

cimport numpy as np
import numpy as np


# For cv::Mat usage
cdef extern from "core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F

cdef extern from "core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

cdef extern from "hog.cpp" namespace "cv":
    # Define the functions from the C++ code you want to use in Python
    # Make sure to match the signatures and types correctly
    tuple[Mat, Mat] hog(Mat &image,
             int orientations,
             tuple[int, int] pixels_per_cell,
             tuple[int, int] cells_per_block,
             std_string method,
             bint visualize,
             bint transform_sqrt,
             bint flatten)

# Function to convert numpy.ndarray to cv::Mat
cdef Mat np2Mat(np.ndarray[np.float32_t, ndim=2] ary):
    cdef Mat mat
    mat.create(ary.shape[0], ary.shape[1], CV_32F)
    mat.data = <void*> ary.data
    return mat

# Function to convert cv::Mat to numpy.ndarray
cdef np.ndarray[np.float32_t, ndim=2] Mat2np(Mat mat):
    cdef np.ndarray[np.float32_t, ndim=2] ary
    ary = np.ndarray[np.float32_t, ndim=2](shape=(mat.rows, mat.cols), buffer=<np.float32_t*>mat.data)
    return ary

# Define a Python function that will be used to call the C++ function
def calculate_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                  method="L2-Hys", visualize=False, transform_sqrt=True, flatten=True):
    cdef Mat img = np2Mat(image)
    return hog(img, orientations, pixels_per_cell, cells_per_block, method, visualize, transform_sqrt, flatten)

