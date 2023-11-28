import cv2
import numpy as np

prev_frame = None


def background_subtraction(frame, frame_count, background_frames, num_frames=50, scale_factor=0.5,
                           initial_threshold_median=30,
                           initial_threshold_fd=15, learning_rate=0.02,
                           update_interval=20,
                           bs_weight=0.5, fd_weight=0.5):
    global prev_frame
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Resize the frame to a smaller size

    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count < num_frames:
        # Capture initial frames for background modeling
        background_frames.append(gray_frame)

    elif frame_count % update_interval == 0:
        # Update the background model with the current frame
        background_frames.pop(0)
        background_frames.append(gray_frame)
    # else:
    # Calculate the median frame for background subtraction
    background_median = np.median(np.array(background_frames), axis=0).astype(np.uint8)

    # Calculate the absolute difference between the current frame and the background
    diff = cv2.absdiff(gray_frame, background_median)

    # Apply Gaussian blur to reduce noise in frame difference
    diff = cv2.GaussianBlur(diff, (9, 9), 0)

    # Calculate a dynamic threshold using adaptive thresholding
    threshold_bs = initial_threshold_median + learning_rate * np.mean(diff)

    # Apply adaptive thresholding
    _, thresh_bs = cv2.threshold(diff, threshold_bs, 255, cv2.THRESH_BINARY)

    # Apply morphological opening to reduce noise
    thresh_bs = cv2.morphologyEx(thresh_bs, cv2.MORPH_OPEN, kernel)

    thresh_bs = np.bitwise_not(thresh_bs)

    # Update the background model with a dynamic learning rate
    for i in range(len(background_frames)):
        background_frames[i] = cv2.addWeighted(background_frames[i], 1 - learning_rate, diff, learning_rate, 0)

    result = thresh_bs

    if prev_frame is not None:
        # Calculate frame difference
        frame_diff = cv2.absdiff(prev_frame, gray_frame)

        # Calculate a dynamic threshold for frame difference
        threshold_fd = initial_threshold_fd + learning_rate * np.mean(frame_diff)

        # Apply adaptive thresholding to frame difference
        _, thresh_fd = cv2.threshold(frame_diff, threshold_fd, 255, cv2.THRESH_BINARY)

        # Apply morphological opening to reduce noise in frame difference
        thresh_fd = cv2.morphologyEx(thresh_fd, cv2.MORPH_OPEN, kernel)

        # Combine background subtraction and frame difference with a weighted sum
        result = cv2.addWeighted(thresh_bs, bs_weight, thresh_fd, fd_weight, 0)

    prev_frame = gray_frame

    # Upscale the result for display
    result = cv2.resize(result, None, fx=1 / scale_factor, fy=1 / scale_factor,
                        interpolation=cv2.INTER_NEAREST)

    # Flip the result horizontally
    result = cv2.flip(result, 1)

    thresh = 127
    post_processed_result = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY)[1]

    return post_processed_result
