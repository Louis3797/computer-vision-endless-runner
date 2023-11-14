import cv2
import numpy as np

# Initialize the camera capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera or specify another camera index


def background_subtraction(camera_capture, num_frames=50, scale_factor=0.5, initial_threshold=30, learning_rate=0.02,
                           bs_weight=0.5, fd_weight=0.5, min_contour_area=200, min_object_size=100):
    background_frames = []
    frame_count = 0

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = camera_capture.read()
        if not ret:
            break

        # Resize the frame to a smaller size
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count < num_frames:
            # Capture initial frames for background modeling
            background_frames.append(gray_frame)
        else:
            # Calculate the median frame for background subtraction
            background_median = np.median(np.array(background_frames), axis=0).astype(np.uint8)

            # Calculate the absolute difference between the current frame and the background
            diff = cv2.absdiff(gray_frame, background_median)

            # Apply Gaussian blur to reduce noise in frame difference
            diff = cv2.GaussianBlur(diff, (7, 7), 0)

            # Calculate a dynamic threshold using adaptive thresholding
            threshold = initial_threshold + learning_rate * np.mean(diff)

            # Apply adaptive thresholding
            _, thresh_bs = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

            # Apply morphological opening to reduce noise
            thresh_bs = cv2.morphologyEx(thresh_bs, cv2.MORPH_OPEN, kernel)

            # Update the background model with a dynamic learning rate
            for i in range(len(background_frames)):
                background_frames[i] = cv2.addWeighted(background_frames[i], 1 - learning_rate, diff, learning_rate, 0)

            # Calculate frame difference
            frame_diff = cv2.absdiff(gray_frame, cv2.medianBlur(gray_frame, 5))

            # Calculate a dynamic threshold for frame difference
            threshold_fd = threshold + learning_rate * np.mean(frame_diff)

            # Apply adaptive thresholding to frame difference
            _, thresh_fd = cv2.threshold(frame_diff, threshold_fd, 255, cv2.THRESH_BINARY)

            # Apply morphological opening to reduce noise in frame difference
            thresh_fd = cv2.morphologyEx(thresh_fd, cv2.MORPH_OPEN, kernel)

            # Combine background subtraction and frame difference with a weighted sum
            result = cv2.addWeighted(thresh_bs, bs_weight, thresh_fd, fd_weight, 0)
            # result = cv2.addWeighted(thresh_bs, 1 - frame_difference_weight, thresh_fd, frame_difference_weight, 0)
            # Upscale the result for display
            result = cv2.resize(result, None, fx=1 / scale_factor, fy=1 / scale_factor,
                                interpolation=cv2.INTER_NEAREST)

            # Flip the result horizontally
            result = cv2.flip(result, 1)

            # Apply post-processing to fill out the detected human objects
            # post_processed_result = post_process(result, kernel)

            # Apply post-processing to fill out the detected human objects and filter small objects
            post_processed_result = post_process(result, kernel, min_contour_area, min_object_size)

            thresh = 127
            post_processed_result = cv2.threshold(post_processed_result, thresh, 255, cv2.THRESH_BINARY)[1]

            # Display the original frame and the post-processed result
            # cv2.imshow("Live Camera Feed", frame)
            cv2.imshow("Foreground Detection", post_processed_result)

        # Press 'q' to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1


def post_process(thresh, kernel, min_contour_area, min_object_size):
    # Create a mask for the areas of interest (detected objects)
    areas_of_interest = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(areas_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask to fill the selected contours
    filled_mask = np.zeros_like(areas_of_interest)

    # Filter and draw contours with an area above the threshold
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(filled_mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Use bitwise_and to apply the filled mask to the thresholded image
    post_processed_result = cv2.bitwise_and(thresh, filled_mask)

    # Remove small isolated pixels
    post_processed_result = cv2.morphologyEx(post_processed_result, cv2.MORPH_CLOSE, kernel, iterations=2)

    return post_processed_result


# Call the background_subtraction method with custom parameters
background_subtraction(cap, num_frames=50, scale_factor=0.3, initial_threshold=30, learning_rate=0.1, bs_weight=1,
                       fd_weight=1, min_contour_area=100, min_object_size=200)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
