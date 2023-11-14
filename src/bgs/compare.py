import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

from bgs import background_subtraction


def save_to_csv(results, output_csv):
    """
    Save results to a CSV file.

    Parameters:
    - results: List of tuples (algorithm_name, recognized_pixels)
    - output_csv: Path to the CSV file
    """

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Algorithm', 'Recognized Pixels'])

        for algorithm_name, recognized_pixels in results:
            csv_writer.writerow([algorithm_name] + recognized_pixels)


def compare_algorithms(video_path, algorithms, output_csv):
    """
    Compare background subtraction algorithms and visualize the results.

    Parameters:
    - video_path: Path to the input video for testing
    - algorithms: List of background subtraction algorithms (functions) to compare

    Returns:
    - None (Displays the Matplotlib plot)
    """

    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()

    results = []

    for algorithm in algorithms:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video capture to the beginning
        algorithm_name, bg_subtractor = algorithm
        recognized_pixels = []

        background_frames = []
        frame_count = 0
        prev_frame = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            mask = np.zeros(frame.shape)
            if algorithm_name == "Custom":

                # static bg
                mask = background_subtraction(frame, frame_count, background_frames, prev_frame, num_frames=50,
                                              scale_factor=0.3, initial_threshold_median=25,
                                              initial_threshold_fd=50, learning_rate=0.05,
                                              update_interval=20, bs_weight=0.3,
                                              fd_weight=0.7, min_contour_area=200, min_object_size=100)



            else:
                # Apply the background subtraction algorithm
                mask = bg_subtractor.apply(frame)

            # Count the number of recognized pixels
            recognized_pixels.append(np.sum(mask > 0))
            frame_count += 1
        results.append((algorithm_name, recognized_pixels))

    print(results)
    # Save the results to a CSV file
    save_to_csv(results, output_csv)

    # Plot the results
    plot_results(results)

    cap.release()
    cv2.destroyAllWindows()


def plot_results(results):
    """
    Plot the results of the background subtraction algorithms.

    Parameters:
    - results: List of tuples (algorithm_name, recognized_pixels)
    """

    plt.figure(figsize=(10, 6))

    for algorithm_name, recognized_pixels in results:
        plt.plot(recognized_pixels, label=algorithm_name)

    plt.title('Comparison of Background Subtraction Algorithms')
    plt.xlabel('Frame Number')
    plt.ylabel('Recognized Pixels')
    plt.legend()
    plt.show()


mog2_algorithm = ('MOG2', cv2.createBackgroundSubtractorMOG2())
knn_algorithm = ('KNN', cv2.createBackgroundSubtractorKNN())
custom_algorithm = ('Custom', None)  # The custom algorithm doesn't require a pre-built subtractor
algorithms_to_compare = [mog2_algorithm, knn_algorithm, custom_algorithm]

# Specify the path to the input video for testing
video_path = '/Users/louis/Documents/Uni/Semester 5/Computer Vision/static_bg.mov'

# Compare the algorithms and plot the results
compare_algorithms(video_path, algorithms_to_compare, "recognized_pixel_static_bg.csv")
