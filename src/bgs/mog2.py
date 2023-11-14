import cv2

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Open a connection to the camera (you can change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Apply the background subtractor to the current frame
    fg_mask = bg_subtractor.apply(frame)

    # You can further process the foreground mask, e.g., applying morphological operations
    # to clean up the mask or use it for object detection.

    # Display the original frame and the foreground mask
    # cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
