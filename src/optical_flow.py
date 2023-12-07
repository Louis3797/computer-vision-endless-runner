import cv2
import numpy as np

def track_optical_flow(prev_gray, frame, prev_dot, bbox):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dot_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    dot_prev = np.array([dot_center], dtype=np.float32)
    dot_next, dot_status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, dot_prev, None)

    if dot_next is not None:
        dot_position = dot_next.ravel()
    else:
        dot_position = dot_center

    movement = [dot_position[0] - dot_center[0], dot_position[1] - dot_center[1]]

    bbox[0] += int(movement[0])
    bbox[2] += int(movement[0])

    prev_gray = gray.copy()
    prev_dot = dot_position

    return prev_gray, prev_dot, bbox


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bbox_width, bbox_height = 200, height - 100
    bbox = [(width - bbox_width) // 2, (height - bbox_height) // 2,
            (width + bbox_width) // 2, (height + bbox_height) // 2]

    prev_gray = None
    prev_dot = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_gray is None:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        prev_gray, prev_dot, bbox = track_optical_flow(prev_gray, frame, prev_dot, bbox)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.circle(frame, (int(prev_dot[0]), int(prev_dot[1])), 5, (255, 0, 0), -1)

        cv2.imshow('Optical Flow Tracking', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
