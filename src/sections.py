import cv2

from src.optical_flow import track_optical_flow


def calculate_sections(width, height):
    section1 = (0, 0, width // 3, height)
    section2 = (width // 3, 0, (width // 3) * 2, height)
    section3 = ((width // 3) * 2, 0, width, height)
    return [section1, section2, section3]


def draw_sections(frame, sections):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for rect, color in zip(sections, colors):
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)


def process_frames(frame, dot, sections):
    if dot < sections[0][2]:
        section_text = "Section: 1"
    elif sections[1][0] <= dot < sections[1][2]:
        section_text = "Section: 2"
    else:
        section_text = "Section: 3"

    cv2.putText(frame, section_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    draw_sections(frame, sections)


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error opening camera")
        exit(-1)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    sections = calculate_sections(width, height)

    bbox = [500, 25, 300, 300]

    prev_gray = None
    prev_dot = None

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("No frame captured from camera")
            break

        if prev_gray is None:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        prev_gray, prev_dot, bbox = track_optical_flow(prev_gray, frame, prev_dot, bbox)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.circle(frame, (int(prev_dot[0]), int(prev_dot[1])), 5, (255, 0, 0), -1)

        process_frames(frame, prev_dot[0], sections)

        cv2.imshow("Split Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
