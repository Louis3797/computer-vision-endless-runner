import pygame
import cv2
import numpy as np

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.camera = cv2.VideoCapture(0)  # Use the default camera (usually the webcam)

    def update(self):
        # Read a frame from the camera
        ret, frame = self.camera.read()
        if not ret:
            return

        # Convert the OpenCV frame to a Pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        # Display the camera input on the Pygame window
        cam_width, cam_height = frame.get_size()
        self.screen.blit(frame, (10, 10))  # Adjust the position as needed

    def draw(self):
        pygame.display.flip()

    def __del__(self):
        self.camera.release()
