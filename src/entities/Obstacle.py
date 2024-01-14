import pygame


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, x, y, velocity, image_paths: list[str], frame_delay=250):
        super().__init__()
        self.x = x
        self.y = y
        self.image_paths = image_paths
        self.num_frames = len(image_paths)
        self.current_frame = 0
        self.frame_delay = frame_delay  # Delay between frames in milliseconds
        self.last_frame_time = 0
        self.image = pygame.image.load(image_paths[0]).convert()
        self.image = pygame.transform.scale(self.image, (32, 32))  # Scale the image to 64x64
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel = velocity
        self.frames = []

        # Load and store individual frames from the sprite sheet
        for path in image_paths:
            frame = pygame.image.load(path).convert()
            frame = pygame.transform.scale(frame, (32, 32))
            self.frames.append(frame)

    def update(self, current_time):
        if current_time - self.last_frame_time > self.frame_delay:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.image = self.frames[self.current_frame]
            self.last_frame_time = current_time

        # move coin backwards
        self.rect.y -= self.vel
