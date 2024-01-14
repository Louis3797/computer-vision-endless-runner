import pygame


class Character(pygame.sprite.Sprite):
    def __init__(self, x, y, image_paths: list[str], frame_delay=500):
        super().__init__()
        self.x = x
        self.y = y
        self.image_paths = image_paths
        self.num_frames = len(image_paths)
        self.current_frame = 0
        self.frame_delay = frame_delay  # Delay between frames in milliseconds
        self.last_frame_time = 0
        self.image = pygame.image.load(image_paths[0]).convert()
        self.image = pygame.transform.scale(self.image, (64, 64))  # Scale the image to 64x64
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel = 5
        self.frames = []

        # Load and store individual frames from the sprite sheet
        for path in image_paths:
            frame = pygame.image.load(path).convert()
            frame = pygame.transform.scale(frame, (64, 64))
            self.frames.append(frame)

    def update(self, current_time):
        if current_time - self.last_frame_time > self.frame_delay:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.image = self.frames[self.current_frame]
            self.last_frame_time = current_time

    def move(self, direction):
        if direction == 'left' and self.rect.x > 384:
            self.rect.x -= self.vel
        elif direction == 'right' and self.rect.x < 512:
            self.rect.x += self.vel

        elif direction == 'middle' and (self.rect.x > 448 or self.rect.x < 448):
            if self.rect.x < 448:
                self.rect.x += self.vel
            else:
                self.rect.x -= self.vel


