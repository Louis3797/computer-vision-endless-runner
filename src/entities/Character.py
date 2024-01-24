from enum import Enum

import pygame

from src.utils.sections import Sections


class CharacterMovement(Enum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3


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
        self.vel = 8
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

    def _move(self, direction: CharacterMovement):
        if direction == CharacterMovement.LEFT and self.rect.x > 384:
            self.rect.x -= self.vel
        elif direction == CharacterMovement.RIGHT and self.rect.x < 512:
            self.rect.x += self.vel

        elif direction == CharacterMovement.MIDDLE and (self.rect.x > 448 or self.rect.x < 448):
            if self.rect.x < 448:
                self.rect.x += self.vel
            else:
                self.rect.x -= self.vel

    def move_player(self, player_move):
        if player_move == Sections.LEFT:
            self._move(CharacterMovement.LEFT)

        if player_move == Sections.MID:
            self._move(CharacterMovement.MIDDLE)

        if player_move == Sections.RIGHT:
            self._move(CharacterMovement.RIGHT)



