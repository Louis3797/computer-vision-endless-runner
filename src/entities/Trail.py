import pygame

from src.utils.constants import HEIGHT, SCROLL_SPEED


class Trail:
    def __init__(self, max_length):
        self.image = pygame.image.load("../assets/images/kenney_tiny-ski/Tiles/tile_0058.png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (64, 64))
        self.max_length = max_length
        self.trail_images = []  # List to store trail images
        self.image_width, self.image_height = 64, 64  # Width and height of the character image
        self.trail_positions = [(448, HEIGHT // 2 - 132)]

    def add_frame(self, character_rect):
        # Create a new surface with transparency
        trail_image = pygame.Surface((self.image_width, self.image_height), pygame.SRCALPHA)
        # Blit the character image onto the trail surface
        trail_image.blit(self.image, (0, 0))
        # Append the trail image to the list with the correct position
        self.trail_images.append((trail_image, character_rect.topleft))

        # If the trail exceeds the maximum length, remove the oldest frame
        if len(self.trail_images) > self.max_length:
            self.trail_images.pop(0)

        # Store the character's current position for the next frame
        self.trail_positions.append(character_rect.topleft)


    def update(self):
        if self.trail_positions:
            # Simulate y-axis movement by increasing the y-coordinate of all trail positions
            self.trail_positions = [(x, y - SCROLL_SPEED) for x, y in self.trail_positions]

        # Remove the character's previous position if it exceeds the maximum length
        if len(self.trail_positions) > self.max_length:
            self.trail_positions.pop(0)


    def draw(self, screen):
        for (trail_image, position), (x, y) in zip(self.trail_images, self.trail_positions):
            screen.blit(trail_image, (x, y))