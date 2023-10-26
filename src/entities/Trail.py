import pygame
class Trail:
    def __init__(self, max_length):
        self.image = pygame.image.load("assets/images/kenney_tiny-ski/Tiles/tile_0058.png").convert()
        self.image = pygame.transform.scale(self.image, (64, 64))
        self.max_length = max_length
        self.trail_images = []  # List to store trail images
        self.image_width, self.image_height = 64, 64  # Width and height of the character image

    def add_frame(self, character_rect):
        # Create a new surface with transparency
        trail_image = pygame.Surface((self.image_width, self.image_height), pygame.SRCALPHA)
        # Blit the character image onto the trail surface
        trail_image.blit(self.image, (0, 0))
        # Append the trail image to the list
        self.trail_images.append((trail_image, character_rect.topleft))

        # If the trail exceeds the maximum length, remove the oldest frame
        if len(self.trail_images) > self.max_length:
            self.trail_images.pop(0)

    def draw(self, screen):
        for trail_image, position in reversed(self.trail_images):

            screen.blit(trail_image, (position[0], position[1] - 32))