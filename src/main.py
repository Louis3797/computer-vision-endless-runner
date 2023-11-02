import math

import cv2
import numpy as np
import pygame

from src.entities.Character import Character
from src.entities.Trail import Trail
from src.utils.constants import WIDTH, HEIGHT, FPS, SCROLL_SPEED, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_RECT_MARGIN, \
    WINDOW_CAPTION

# Paths to character sprites and images
character_sprite_paths = [
    "assets/images/kenney_tiny-ski/Tiles/tile_0071.png",
    "assets/images/kenney_tiny-ski/Tiles/tile_0070.png"
]
gondola_image_foreground = "assets/images/ski_gondola.png"
background_image = "assets/images/ground.png"

def main():
    pygame.init()
    pygame.mixer.init()

    # Initialize Pygame screen and clock
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(WINDOW_CAPTION)
    clock = pygame.time.Clock()

    # Initialize the camera capture
    cap = cv2.VideoCapture(0)


    # Load and scale ground image
    ground_image = pygame.image.load(background_image).convert()
    ground_image = pygame.transform.scale(ground_image, (ground_image.get_width() * 4, ground_image.get_height() * 4))
    scroll_ground = 0
    ground_tiles = math.ceil(HEIGHT / ground_image.get_height()) + 1

    # Load and scale ski gondola image
    gondola_image = pygame.image.load(gondola_image_foreground).convert_alpha()
    gondola_image = pygame.transform.scale(gondola_image, (gondola_image.get_width() * 4, gondola_image.get_height() * 4))
    scroll_gondola = 0
    gondola_tiles = math.ceil(HEIGHT / gondola_image.get_height()) + 1

    # Create the character
    character = Character(448, HEIGHT // 2 - 100, character_sprite_paths)
    all_sprites = pygame.sprite.Group()
    all_sprites.add(character)

    trail = Trail(max_length=50)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                running = False
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            character.move('left')
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            character.move('right')

        # Clear the screen
        screen.fill((0, 0, 0))

        # Draw the ground tiles
        i = 0
        while i < ground_tiles:
            screen.blit(ground_image, (0, ground_image.get_height() * i - scroll_ground))
            i += 1
        scroll_ground += SCROLL_SPEED

        if scroll_ground > ground_image.get_height():
            scroll_ground = 0

        # Update character animation
        current_time = pygame.time.get_ticks()
        character.update(current_time)

        # Add the character frame to the trail
        trail.add_frame(character.rect)
        trail.update()
        trail.draw(screen)  # Draw the trail on the screen
        # Draw all sprites
        all_sprites.draw(screen)

        # Draw the gondola tiles
        b = 0
        while b < gondola_tiles:
            screen.blit(gondola_image, (0, gondola_image.get_height() * b - scroll_gondola))
            b += 1
        scroll_gondola += SCROLL_SPEED

        if scroll_gondola > gondola_image.get_height():
            scroll_gondola = 0

        # Capture a frame from the camera
        ret, frame = cap.read()
        if ret:

            # Todo put frames in algo

            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            frame = np.rot90(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_surface = pygame.surfarray.make_surface(frame)
            screen.blit(camera_surface, (WIDTH - CAMERA_WIDTH - CAMERA_RECT_MARGIN, CAMERA_RECT_MARGIN))

        pygame.display.update()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
