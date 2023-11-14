import math

import cv2
import numpy as np
import pygame
import random

from src.cv.bgs import background_subtraction
from src.entities.Character import Character
from src.entities.Coin import Coin
from src.entities.Trail import Trail
from src.utils.constants import WIDTH, HEIGHT, FPS, SCROLL_SPEED, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_RECT_MARGIN, \
    WINDOW_CAPTION

# Paths to character sprites and images
character_sprite_paths = [
    "assets/images/kenney_tiny-ski/Tiles/tile_0071.png",
    "assets/images/kenney_tiny-ski/Tiles/tile_0070.png"
]
coin_sprite_paths = [
    "assets/images/coin_1.png",
    "assets/images/coin_2.png"
]
gondola_image_foreground = "assets/images/ski_gondola.png"
background_image = "assets/images/ground.png"

lanes = [(((WIDTH - (64 * 3)) // 2) + x * 64) for x in range(0, 3)]

coin_spawn_delay = 2000  # Delay between spawns in milliseconds
coin_last_spawn_time = 0
collected_coins_score = 0

background_frames = []
frame_count = 0
prev_frame = None


def load_digit_images(scale_factor=3):
    digit_images = [pygame.transform.scale(
        pygame.image.load(f"assets/images/kenney_tiny-ski/Tiles/tile_00{84 + i}.png").convert_alpha(), (16 * scale_factor, 16 * scale_factor))
                    for i in range(10)]
    return digit_images


def render_score(score, digit_images, spacing=-10):
    score_str = str(score)
    digit_width, digit_height = digit_images[0].get_size()
    total_width = len(score_str) * (digit_width + spacing) - spacing
    rendered_score = pygame.Surface((total_width, digit_height), pygame.SRCALPHA)

    for i, digit_char in enumerate(score_str):
        digit_index = int(digit_char)
        rendered_score.blit(digit_images[digit_index], (i * (digit_width + spacing), 0))

    return rendered_score


def main():
    global coin_last_spawn_time, collected_coins_score
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
    gondola_image = pygame.transform.scale(gondola_image,
                                           (gondola_image.get_width() * 4, gondola_image.get_height() * 4))
    scroll_gondola = 0
    gondola_tiles = math.ceil(HEIGHT / gondola_image.get_height()) + 1

    # Create the character
    character = Character(448, HEIGHT // 2 - 100, character_sprite_paths)
    all_sprites = pygame.sprite.Group()
    coins = pygame.sprite.Group()
    all_sprites.add(character)

    trail = Trail(max_length=50)

    digit_images = load_digit_images()
    font = pygame.font.Font(None, 36)

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

        # Todo create CoinSpawner class that handles coin spawning
        if current_time - coin_last_spawn_time > coin_spawn_delay:
            # spawn coin
            num_coins = random.randint(1, 3)
            lane_x = random.choice(lanes)
            for i in range(num_coins):
                coin = Coin(lane_x, HEIGHT + (i * 64), SCROLL_SPEED, coin_sprite_paths)
                coins.add(coin)
            coin_last_spawn_time = current_time

        collected_coins = pygame.sprite.spritecollide(character, coins, True)
        for _ in collected_coins:
            collected_coins_score += 1
            print("Collected coin!")
            print(f"Score: {collected_coins_score}")

        # kill coin if its out of the image
        for coin in coins:
            if coin.rect.y < -64:
                print("Killed coin")
                coin.kill()

        # Add the character frame to the trail
        trail.add_frame(character.rect)
        trail.update()
        trail.draw(screen)  # Draw the trail on the screen
        # Draw all sprites
        all_sprites.draw(screen)
        coins.update(current_time)
        coins.draw(screen)

        # Draw the gondola tiles
        b = 0
        while b < gondola_tiles:
            screen.blit(gondola_image, (0, gondola_image.get_height() * b - scroll_gondola))
            b += 1
        scroll_gondola += SCROLL_SPEED

        if scroll_gondola > gondola_image.get_height():
            scroll_gondola = 0

            # Display the collected coin count
        # coin_text = font.render(f"Coins: {collected_coins_score}", True, (0, 0, 0))
        coin_text = render_score(collected_coins_score, digit_images)
        screen.blit(coin_text, (WIDTH /2 - (coin_text.get_width() /2), 10))

        # Capture a frame from the camera
        ret, frame = cap.read()
        if ret:
            # Todo put frames in algo
            frame = np.rot90(frame)
            # frame = np.rot90(frame)
            # frame = np.rot90(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = background_subtraction(frame, frame_count, background_frames, prev_frame, num_frames=100,
                                                    scale_factor=0.3, initial_threshold_median=45,
                                                    initial_threshold_fd=25, learning_rate=0.1,
                                                    update_interval=20, bs_weight=0.7,
                                                    fd_weight=1, min_contour_area=200, min_object_size=100)

            if result is not None:
                result = cv2.resize(result, (CAMERA_WIDTH, CAMERA_HEIGHT))
                camera_surface = pygame.surfarray.make_surface(result)
                screen.blit(camera_surface, (WIDTH - CAMERA_WIDTH - CAMERA_RECT_MARGIN, CAMERA_RECT_MARGIN))

        pygame.display.update()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
