import math
import random

import cv2
import numpy as np
import pygame

from src.entities.Character import Character
from src.entities.Coin import Coin
from src.entities.Obstacle import Obstacle
from src.entities.Trail import Trail
from src.sections import calculate_sections, process_frames, calculate_dot_section
from src.utils.constants import WIDTH, HEIGHT, FPS, SCROLL_SPEED, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_RECT_MARGIN, \
    WINDOW_CAPTION
from src.optical_flow import track_optical_flow

# from src.tracking import HOGDescriptor, PersonDetector, PersonTracker

# Paths to character sprites and images
character_sprite_paths = [
    "assets/images/kenney_tiny-ski/Tiles/tile_0071.png",
    "assets/images/kenney_tiny-ski/Tiles/tile_0070.png"
]
coin_sprite_paths = [
    "assets/images/coin_1.png",
    "assets/images/coin_2.png"
]

rock_sprite_paths = [
    "assets/images/kenney_tiny-ski/Tiles/tile_0081.png"
]

gondola_image_foreground = "assets/images/ski_gondola.png"
background_image = "assets/images/ground.png"

lanes = [(((WIDTH - (64 * 3)) // 2) + x * 64) for x in range(0, 3)]

coin_spawn_delay = 2000  # Delay between spawns in milliseconds
coin_last_spawn_time = 0
collected_coins_score = 0

rock_spawn_delay = 2000
rock_last_spawn_time = 0

background_frames = []
frame_count = 0


def load_digit_images(scale_factor=3):
    digit_images = [pygame.transform.scale(
        pygame.image.load(f"assets/images/kenney_tiny-ski/Tiles/tile_00{84 + i}.png").convert_alpha(),
        (16 * scale_factor, 16 * scale_factor))
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


def move_player(player_move, character):
    if player_move == "Section: 1":
        character.move('left')

    if player_move == "Section: 2":
        character.move('middle')

    if player_move == "Section: 3":
        character.move('right')


def main():
    global coin_last_spawn_time, collected_coins_score, rock_last_spawn_time
    pygame.init()
    pygame.mixer.init()

    # Initialize Tracking classes
    # hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.19
    size = (96, 160)
    stepSize = (10, 10)
    detection_threshold_1 = 0.5
    detection_threshold_2 = 0.5
    overlap_threshold = 0.6
    downscale = 1.15

    # personDetector = PersonDetector(
    #     "/Users/louis/CLionProjects/Tracking/models/svm_model_inria_96_160_with_flipped.xml",
    #     "/Users/louis/CLionProjects/Tracking/cmake-build-debug/svm_model_tt_96_160_with_cropped_10000.xml",
    #     hogDescriptor, scale_factor,
    #     size, stepSize, detection_threshold_1, detection_threshold_2, overlap_threshold,
    #     downscale)
    #
    # personTracker = PersonTracker(personDetector)

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
    rocks = pygame.sprite.Group()
    all_sprites.add(character)

    trail = Trail(max_length=50)

    digit_images = load_digit_images()
    font = pygame.font.Font(None, 36)

    running = True

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    sections = calculate_sections(width, height)

    bbox = [500, 25, 300, 300]
    # TODO GET BOUNDING BOX FROM DETECTION

    prev_gray = None
    prev_dot = None

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
            # print("Collected coin!")
            # print(f"Score: {collected_coins_score}")

        # kill coin if its out of the image
        for coin in coins:
            if coin.rect.y < -64:
                # print("Killed coin")
                coin.kill()

        if current_time - rock_last_spawn_time > rock_spawn_delay:
            # spawn coin
            lane_x = random.choice(lanes)
            rock = Obstacle(lane_x, HEIGHT + (i * 64), SCROLL_SPEED, rock_sprite_paths)
            rocks.add(rock)
            overlapping_rocks = pygame.sprite.spritecollide(rock, coins, True)
            rock_last_spawn_time = current_time

        for rock in rocks:
            if character.rect.colliderect(rock.rect):
                collected_coins_score = 0
            if rock.rect.y < -64:
                # print("Killed rock")
                rock.kill()

        # Add the character frame to the trail
        trail.add_frame(character.rect)
        trail.update()
        trail.draw(screen)  # Draw the trail on the screen
        # Draw all sprites
        all_sprites.draw(screen)
        coins.update(current_time)
        coins.draw(screen)
        rocks.update(current_time)
        rocks.draw(screen)

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
        screen.blit(coin_text, (WIDTH / 2 - (coin_text.get_width() / 2), 10))

        # Capture a frame from the camera
        # ret, frame = cap.read()
        # if ret:
        #
        #     frame = cv2.resize(frame, (365, 205), interpolation=cv2.INTER_AREA)
        #     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        #     trackRes = personTracker.track(grey)
        #
        #     print(trackRes)
        #
        #     if prev_gray is None:
        #         prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        #     prev_gray, prev_dot, bbox = track_optical_flow(prev_gray, frame, prev_dot, bbox)
        #
        #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #
        #     cv2.circle(frame, (int(prev_dot[0]), int(prev_dot[1])), 5, (255, 0, 0), -1)
        #
        #     player_move = process_frames(frame, prev_dot[0], sections)
        #
        #     move_player(player_move, character)
        #     # result = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
        #
        #     cv2.imshow("Split Frame", frame)

        # result = np.rot90(result)
        # camera_surface = pygame.surfarray.make_surface(result)
        # screen.blit(camera_surface, (WIDTH - CAMERA_WIDTH - CAMERA_RECT_MARGIN, CAMERA_RECT_MARGIN))

        pygame.display.update()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
