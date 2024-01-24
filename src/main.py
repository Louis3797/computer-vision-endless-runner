import math
import random

import cv2
import numpy as np
import pygame
from src.cv.tracking.tracker import Tracker
from src.entities.Character import Character
from src.entities.Coin import Coin
from src.entities.Obstacle import Obstacle
from src.entities.Trail import Trail
from src.utils.sections import get_section_with_most_boxes
from src.utils.constants import (WIDTH, HEIGHT, FPS, SCROLL_SPEED, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_RECT_MARGIN, \
                                 WINDOW_CAPTION)
from src.bindings.tracking.build.tracking import HOGDescriptor, PersonDetector # ignore this error

# Paths to character sprites and images
character_sprite_paths = [
    "../assets/images/kenney_tiny-ski/Tiles/tile_0071.png",
    "../assets/images/kenney_tiny-ski/Tiles/tile_0070.png"
]
coin_sprite_paths = [
    "../assets/images/coin_1.png",
    "../assets/images/coin_2.png"
]

rock_sprite_paths = [
    "../assets/images/kenney_tiny-ski/Tiles/tile_0081.png"
]

gondola_image_foreground = "../assets/images/ski_gondola.png"
background_image = "../assets/images/ground.png"

lanes = [(((WIDTH - (64 * 3)) // 2) + x * 64) for x in range(0, 3)]

coin_spawn_delay = 2000  # Delay between spawns in milliseconds
coin_last_spawn_time = 0
collected_coins_score = 0

rock_spawn_delay = 2000
rock_last_spawn_time = 0

background_frames = []
frame_count = 0

cv2.setUseOptimized(True)
cv2.setNumThreads(8)


def load_digit_images(scale_factor=3):
    digit_images = [pygame.transform.scale(
        pygame.image.load(f"../assets/images/kenney_tiny-ski/Tiles/tile_00{84 + i}.png").convert_alpha(),
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


def main():
    global coin_last_spawn_time, collected_coins_score, rock_last_spawn_time
    pygame.init()
    pygame.mixer.init()

    # Initialize Tracking classes
    hogDescriptor = HOGDescriptor(9, (8, 8), (3, 3), 4, False, False, True, True)

    scale_factor = 0.3
    size = (96, 160)
    detection_threshold_1 = 0.5
    overlap_threshold = 0.3
    bgs_history = 500
    bgs_threshold = 15
    bgs_detectShadows = False
    bgs_learning_rate = 0.01
    bgs_shadow_threshold = 0.5

    personDetector = PersonDetector(
        "../src/cv/svm_models/svm_model_inria+neg_tt+daimler_16Kpos_15Kneg_no_flipped.xml",
        hogDescriptor, scale_factor,
        size, detection_threshold_1, overlap_threshold, bgs_history, bgs_threshold, bgs_detectShadows,
        bgs_learning_rate, bgs_shadow_threshold)

    tracker = Tracker(max_age=50,
                      min_hits=3,
                      iou_threshold=0.3, iou_metric_weight=2.0,
                      orb_metric_weight=1.0)
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

        # kill coin if its out of the image
        for coin in coins:
            if coin.rect.y < -64:
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
                character.rect.x = 448
                character.rect.y = HEIGHT // 2 - 100
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
        coin_text = render_score(collected_coins_score, digit_images)
        screen.blit(coin_text, (WIDTH / 2 - (coin_text.get_width() / 2), 10))

        # Capture a frame from the camera
        ret, frame = cap.read()
        if ret:
            frame = np.flip(frame, 1)
            output = frame.copy()

            # The tracking takes place here
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_grey = cv2.resize(grey, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            detections = personDetector.detect(resized_grey, 10000, 3, 7, 9, 9)

            rects = np.array(detections[0])  # rect format is [x, y, width, height]
            confidenceScores = np.array(detections[1])  # array of float - example [0.87, 1.0, 0.9]

            if len(rects) > 0:
                rects[:, 2:4] += rects[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

                tracks = tracker.update(frame, rects, confidenceScores)

                temp_bboxes = [t[:4].astype(int) for t in tracks]

                for t in tracks:
                    bbox = t[:4].astype(int)
                    track_id = int(t[4])
                    track_conf = t[5]
                    track_color_r = t[6]
                    track_color_g = t[7]
                    track_color_b = t[8]

                    cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (track_color_r, track_color_b, track_color_g), 5)
                    cv2.putText(output, f"id: {track_id} conf: {track_conf:0.2f}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (track_color_r, track_color_b, track_color_g), 2)

                player_move = get_section_with_most_boxes(frame, temp_bboxes)

                character.move_player(player_move)


            # For debugging
            if False:  # Change to True to display the tracking output
                cv2.imshow("output", output)

            # Display tracking output in the right top corner of the game
            result = cv2.resize(output, (CAMERA_WIDTH, CAMERA_HEIGHT))
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result = np.flip(result, 1)
            result = np.rot90(result)
            camera_surface = pygame.surfarray.make_surface(result)
            screen.blit(camera_surface, (WIDTH - CAMERA_WIDTH - CAMERA_RECT_MARGIN, CAMERA_RECT_MARGIN))

        pygame.display.update()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
