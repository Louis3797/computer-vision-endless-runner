import numpy as np
import pygame
import random

class Tilemap:
    def __init__(self, tileset, size=(10, 20), tile_size=(64, 64), rect=None, scroll_speed=5, buffer_size=60):
        self.size = size
        self.tileset = tileset
        self.tile_size = tile_size

        self.scroll_speed = scroll_speed
        self.buffer_size = buffer_size
        self.visible_rows = min(self.buffer_size, size[1])

        h, w = self.size
        self.image = pygame.Surface((self.tile_size[0] * w, self.tile_size[1] * h))
        if rect:
            self.rect = pygame.Rect(rect)
        else:
            self.rect = self.image.get_rect()

        # Circular buffer to manage map rows
        self.map_buffer = np.zeros((self.buffer_size, w), dtype=int)
        self.buffer_index = 0

        self.generate()
        self.update_count = 0

    def generate(self):
        mid = (len(self.map_buffer[0]) - 1) // 2

        # Randomly select between 2 and 3 for each index
        probabilities = [0.8, 0.2]  # 80% chance of 2, 20% chance of 3
        self.map_buffer[:, mid - 1:mid + 2] = np.random.choice([2, 3], size=(self.buffer_size, 3), p=probabilities)

        # Add slope edges
        # Left edges
        self.map_buffer[:, mid - 2] = 1
        # Right edges
        self.map_buffer[:, mid + 2] = 4

    def update(self):
        self.update_count += 1

        # Move the buffer index and replace the row that moves off the top
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size

        if self.update_count % self.scroll_speed == 0:
            # Randomly generate a new row for the buffer
            new_row = np.zeros(self.size[1], dtype=int)

            mid = (len(new_row) - 1) // 2

            # Make the slopes
            new_row[mid - 1:mid + 2] = 2

            # Add slope edges
            # Left edges
            new_row[mid - 2] = 1
            # Right edges
            new_row[mid + 2] = 4

            self.map_buffer[self.buffer_index] = new_row

    def render(self):
        m, n = self.size
        for i in range(m - self.visible_rows, m):
            for j in range(n):
                tile = self.tileset.tiles[
                    self.map_buffer[(self.buffer_index + i - (m - self.visible_rows)) % self.buffer_size, j]]
                scaled_tile = pygame.transform.scale(tile, self.tile_size)
                self.image.blit(scaled_tile, (j * self.tile_size[0], (i - (m - self.visible_rows)) * self.tile_size[1]))

    def __str__(self):
        return f'{self.__class__.__name__} {self.size}'



