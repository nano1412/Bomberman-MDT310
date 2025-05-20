'''
Bomberman game - 2 players
Coded by: Kejkaew Thanasuan
Date: 2025-04-16

เริ่มเล่นเกมโดยการรัน main.py
$ python3 main.py
'''
'''
อันนี้เป็น main.py file
ใช้สำหรับกำหนดค่าพื้นฐานของเกม และเรียกใช้งาน game.py
สามารถตั้งค่า map ที่จะใช้ และเลือก algorithm ของผู้เล่นและศัตรูได้
'''
import pygame

import game
from enums.algorithm import Algorithm
from layout import *

WINDOW_SCALE = 0.75

#set frame rate
FPS = 20

# เลือก map ที่จะใช้ จาก map folder
map_file = './map/grid_test.txt'

# set algorithm ของ player
'''
Algorithm ที่ใช้ในการเล่นของ bomberman สามารถเลือกได้ 3 แบบ
1. RANDOM สุ่มอย่างเดียว
2. PLAYER = คุณควบคุมเองโดยใช้ keyboard
3. YourAlgorithm = อันที่คุณเขียนเอง

Algorithm ที่ใช้ในการเล่นของ enemy สามารถเลือกได้ 2 แบบ
1. MANHATTAN = Manhattan distance
2. RANDOM = Random

ตัวอย่างการ setting
player_alg1 = Algorithm.DFS
player_alg1 = Algorithm.YourAlgorithm
player_alg2 = Algorithm.RANDOM
player_alg1 = Algorithm.PLAYER (อันนี้คือบอกว่าเล่นเองโดยใช้ keyboard : ปุ่มลูกศรขึ้นลงซ้ายขวา และ spacebar สำหรับวางระเบิด)

Note: เราต้องสามารถเล่นเองโดยใช้ keyboard ได้แค่ bomberman player 1 เท่านั้น
'''

player_alg1 = Algorithm.YourAlgorithm
player_alg2 = Algorithm.DFS
en1_alg = Algorithm.RANDOM
en2_alg = Algorithm.MANHATTAN

GRID_BASE = create_map(read_line(map_file))
w = len(GRID_BASE)
h = len(GRID_BASE[0])

# set window size to be 612x510
pygame.display.init()
current_h = 982
TILE_SIZE = int(current_h * 0.035)
WINDOW_SIZE = (w * TILE_SIZE, (h+2) * TILE_SIZE)

clock = None

show_path = True
surface = pygame.display.set_mode(WINDOW_SIZE)
# เอา algorithm ใสใน list เพื่อส่งให้ game.py
en_alg = [en1_alg,en2_alg] 

def run_game():
    # before running the game, initialize pygame
    pygame.init()
    pygame.display.set_caption('Bomberman')
    clock = pygame.time.Clock()
    game.game_init(surface, show_path, player_alg1,player_alg2, en_alg, TILE_SIZE,GRID_BASE, FPS)


if __name__ == "__main__":
    #run the game directly
    run_game()

