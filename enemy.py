'''
ใช้สำหรับจัดการ ghost หรือศัตรูในเกม
'''
import pygame
import random
from bomb import Bomb
from enums.algorithm import Algorithm

import numpy as np

# ฟังก์ชันนี้ใช้ในการหาระยะห่างระหว่างจุด โดยใช้ manhattan distance
def manhattan_distance_np(point1, point2):
    return np.sum(np.abs(np.array(point2) - np.array(point1)))

class Enemy:
    # TILE_SIZE defines how many pixels correspond to one grid cell (or tile) in the game.
    TILE_SIZE = 4
    dire = [[1, 0, 1], [0, 1, 0], [-1, 0, 3], [0, -1, 2]]

    def __init__(self, x, y, alg, en_id):
        self.life = True
        self.path = []
        self.movement_path = []
        self.pos_x = x * Enemy.TILE_SIZE
        self.pos_y = y * Enemy.TILE_SIZE
        self.start_x = x
        self.start_y = y

        self.direction = 0
        self.frame = 0
        self.animation = []
        self.algorithm = alg
        self.en_id = en_id

    def move(self, map, players, bombs, explosions, enemy):
        # print("-----------------")
        # print("enemy move",(self.pos_x,self.pos_y))
        # print("movement path ",self.en_id,self.movement_path)
        # print("path ",self.en_id,self.path)

        # update grid before move
        grid = self.create_grid(map, players, bombs, explosions, enemy)
        # check whether can move or not ดูว่าตำแหน่งถัดไป สามารถเดินได้หรือไม่
        if len(self.movement_path) > 0 and (grid[self.path[1][0]][self.path[1][1]] != 3 and grid[self.path[1][0]][self.path[1][1]] != 2):
            if self.direction == 0:
                self.pos_y += 1
            elif self.direction == 1:
                self.pos_x += 1
            elif self.direction == 2:
                self.pos_y -= 1
            elif self.direction == 3:
                self.pos_x -= 1

            #ตรวจสอบว่าศัตรูถึงตำแหน่งที่เป็น "จุดกึ่งกลางของ Tile"

            if self.pos_x % Enemy.TILE_SIZE == 0 and self.pos_y % Enemy.TILE_SIZE == 0:
                self.movement_path.pop(0)
                self.path.pop(0)
                if len(self.path) > 1:
                    # ถ้ายังมี path ที่ต้องการเดินอยู่ ให้เช็คก่อนว่าเดินได้หรือไม่
                    # ถ้าเดินไม่ได้ ให้หยุดเคลื่อนที่ และ หา path ใหม่
                    grid = self.create_grid(map, players, bombs, explosions, enemy)
                    next_step = self.path[1]
                    if grid[next_step[0]][next_step[1]] > 1:
                        self.movement_path.clear()
                        self.path.clear()

            # Check if the ghost collides with the players
            # ถ้าศัตรูเดินไปชนผู้เล่น จะทำให้ผู้เล่นเสียคะแนน
            for player in players:
                if self.pos_x == player.pos_x and self.pos_y == player.pos_y:
                    # player.life = False  # Kill the player
                    player.set_score(player.get_score()-10)  # Deduct points from the player
                    if player.get_score() < 0:
                        player.set_score(0)
                    print("Player is losing points!")
                    
        elif len(self.movement_path) > 0 and (grid[self.path[1][0]][self.path[1][1]] == 3 or grid[self.path[1][0]][self.path[1][1]] == 2):
            # ที่ต้องทำแบบนี้ เพราะเดินๆ ไปแล้วเจอ bomb จะเดินไม่ได้ ต้องหาทางใหม่
            self.movement_path.clear()
            self.path.clear()
            self.pos_x = self.pos_x // Enemy.TILE_SIZE * Enemy.TILE_SIZE
            self.pos_y = self.pos_y // Enemy.TILE_SIZE * Enemy.TILE_SIZE
        
        self.frame = (self.frame + 1) % 3

    def reborn(self):
        # ทำการเกิดใหม่ของศัตรู
        # ให้กลับไปที่ตำแหน่งเริ่มต้น
        self.life = True
        self.path = []
        self.movement_path = []
        self.pos_x = self.start_x * Enemy.TILE_SIZE
        self.pos_y = self.start_y * Enemy.TILE_SIZE
        
        self.direction = 0
        self.frame = 0
        print("Enemy reborn!")

    def check_death(self, exp):
        # ตรวจสอบว่าศัตรูตายหรือไม่
        # ถ้าศัตรูตาย ให้ทำการเกิดใหม่
        for e in exp:
            for s in e.sectors:
                if int(self.pos_x/Enemy.TILE_SIZE ) == s[0] and int(self.pos_y/Enemy.TILE_SIZE  ) == s[1]:
                    self.life = False
                    e.bomber.set_score(e.bomber.get_score() + 100)
                    print("Enemy killed by bomb!")
                    self.reborn()
                    return

    def make_move(self,map, bombs, explosions, players, enemy):
        if not self.life:
            return

        if len(self.movement_path) == 0:
            if self.algorithm is Algorithm.RANDOM:
                self.random_move(self.create_grid(map, players, bombs, explosions, enemy))
            else:
                self.manhatton_move(self.create_grid(map, players, bombs, explosions, enemy),players)
        else:
            self.direction = self.movement_path[0]
            self.move(map, players, bombs, explosions, enemy)
    
    def random_move(self, grid):
        # เดินได้ที่ละช่อง แบบสุ่ม 
        # อันดับแรก ถ้าเจอ player ในช่องที่เดินได้ จะเดินไปที่ player นั้น
        # อันดับสอง เดินไปที่ช่องที่เป็น safe หรือ 0 หรือเป็น ghost หรือ 4

        # map = grid
#         •	0 - safe
#         •	1 - unsafe
#         •	2 - destryable = box ทำลายกล่องได้
#         •	3 – unreachable คือ wall + bomb ที่เดินทะลุไม่ได้
#         •	4 – ghost positions
#         •	5 – player positions
        start = [int(self.pos_x/Enemy.TILE_SIZE ), int(self.pos_y/Enemy.TILE_SIZE )]
        self.path = [start]
        random.shuffle(self.dire)
        current = start
        # สุ่มวนลูป 5 รอบ 
        for i in range(5):
            for direction in self.dire:
                next_x = current[0] + direction[0]
                next_y = current[1] + direction[1]
                if 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and grid[next_x][next_y] == 5:
                    # target อยู่ตรงนั้น
                    self.path.append([next_x, next_y])
                    self.movement_path.append(direction[2])
                    current = [next_x, next_y]
                    break
                elif 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and (grid[next_x][next_y] in [0,4]):
                    self.path.append([next_x, next_y])
                    self.movement_path.append(direction[2])
                    current = [next_x, next_y]
                    break

    def manhatton_move(self, grid,players):
        # เดินได้แค่ที่ละ 1 ช่อง ดังนั้น ถ้าเจอสิ่งกีดขวาง หรืออยู่ในแนวระเบิด จะหยุดเดิน
        # จะเดินไปที่ player ที่อยู่ใกล้ที่สุดโดยใช้ manhatton distance และเป็นจุดที่ปลอดภัย

        start = [int(self.pos_x/Enemy.TILE_SIZE ), int(self.pos_y/Enemy.TILE_SIZE )]
        self.path = [start]
        random.shuffle(self.dire)
        current = start
        # map = grid
#         •	0 - safe
#         •	1 - unsafe
#         •	2 - destryable = box ทำลายกล่องได้
#         •	3 – unreachable คือ wall + bomb ที่เดินทะลุไม่ได้
#         •	4 – ghost positions
#         •	5 – player positions
        min_dist = 1000
        min_move = [0,0,-1]
        for player in players:
            for direction in self.dire:
                next_x = current[0] + direction[0]
                next_y = current[1] + direction[1]
                if 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]):
                    dist = manhattan_distance_np([next_x,next_y],[int(player.pos_x/Enemy.TILE_SIZE),int(player.pos_y/Enemy.TILE_SIZE)])
                    if dist < min_dist and 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and (grid[next_x][next_y] == 0 or grid[next_x][next_y] == 5):
                    # if dist < min_dist and 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and (grid[next_x][next_y] <= 2):
                        min_dist = dist
                        min_move = direction

        next_x = current[0] + min_move[0]
        next_y = current[1] + min_move[1]
        self.path.append([next_x, next_y])
        self.movement_path.append(min_move[2])

    def create_grid(self, map, players, bombs, explosions, enemys):
        # สร้าง grid สำหรับการเดินของศัตรู โดยการระบุตำแหน่งที่เดินได้ และไม่สามารถเดินได้ ตามตัวเลขด้านล่าง
        grid = [[0] * len(map[0]) for _ in range(len(map))]

        # map = grid
#         •	0 - safe
#         •	1 - unsafe
#         •	2 - destryable = box ทำลายกล่องได้
#         •	3 – unreachable คือ wall + bomb ที่เดินทะลุไม่ได้
#         •	4 – ghost positions
#         •	5 – player positions


        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j] in [1, 2]:  # Walls or other obstacles
                    grid[i][j] = 3  # Mark as unreachable

        # Mark tiles affected by bombs as dangerous
        for bomb in bombs:
            bomb.get_range(map)
            for x, y in bomb.sectors:
                grid[x][y] = 1  # Mark as dangerous
            grid[bomb.pos_x][bomb.pos_y] = 3  # Bomb position is unreachable, cannot move through it

        # Mark tiles affected by explosions as deadly
        for explosion in explosions:
            for x, y in explosion.sectors:
                grid[x][y] = 3  # Mark as deadly

        # Mark other enemies as number 4
        for other in enemys:
            if other == self or not other.life:
                continue
            grid[int(other.pos_x/Enemy.TILE_SIZE)][int(other.pos_y/Enemy.TILE_SIZE)] = 4

        # Mark the player's position as the target
        for player in players:
            grid[int(player.pos_x/player.TILE_SIZE)][int(player.pos_y/player.TILE_SIZE)] = 5
        
        return grid
    
    def load_animations(self, en, scale):
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale

        image_path = 'images/enemy/e'

        f1 = pygame.image.load(image_path + en + 'f0.png')
        f2 = pygame.image.load(image_path + en + 'f1.png')
        f3 = pygame.image.load(image_path + en + 'f2.png')

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load(image_path + en + 'r0.png')
        r2 = pygame.image.load(image_path + en + 'r1.png')
        r3 = pygame.image.load(image_path + en + 'r2.png')

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load(image_path + en + 'b0.png')
        b2 = pygame.image.load(image_path + en + 'b1.png')
        b3 = pygame.image.load(image_path + en + 'b2.png')

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load(image_path + en + 'l0.png')
        l2 = pygame.image.load(image_path + en + 'l1.png')
        l3 = pygame.image.load(image_path + en + 'l2.png')

        l1 = pygame.transform.scale(l1, (resize_width, resize_height))
        l2 = pygame.transform.scale(l2, (resize_width, resize_height))
        l3 = pygame.transform.scale(l3, (resize_width, resize_height))

        left.append(l1)
        left.append(l2)
        left.append(l3)

        self.animation.append(front)
        self.animation.append(right)
        self.animation.append(back)
        self.animation.append(left)
    

