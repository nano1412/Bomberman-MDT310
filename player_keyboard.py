import pygame
import math

from bomb import Bomb
from enums.algorithm import Algorithm

# ตอนนี้ทำได้แค่ keyboard อย่างเดียว ต้องทำให้เล่นเองได้ด้วย
class PlayerKeyboard:
    
    # 0 - down
    # 1 - right
    # 2 - up
    # 3 - left
    dire = [[1, 0, 1], [0, 1, 0], [-1, 0, 3], [0, -1, 2]]

    TILE_SIZE = 4

    def __init__(self, player_id,x,y, alg):
        self.algorithm = alg
        self.player_id = player_id

        self.life = True
        self.pos_x = x * PlayerKeyboard.TILE_SIZE
        self.pos_y = y * PlayerKeyboard.TILE_SIZE
        self.direction = 0
        # frame คือ รูปที่ใช้สำหรับ animation มี 0-2
        self.frame = 0
        self.animation = []
        self.range = 3
        # bomb_limit จำนวน bomb ที่สามารถวางได้ในแต่ละครั้ง
        self.bomb_limit = 1
        self.__score = 1000
        self.set_bomb = 0

            

    def move(self, dx, dy, grid, enemys):
        tempx = int(self.pos_x / PlayerKeyboard.TILE_SIZE)
        tempy = int(self.pos_y / PlayerKeyboard.TILE_SIZE)
        # test: __score every move
        # self.__score -= 1

        map = []

        for i in range(len(grid)):
            map.append([])
            for j in range(len(grid[i])):
                map[i].append(grid[i][j])

        for x in enemys:
            if x == self:
                continue
            elif not x.life:
                continue
            else:
                map[int(x.pos_x / PlayerKeyboard.TILE_SIZE)][int(x.pos_y / PlayerKeyboard.TILE_SIZE)] = 2

        if self.pos_x % PlayerKeyboard.TILE_SIZE != 0 and dx == 0:
            if self.pos_x % PlayerKeyboard.TILE_SIZE == 1:
                self.pos_x -= 1
            elif self.pos_x % PlayerKeyboard.TILE_SIZE == 3:
                self.pos_x += 1
            return
        if self.pos_y % PlayerKeyboard.TILE_SIZE != 0 and dy == 0:
            if self.pos_y % PlayerKeyboard.TILE_SIZE == 1:
                self.pos_y -= 1
            elif self.pos_y % PlayerKeyboard.TILE_SIZE == 3:
                self.pos_y += 1
            return

        # right
        if dx == 1:
            if map[tempx+1][tempy] == 0:
                self.pos_x += 1
        # left
        elif dx == -1:
            tempx = math.ceil(self.pos_x / PlayerKeyboard.TILE_SIZE)
            if map[tempx-1][tempy] == 0:
                self.pos_x -= 1

        # bottom
        if dy == 1:
            if map[tempx][tempy+1] == 0:
                self.pos_y += 1
        # top
        elif dy == -1:
            tempy = math.ceil(self.pos_y / PlayerKeyboard.TILE_SIZE)
            if map[tempx][tempy-1] == 0:
                self.pos_y -= 1

    def plant_bomb(self, map):
        b = Bomb(self.range, round(self.pos_x / PlayerKeyboard.TILE_SIZE), round(self.pos_y / PlayerKeyboard.TILE_SIZE), map, self)
        return b

    def check_death(self, exp):
        for e in exp:
            for s in e.sectors:
                if int(self.pos_x / PlayerKeyboard.TILE_SIZE) == s[0] and int(self.pos_y / PlayerKeyboard.TILE_SIZE) == s[1]:
                    self.life = False
                    print("Player ", self.player_id, " is dead")
                    if e.bomber == self:
                        print("Player ", self.player_id, " is dead by himself")
                        self.__score -= self.__score//2
                    else:
                        temp = self.__score//2
                        self.__score -= temp
                        e.bomber.__score += temp
                    if self.__score < 0:
                        self.__score = 0
                    self.reborn()
                    return
                
    def get_score(self):
        return self.__score

    def load_animations(self, scale):
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale

        f1 = pygame.image.load('images/hero/p1f0.png')
        f2 = pygame.image.load('images/hero/p1f1.png')
        f3 = pygame.image.load('images/hero/p1f2.png')

        f1 = pygame.transform.scale(f1, (resize_width, resize_height))
        f2 = pygame.transform.scale(f2, (resize_width, resize_height))
        f3 = pygame.transform.scale(f3, (resize_width, resize_height))

        front.append(f1)
        front.append(f2)
        front.append(f3)

        r1 = pygame.image.load('images/hero/p1r0.png')
        r2 = pygame.image.load('images/hero/p1r1.png')
        r3 = pygame.image.load('images/hero/p1r2.png')

        r1 = pygame.transform.scale(r1, (resize_width, resize_height))
        r2 = pygame.transform.scale(r2, (resize_width, resize_height))
        r3 = pygame.transform.scale(r3, (resize_width, resize_height))

        right.append(r1)
        right.append(r2)
        right.append(r3)

        b1 = pygame.image.load('images/hero/p1b0.png')
        b2 = pygame.image.load('images/hero/p1b1.png')
        b3 = pygame.image.load('images/hero/p1b2.png')

        b1 = pygame.transform.scale(b1, (resize_width, resize_height))
        b2 = pygame.transform.scale(b2, (resize_width, resize_height))
        b3 = pygame.transform.scale(b3, (resize_width, resize_height))

        back.append(b1)
        back.append(b2)
        back.append(b3)

        l1 = pygame.image.load('images/hero/p1l0.png')
        l2 = pygame.image.load('images/hero/p1l1.png')
        l3 = pygame.image.load('images/hero/p1l2.png')

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
