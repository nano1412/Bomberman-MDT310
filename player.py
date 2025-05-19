import pygame
import random
from bomb import Bomb
from enums.algorithm import Algorithm


class Player:

    # 0 - down
    # 1 - right
    # 2 - up
    # 3 - left
    dire = [[0, 1, 0],[1, 0, 1],  [0, -1, 2],[-1, 0, 3]]

    TILE_SIZE = 4

    def __init__(self, player_id, x, y, alg):
        '''
        สำหรับ setting ค่าพื้นฐานของ player
        '''
        self.player_id = player_id

        self.life = True
        self.path = []
        self.movement_path = []
        self.pos_x = x * Player.TILE_SIZE
        self.pos_y = y * Player.TILE_SIZE
        self.direction = 0
        # frame คือ รูปที่ใช้สำหรับ animation มี 0-2
        self.frame = 0
        self.animation = []
        self.range = 3
        # bomb_limit จำนวน bomb ที่สามารถวางได้ในแต่ละครั้ง
        self.bomb_limit = 1
        # plant คือ ต้องการวาง bomb หรือไม่
        self.plant = [False] * self.bomb_limit
        self.algorithm = alg
        self.__score = 1000
        self.step = 0
        self.start_x = x
        self.start_y = y
        self.just_dead = 0
        self.set_bomb = 0

    def move(self, map, bombs, explosions, players,enemy):
        '''
        สำหรับการเคลื่อนที่ของ player แต่ละตัว
        ค่าของ direction เคลื่อนที่ได้ 4 ทิศทาง ใส่ค่าแทนด้วยตัวเลข
        0 - down
        1 - right
        2 - up
        3 - left
        ดูตาม dire = [[0, 1, 0],[1, 0, 1],  [0, -1, 2],[-1, 0, 3]]
        '''
   
        # print("-----------------")
        # print("player move",(self.pos_x,self.pos_y))
        # print("movement path ",self.player_id,self.movement_path)
        # print("path ",self.player_id,self.path)
        
        # อัพเดต grid ทุกครั้งที่เคลื่อนที่ จะได้ไม่เดินหลุดไปที่ที่ไม่สามารถเดินได้
        grid = self.create_grid(map, bombs, explosions, players,enemy)

        if len(self.movement_path) > 0 and (grid[self.path[1][0]][self.path[1][1]] != 3 and grid[self.path[1][0]][self.path[1][1]] != 2):

            if self.direction == 0:
                self.pos_y += 1
            elif self.direction == 1:
                self.pos_x += 1
            elif self.direction == 2:
                self.pos_y -= 1
            elif self.direction == 3:
                self.pos_x -= 1
            
            self.step += 1

            if self.pos_x % Player.TILE_SIZE == 0 and self.pos_y % Player.TILE_SIZE == 0:
                self.movement_path.pop(0)
                self.path.pop(0)
                if len(self.path) > 1:
                    grid = self.create_grid(map, bombs, explosions, players,enemy)
                    next = self.path[1]
                    if grid[next[0]][next[1]] > 1:
                        self.movement_path.clear()
                        self.path.clear()
        elif len(self.movement_path) > 0 and (grid[self.path[1][0]][self.path[1][1]] == 3 or grid[self.path[1][0]][self.path[1][1]] == 2):
            self.movement_path.clear()
            self.path.clear()
            self.pos_x = self.pos_x // Player.TILE_SIZE * Player.TILE_SIZE
            self.pos_y = self.pos_y // Player.TILE_SIZE * Player.TILE_SIZE

        self.frame = (self.frame + 1) % 3

    def make_move(self, map, bombs, explosions, players, enemy):
        '''
        สำหรับกำหนดวิธีการเคลื่อนที่ของ player และการวาง bomb
        เงื่อนไขการวาง bomb คือ ถ้าไม่มีการเคลื่อนที่ในรอบนี้
        จะวาง bomb ในตำแหน่งที่ player อยู่
        ถ้ามีการเคลื่อนที่ จะไม่วาง bomb แต่จะเดินไปตาม path ที่กำหนด
        - map คือ grid ที่ใช้ในการเคลื่อนที่มาจากแต่มาจาก generate_map ต้องมีการอัพเดตอีกเพื่อให้ได้ข้อมูลครบ
        - bombs คือ ข้อมูลของระเบิดทั้งหมดที่ถูกวางในเกม
        - explosions คือ ข้อมูลของการระเบิดทั้งหมดที่เกิดขึ้นในเกม
        - players คือ ข้อมูลของผู้เล่นทั้งหมดในเกม
        - enemy คือ ข้อมูลของศัตรูทั้งหมดในเกม
        '''
        if not self.life:
            return
        if len(self.movement_path) == 0:
            '''
            ถ้าไม่มีการเคลื่อนที่ในรอบนี้ จะวาง bomb ในตำแหน่งที่ player อยู่
            และทำการสร้าง path สำหรับเคลื่อนที่ใหม่ตาม algorithm ที่กำหนด
            '''
            for i in range(len(self.plant)):
                if self.plant[i] and self.set_bomb < self.bomb_limit:
                    bombs.append(self.PlantBomb(map))
                    self.plant[i] = False
                    map[int(self.pos_x / Player.TILE_SIZE)][int(self.pos_y / Player.TILE_SIZE)] = 3
        
            if self.algorithm is Algorithm.YourAlgorithm:
                self.your_algorithm(self.create_grid(map, bombs, explosions, players,enemy))
            else:
                self.random_move(self.create_grid(map, bombs, explosions, players,enemy))

        else:
            '''
            เดินไปตาม path ที่กำหนด
            '''
            self.direction = self.movement_path[0]
            self.move(map, bombs, explosions, players,enemy)

    def PlantBomb(self, map):
        '''
        วางระเบิดในตำแหน่งที่ player อยู่
        '''
        b = Bomb(self.range, round(self.pos_x / Player.TILE_SIZE), round(self.pos_y / Player.TILE_SIZE), map, self)
        self.set_bomb += 1
        return b

    def check_death(self, exp):
        '''
        เช็คว่าตายหรือไม่
        '''
        for e in exp:
            for s in e.sectors:
                if int(self.pos_x / Player.TILE_SIZE) == s[0] and int(self.pos_y / Player.TILE_SIZE) == s[1]:
                    # นับไปอีก 3 frames ค่อยตายใหม่
                    if self.just_dead > 0:
                        self.just_dead -= 1
                        return
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
                    self.just_dead = 3

                    self.reborn()
                    return
    def reborn(self):
        '''
        สำหรับ setting ค่าพื้นฐานของ player ที่เกิดใหม่
        '''
        self.life = True
        self.path = []
        self.movement_path = []
        self.pos_x = self.start_x * Player.TILE_SIZE
        self.pos_y = self.start_y * Player.TILE_SIZE
        
        self.direction = 0
        self.frame = 0
        self.bomb_limit = 1
        self.plant = [False] * self.bomb_limit
        print("player",self.player_id," reborn!")

    def random_move(self, grid):
        '''
        เดินได้ที่ละช่อง แบบสุ่ม 
        อันดับแรก ถ้าเจอ player ในช่องที่เดินได้ จะเดินไปที่ player นั้น
        อันดับสอง เดินไปที่ช่องที่เป็น safe หรือ 0 หรือเป็น ghost หรือ 4

        map = grid
        •	0 - safe (ปลอดภัย เดินไปได้)
        •	1 - unsafe (ไม่ปลอดภัย เดินไปไม่ได้)
        •	2 - destryable (box ที่สามารถทำลายได้)
        •	3 – unreachable (คือ wall + bomb ที่เดินทะลุไม่ได้)
        •	4 – ghost positions (ตำแหน่งของศัตรู)
        •	5 – player positions (ตำแหน่งของผู้เล่นทั้งหมด)
        '''
        start = [int(self.pos_x/Player.TILE_SIZE ), int(self.pos_y/Player.TILE_SIZE )]
        self.path = [start]
        new_choice = [self.dire[0], self.dire[1], self.dire[2], self.dire[3],[0,0,-1]]
        random.shuffle(new_choice)
        current = start
        for i in range(3):
            for direction in new_choice:
                next_x = current[0] + direction[0]
                next_y = current[1] + direction[1]
                if self.bomb_limit > 0 and direction[2] == -1:
                    for i in range(len(self.plant)):
                        if not self.plant[i]:
                            self.plant[i] = True
                            break
                elif 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and grid[next_x][next_y] in [4,5]:
                    # target อยู่ตรงนั้น
                    self.path.append([next_x, next_y])
                    self.movement_path.append(direction[2])
                    current = [next_x, next_y]
                    break
                elif 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and grid[next_x][next_y] not in [2,3]:
                    self.path.append([next_x, next_y])
                    self.movement_path.append(direction[2])
                    current = [next_x, next_y]
                    break

    def create_grid(self, map, bombs, explosions, players, enemys):
        '''
        สำหรับการสร้าง grid สำหรับการเดินของ player 
        โดยการใส่ข้อมูลให้ละเอียดมากขึ้น เป็นข้อมูลที่มาจาก map, bombs, explosions, players และ enemys ลงไปใน grid
        และทำการกำหนดให้
         map ก็คือ grid
        •	0 - safe (ปลอดภัย เดินไปได้)
        •	1 - unsafe (ไม่ปลอดภัย เดินไปไม่ได้)
        •	2 - destryable (box ที่สามารถทำลายกล่องได้)
        •	3 – unreachable (คือ wall + bomb ที่เดินทะลุไม่ได้)
        •	4 – ghost positions (ตำแหน่งของศัตรู)
        •	5 – player positions (ตำแหน่งของผู้เล่นทั้งหมด)
        '''
        # grid size 18x13, i = 18, j = 13
        grid = [[0] * len(map[0]) for r in range(len(map))]

        for b in bombs:
            b.get_range(map)
            for x in b.sectors:
                grid[x[0]][x[1]] = 1
            grid[b.pos_x][b.pos_y] = 3

        for e in explosions:
            for s in e.sectors:
                grid[s[0]][s[1]] = 3

        for i in range(len(map)):
            for j in range(len(map[i])):                
                if map[i][j] == 1: # 1 เป็นกำแพง
                    grid[i][j] = 3
                elif map[i][j] == 2: # 2 เป็นกล่อง
                    grid[i][j] = 2

        for x in enemys:
            grid[int(x.pos_x / Player.TILE_SIZE)][int(x.pos_y / Player.TILE_SIZE)] = 4

        for x in players:
            if x == self:
                continue
            elif not x.life:
                continue
            else:
                grid[int(x.pos_x / Player.TILE_SIZE)][int(x.pos_y / Player.TILE_SIZE)] = 5

        return grid

    def get_score(self):
        return self.__score
    
    def set_score(self, score):
        self.__score = score
    
    def load_animations(self, en, scale):
        '''
        สำหรับการโหลด animation ของ player
        '''
        front = []
        back = []
        left = []
        right = []
        resize_width = scale
        resize_height = scale

        
        image_path = 'images/hero/p'

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

