import random
from player import Player

class YourPlayer(Player):
    def your_algorithm(self, grid):
        '''
        อันนี้ให้ใส่ algorithm ที่คุณเขียนเอง
        ตัวแปรที่ควรรู้จัก
        - grid = map ที่ใช้ในการเล่น
        •	0 - safe (ปลอดภัย เดินไปได้)
        •	1 - unsafe (ไม่ปลอดภัย เดินไปไม่ได้)
        •	2 - destryable (box ที่สามารถทำลายได้)
        •	3 – unreachable (คือ wall + bomb ที่เดินทะลุไม่ได้)
        •	4 – ghost positions (ตำแหน่งของศัตรู)
        •	5 – player positions (ตำแหน่งของผู้เล่นทั้งหมด)
        - self.pos_x, self.pos_y = ตำแหน่งของ player
        - self.dire = ทิศทางที่ player สามารถเดินได้
            - [1,0,1] = ขวา
            - [-1,0,3] = ซ้าย
            - [0,-1,2] = ขึ้น
            - [0,1,0] = ลง
            ดูตาม dire = [[0, 1, 0],[1, 0, 1],  [0, -1, 2],[-1, 0, 3]]
        - self.bomb_limit = จำนวน bomb ที่สามารถวางได้
        - self.plant = list ของ bomb ที่อยู่ใน map
            - [False] = bomb ยังไม่ถูกวาง
            - [True] = bomb ถูกวางแล้ว
        - self.path = list ของตำแหน่งที่ player จะเดินไป
            - สมมุติ self.path คือ [[0,0],[1,0],[2,0]] = player จะเดินไปที่ [2,0] โดยเริ่มจาก [0,0]
        - self.movement_path = list ของทิศทางที่ player จะเดินไป
            - สมมุติ self.movement_path คือ [1,1] = player จะเดินไปที่ [2,0] โดยเริ่มจาก [0,0] โดยเดินไปทางขวา 2 ครั้ง
        '''
        start = [int(self.pos_x/Player.TILE_SIZE ), int(self.pos_y/Player.TILE_SIZE )]
        self.path = [start]
        # [0,0,-1] คือวางระเบิด
        new_choice = [self.dire[0], self.dire[1], self.dire[2], self.dire[3],[0,0,-1]]
        random.shuffle(new_choice)
        current = start
        for i in range(3):
            for direction in new_choice:
                next_x = current[0] + direction[0]
                next_y = current[1] + direction[1]
                if direction[2] == -1 and self.set_bomb < self.bomb_limit:
                    
                    for i in range(len(self.plant)):
                        if not self.plant[i]:
                            self.plant[i] = True
                            break
                if 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and grid[next_x][next_y] not in [2,3]:
                    self.path.append([next_x, next_y])
                    self.movement_path.append(direction[2])
                    current = [next_x, next_y]
                    break
        # สิ่งที่ต้องการคือ จะต้องสร้าง path ที่ bomberman จะเดินได้ นั้นคือ self.path และ self.movement_path เช่น
        # self.path = [[0,0],[1,0],[2,0]]
        # self.movement_path = [1,1]