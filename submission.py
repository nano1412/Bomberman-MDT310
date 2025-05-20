import random
from player import Player
import numpy as np
from collections import deque
import math
import heapq
import time

import game
from enemy import Enemy
from enums.algorithm import Algorithm

class YourPlayer(Player):
    def __init__(self, player_id, x, y, alg):
        super().__init__(player_id, x, y, alg)
        self.strategy_mode = "random" 
        self.danger_map = None
        self.me = None
        self.another = None

        self.theTarget = None
        self.theTargetPosition = None
        self.theTargetSpawnPoint = None

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def euclidean_distance(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def update_danger_map(self, grid, bombs, explosions):
        """สร้างแผนที่อันตรายจากระเบิดและการระเบิด"""
        self.danger_map = np.zeros((len(grid), len(grid[0])))
        
        # อันตรายจากระเบิด
        for bomb in bombs:
            x, y = bomb.pos_x, bomb.pos_y
            self.danger_map[x][y] = 4  # อันตรายสูงสุด
            
            # รัศมีระเบิด
            for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
                for i in range(1, bomb.range + 1):
                    nx, ny = x + direction[0]*i, y + direction[1]*i
                    if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                        if grid[nx][ny] in [1, 2]:  # กำแพงหรือกล่อง
                            break
                        self.danger_map[nx][ny] = max(self.danger_map[nx][ny], 4 - i)  # อันตรายลดลงตามระยะทาง

        # อันตรายจากการระเบิด
        for explosion in explosions:
            for (x, y) in explosion.sectors:
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                    self.danger_map[x][y] = 5  # อันตรายสูงสุด

    def find_targets(self, grid):
        """ค้นหาเป้าหมายที่สำคัญ (ศัตรู, ผู้เล่น, กล่อง)"""
        targets = []
        current_pos = (int(self.pos_x/Player.TILE_SIZE), int(self.pos_y/Player.TILE_SIZE))
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 4:  # ศัตรู
                    targets.append((i, j, 10, "enemy"))  # คะแนนความสำคัญสูง
                elif grid[i][j] == 5:  # ผู้เล่นอื่น
                    targets.append((i, j, 8, "player"))
                elif grid[i][j] == 2:  # กล่องที่ทำลายได้
                    # ให้ความสำคัญกับกล่องที่อยู่ใกล้ศัตรูหรือผู้เล่น
                    box_value = 20
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                            if grid[ni][nj] in [4, 5]:
                                box_value = 6  # เพิ่มความสำคัญ
                                break
                    targets.append((i, j, box_value, "box"))
        
        # เรียงลำดับตามคะแนนความสำคัญและระยะทาง
        targets.sort(key=lambda t: (-t[2], self.manhattan_distance(current_pos, (t[0], t[1]))))
        return targets

    def a_star_search(self, grid, start, goal, bombs):
        """A* algorithm สำหรับหาเส้นทางที่ดีที่สุด"""
        def heuristic(a, b):
            return self.manhattan_distance(a, b) + self.danger_map[a[0]][a[1]]
        
        neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []
        
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            close_set.add(current)
            for di, dj in neighbors:
                neighbor = current[0] + di, current[1] + dj
                
                if not (0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])):
                    continue
                
                if grid[neighbor[0]][neighbor[1]] in [2, 3]:  # กล่องหรือกำแพง
                    continue
                    
                # ตรวจสอบว่าตำแหน่งนี้ปลอดภัยจากระเบิดหรือไม่
                is_safe = True
                for bomb in bombs:
                    if (bomb.pos_x == neighbor[0] and bomb.pos_y == neighbor[1]) or \
                       (abs(bomb.pos_x - neighbor[0]) <= bomb.range and bomb.pos_y == neighbor[1]) or \
                       (abs(bomb.pos_y - neighbor[1]) <= bomb.range and bomb.pos_x == neighbor[0]):
                        is_safe = False
                        break
                
                if not is_safe:
                    continue
                
                tentative_g_score = gscore[current] + 1 + self.danger_map[neighbor[0]][neighbor[1]]
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                    
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None  # ไม่พบเส้นทาง
    
    def GetMe(self,current_pos):
        for pl in game.player_list:
            if pl.algorithm == Algorithm.YourAlgorithm and current_pos == (pl.start_x,pl.start_y):
                self.me = pl
                print(current_pos)
                print((self.me.start_x,self.me.start_y))
            else:
                self.another = pl

    def GetTheEnemyThatFollowPlayer(self):
        for en in game.enemy_list:
            if en.algorithm == Algorithm.MANHATTAN or en.algorithm == Algorithm.DFS:
                self.theTarget = en
        if self.theTarget is not None:
            self.theTargetSpawnPoint = (self.theTarget.start_x,self.theTarget.start_y)

    def LureEnemy(self,grid,current_pos):
        
        path = self.a_star_search(grid, current_pos, (int(self.theTarget.pos_x/Enemy.TILE_SIZE),int(self.theTarget.pos_y/Enemy.TILE_SIZE)), [])
        if path and len(path) > 0:
            self.path = [current_pos] + path[:3]
            self.movement_path = []
            for i in range(1, len(self.path)):
                dx = self.path[i][0] - self.path[i-1][0]
                dy = self.path[i][1] - self.path[i-1][1]
                direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                self.movement_path.append(direction)

                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = current_pos[0] + di, current_pos[1] + dj
                        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                            if grid[ni][nj] == 4:
                                self.strategy_mode = "toSpawn"
        else:
            self.strategy_mode = "clearBlock"
            print('no path found')
            
    def GoToSpawn(self,grid,current_pos):
        path = self.a_star_search(grid, current_pos, self.theTargetSpawnPoint, [])
        if path and len(path) > 0:
            self.path = [current_pos] + path[:3]
            self.movement_path = []
            for i in range(1, len(self.path)):
                dx = self.path[i][0] - self.path[i-1][0]
                dy = self.path[i][1] - self.path[i-1][1]
                direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                self.movement_path.append(direction)

        if self.me.life == False:
            self.strategy_mode = "lure"
                
            return
        
    def ClearBlock(self,grid,current_pos,targets):
        path = self.a_star_search(grid, current_pos, (int(self.theTarget.pos_x/Enemy.TILE_SIZE),int(self.theTarget.pos_y/Enemy.TILE_SIZE)), [])
        if path and len(path) > 0:
            self.strategy_mode = "lure"
            return
            
        closest_target_step = 999
        closest_target = None
        path = None
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    for pi,pj in [(0,1),(1,0),(0,-1),(-1,0)]:
                        ajecenti = i+pi
                        ajecentj = j+pj
                        tempPath = self.a_star_search(grid, current_pos, (ajecenti,ajecentj), [])
                        # print((ajecenti,ajecentj))
                        # print(tempPath)
                        if tempPath is not None and len(tempPath) < closest_target_step:
                            closest_target = (ajecenti,ajecentj)
                            path = tempPath

        if closest_target is not None:
            if path and len(path) > 0:
                self.path = [current_pos] + path[:3]
                self.movement_path = []
                for i in range(1, len(self.path)):
                    dx = self.path[i][0] - self.path[i-1][0]
                    dy = self.path[i][1] - self.path[i-1][1]
                    direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                    self.movement_path.append(direction)

                    for i in range(len(self.plant)):
                        if not self.plant[i]:
                            self.plant[i] = True
                            self.strategy_mode = "survive"
                    
        return
    def Random(self,grid):
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
        self.strategy_mode = "survive"
        return    
    def Survive(self,current_pos,grid):
        safe_directions = []
        for direction in self.dire:
            nx, ny = current_pos[0] + direction[0], current_pos[1] + direction[1]
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == 0 and self.danger_map[nx][ny] < 2:
                    safe_directions.append(direction)
    
        if safe_directions:
            direction = random.choice(safe_directions)
            self.path = [current_pos, [current_pos[0] + direction[0], current_pos[1] + direction[1]]]
            self.movement_path = [direction[2]]
        # else:
        #     # ไม่มีทางเดินที่ปลอดภัย ให้อยู่กับที่
        #     self.path = [current_pos]
        #     self.movement_path = []  
        return
    
    def your_algorithm(self, grid):
        # print(self.theTargetSpawnPoint)
        print(self.strategy_mode)
        current_pos = (int(self.pos_x/Player.TILE_SIZE), int(self.pos_y/Player.TILE_SIZE))
        if self.theTarget == None:
            self.GetTheEnemyThatFollowPlayer()
            self.GetMe(current_pos)
            print(self.theTarget)
            self.strategy_mode = "random"

        
        # หาเป้าหมาย
        targets = self.find_targets(grid)
        # อัพเดตแผนที่อันตราย
        self.update_danger_map(grid, [], [])  # หมายเหตุ: ต้องส่ง bombs และ explosions มาจากภายนอก
         # ตรวจสอบสถานะความปลอดภัย
        in_danger = self.danger_map[current_pos[0]][current_pos[1]] > 2




        if self.strategy_mode == "lure":
            # print('in L')
            self.LureEnemy(grid,current_pos)
        elif self.strategy_mode == "clearBlock":
            # print('in CB')
            self.ClearBlock(grid,current_pos,targets)
        elif self.strategy_mode == "random":
            # print('in R')
            self.Random(grid)
        elif self.strategy_mode == "toSpawn":
            # print('in toS')
            self.GoToSpawn(grid,current_pos)
        elif self.strategy_mode == "survive":
            # print('in survive')
            self.Survive(current_pos,grid)
            
        else:
            self.movement_path = [] 

        if self.strategy_mode == "lure" or self.strategy_mode == "toSpawn":
            for i in range(len(self.plant)):
                if not self.plant[i]:
                    self.plant[i] = True
                    
                    break 
        if (self.me.life == False or self.me.get_score() <= self.another.get_score()) and not self.strategy_mode == "toSpawn" and not self.strategy_mode == "clearBlock" and not self.strategy_mode == "random":
            self.strategy_mode = "lure"

        if self.me.get_score() > self.another.get_score():
            self.strategy_mode = "survive"
        # if it still none here, mean enemy is all Random (or use other algo)
        if self.theTarget == None:
            self.strategy_mode = "random"
        return
        
        