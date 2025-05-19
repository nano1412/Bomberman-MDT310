import random
from player import Player
import numpy as np
from collections import deque
import math
import heapq

import game
from enemy import Enemy
from enums.algorithm import Algorithm

class YourPlayer(Player):
    def __init__(self, player_id, x, y, alg):
        super().__init__(player_id, x, y, alg)
        self.last_positions = deque(maxlen=5)  # เก็บตำแหน่งล่าสุดเพื่อป้องกันการเดินวน
        self.strategy_mode = "aggressive"  # หรือ "defensive"
        self.bomb_cooldown = 0
        self.danger_map = None
        self.path_history = []
        self.me = None

        self.enemys = []
        self.players = []
        self.theTarget = None
        self.theTargetPosition = None
        self.theTargetSpawnPoint = None
        self.isStop = False

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
        rawTargets = []
        targets = []
        current_pos = (int(self.pos_x/Player.TILE_SIZE), int(self.pos_y/Player.TILE_SIZE))
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                # if grid[i][j] == 4:  # ศัตรู
                #     rawTargets.append((i, j, 10, "enemy"))  # คะแนนความสำคัญสูง
                # elif grid[i][j] == 5:  # ผู้เล่นอื่น
                #     rawTargets.append((i, j, 8, "player"))
                if grid[i][j] == 2:  # กล่องที่ทำลายได้
                    # ให้ความสำคัญกับกล่องที่อยู่ใกล้ศัตรูหรือผู้เล่น
                    box_value = 3
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                            if grid[ni][nj] in [4, 5]:
                                box_value = 6  # เพิ่มความสำคัญ
                                break
                    rawTargets.append((i, j, box_value, "box"))

        # ให้ตำแหน่งเดินจริงเป็น +- 2 จากเป้าหมาย
        for rawTarget in rawTargets:
            for actualDistinationi,actualDistinationj in [(2,0),(-2,0),(0,2),(0,-2)]:
                actualDistinationi+=rawTarget[0]
                actualDistinationj+=rawTarget[1]
                if 0 <= actualDistinationi < len(grid) and 0 <= actualDistinationj < len(grid[0]):
                    targets.append((actualDistinationi,actualDistinationj,rawTarget[2],rawTarget[3]))
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

    def should_plant_bomb(self, grid, current_pos):
        """ตัดสินใจว่าจะวางระเบิดหรือไม่"""
        if self.bomb_limit <= 0 or self.bomb_cooldown > 0:
            return False
            
        
        # ตรวจสอบว่าวางระเบิดแล้วจะทำลายอะไรได้บ้าง
        # potential_targets = 0
        # for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
        #     for i in range(1, self.range + 1):
        #         nx, ny = current_pos[0] + direction[0]*i, current_pos[1] + direction[1]*i
        #         if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
        #             if grid[nx][ny] == 2:  # กล่อง
        #                 potential_targets += 1
        #                 break
        #             elif grid[nx][ny] in [4, 5]:  # ศัตรูหรือผู้เล่น
        #                 potential_targets += 2
        #                 break
        #             elif grid[nx][ny] in [1, 3]:  # กำแพงหรือสิ่งกีดขวาง
        #                 break
        
        # ตรวจสอบว่ามีทางหนีหลังจากวางระเบิดหรือไม่
        escape_path = False
        for direction in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = current_pos[0] + direction[0], current_pos[1] + direction[1]
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                if grid[nx][ny] == 0 and self.danger_map[nx][ny] < 2:
                    escape_path = True
                    break
        
        return  escape_path

    def find_safe_spot(self, grid, current_pos, bombs):
        """หาตำแหน่งที่ปลอดภัยที่สุดเมื่ออยู่ในอันตราย"""
        safe_spots = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0 and self.danger_map[i][j] < 2:  # เซลล์ที่เดินได้และปลอดภัย
                    # ตรวจสอบว่าไม่มีการระเบิดหรือระเบิดที่กำลังจะระเบิด
                    is_safe = True
                    for bomb in bombs:
                        if (bomb.pos_x == i and bomb.pos_y == j) or \
                           (abs(bomb.pos_x - i) <= bomb.range and bomb.pos_y == j) or \
                           (abs(bomb.pos_y - j) <= bomb.range and bomb.pos_x == i):
                            is_safe = False
                            break
                    if is_safe:
                        safe_spots.append((i, j))
        
        if safe_spots:
            # หาตำแหน่งที่ปลอดภัยและใกล้ที่สุด
            safe_spots.sort(key=lambda pos: self.manhattan_distance(current_pos, pos))
            return safe_spots[0]
        return None

    def avoid_cycles(self, path):
        """ป้องกันการเดินวนเป็นวงกลม"""
        if len(self.last_positions) == self.last_positions.maxlen:
            # ตรวจสอบว่าตำแหน่งซ้ำเกิน 3 ครั้งหรือไม่
            if len(set(self.last_positions)) <= 2:
                return True
        return False
    
    def GetMe(self):
        for pl in game.player_list:
            if pl.algorithm == Algorithm.YourAlgorithm:
                self.me = pl

    def GetTheMANHATTANGuys(self):
        for en in game.enemy_list:
            if en.algorithm == Algorithm.MANHATTAN:
                self.theTarget = en

    def LureEnemy(self,grid,current_pos):
        
        path = self.a_star_search(grid, current_pos, (int(self.theTarget.pos_x/Enemy.TILE_SIZE),int(self.theTarget.pos_y/Enemy.TILE_SIZE)), [])
        print("MANHATTAN position")
        print((self.theTarget.pos_x,self.theTarget.pos_y))
        print(path)
        if path and len(path) > 0:
            self.path = [current_pos] + path[:3]
            self.movement_path = []
            for i in range(1, len(self.path)):
                dx = self.path[i][0] - self.path[i-1][0]
                dy = self.path[i][1] - self.path[i-1][1]
                direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                self.movement_path.append(direction)
        self.GoToSpawn(grid,current_pos)
        
    def GoToSpawn(self,grid,current_pos):
        print("go to spawn")
        path = self.a_star_search(grid, current_pos, (self.theTarget.start_x,self.theTarget.start_y), [])
        print("MANHATTAN position")
        print((self.theTarget.pos_x,self.theTarget.pos_y))
        print(path)
        if path and len(path) > 0:
            self.path = [current_pos] + path[:3]
            self.movement_path = []
            for i in range(1, len(self.path)):
                dx = self.path[i][0] - self.path[i-1][0]
                dy = self.path[i][1] - self.path[i-1][1]
                direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                self.movement_path.append(direction)

                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = self.theTarget.start_x + di, self.theTarget.start_y  + dj
                        if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]):
                            if current_pos == (ni,nj):
                                for i in range(len(self.plant)):
                                    if not self.plant[i]:
                                        self.plant[i] = True
                                        self.bomb_cooldown = 5
                                break
                if current_pos == (self.theTarget.start_x,self.theTarget.start_y):
                    self.isStop = True
            return

    def your_algorithm(self, grid):
        print("run")
        current_pos = (int(self.pos_x/Player.TILE_SIZE), int(self.pos_y/Player.TILE_SIZE))
        if self.theTarget == None:
            self.GetTheMANHATTANGuys()
            self.GetMe()
            print(self.theTarget)
        # อัพเดตแผนที่อันตราย
        self.update_danger_map(grid, [], [])  # หมายเหตุ: ต้องส่ง bombs และ explosions มาจากภายนอก
        
        if self.me.life == False:
            self.isStop = False

        if not self.isStop:
            self.LureEnemy(grid,current_pos)
            
        return
        
        # ตรวจสอบสถานะความปลอดภัย
        in_danger = self.danger_map[current_pos[0]][current_pos[1]] > 2
        
        
        # กำหนดกลยุทธ์
        if in_danger:
            self.strategy_mode = "defensive"
        elif targets and targets[0][2] >= 8:  # มีศัตรูหรือผู้เล่นใกล้เคียง
            self.strategy_mode = "aggressive"
        else:
            self.strategy_mode = "explore"
        
        # ลดคูลดาวน์ระเบิด
        if self.bomb_cooldown > 0:
            self.bomb_cooldown -= 1
        
        # กลยุทธ์ป้องกันตัว
        if in_danger:
            safe_spot = self.find_safe_spot(grid, current_pos, [])  # หมายเหตุ: ต้องส่ง bombs มาจากภายนอก
            if safe_spot:
                path = self.a_star_search(grid, current_pos, safe_spot, [])
                if path and len(path) > 0:
                    self.path = [current_pos] + path[:3]
                    self.movement_path = []
                    for i in range(1, len(self.path)):
                        dx = self.path[i][0] - self.path[i-1][0]
                        dy = self.path[i][1] - self.path[i-1][1]
                        direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                        self.movement_path.append(direction)
                    return
        
        # กลยุทธ์รุก
        if self.strategy_mode == "aggressive" and targets:
            best_target = None
            for target in targets:
                if target[3] in ["enemy", "player"] or (target[3] == "box" and target[2] >= 5):
                    path = self.a_star_search(grid, current_pos, (target[0], target[1]), [])
                    print("player pos")
                    print(current_pos)
                    print(target[0])
                    print(int(self.theTarget.pos_x/Enemy.TILE_SIZE) )
                    print(target[1])
                    print(int(self.theTarget.pos_y/Enemy.TILE_SIZE))
                    if path:
                        best_target = target
                        break
            
            if best_target:
                # ตรวจสอบว่าควรวางระเบิดหรือไม่
                if self.should_plant_bomb(grid, current_pos):
                    for i in range(len(self.plant)):
                        if not self.plant[i]:
                            self.plant[i] = True
                            self.bomb_cooldown = 5
                            break
                
                # เดินทางไปยังเป้าหมาย
                if path and len(path) > 0:
                    self.path = [current_pos] + path[:3]
                    self.movement_path = []
                    for i in range(1, len(self.path)):
                        dx = self.path[i][0] - self.path[i-1][0]
                        dy = self.path[i][1] - self.path[i-1][1]
                        direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                        self.movement_path.append(direction)
                    return
        
        # กลยุทธ์สำรวจ
        if self.strategy_mode == "explore" or not targets:
            # หาเส้นทางไปยังพื้นที่ที่ยังไม่สำรวจ
            unexplored = []
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 0 and (i,j) not in self.path_history:
                        unexplored.append((i,j))
            
            if unexplored:
                unexplored.sort(key=lambda pos: self.manhattan_distance(current_pos, pos))
                target = unexplored[0]
                path = self.a_star_search(grid, current_pos, target, [])
                if path and len(path) > 0:
                    self.path = [current_pos] + path[:3]
                    self.movement_path = []
                    for i in range(1, len(self.path)):
                        dx = self.path[i][0] - self.path[i-1][0]
                        dy = self.path[i][1] - self.path[i-1][1]
                        direction = {(0,1): 0, (1,0): 1, (0,-1): 2, (-1,0): 3}[(dx, dy)]
                        self.movement_path.append(direction)
                    self.path_history.extend(path[:3])
                    return
        
        # ถ้าไม่มีกลยุทธ์อื่น ให้เดินแบบสุ่มแต่ปลอดภัย
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
        else:
            # ไม่มีทางเดินที่ปลอดภัย ให้อยู่กับที่
            self.path = [current_pos]
            self.movement_path = []