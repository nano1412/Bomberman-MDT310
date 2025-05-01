import pygame
import sys
import random

from player_keyboard import PlayerKeyboard
from player import Player
from explosion import Explosion
from enemy import Enemy
from enums.algorithm import Algorithm
from submission import YourPlayer


BACKGROUND_COLOR = (161, 196, 88)

font = None

player_1 = None
# เก็บ object ของ player 2 ตัวที่เป็น bot
player_list = []
# เก็บตำแหน่งของ player 2 ตัว
player_blocks = []
# เก็บ object ของ ghost 2 ตัว
enemy_list = []
# เก็บตำแหน่งของ ghost 2 ตัว
ene_blocks = []
# ใช้เก็บข้อมูลของระเบิดทั้งหมดที่ถูกวางในเกม.
bombs = []
# ใช้เก็บข้อมูลของการระเบิดที่เกิดขึ้น.
explosions = []

# bomberman color
P1_COLOR = (0, 123, 250)
P2_COLOR = (167, 77, 210)

# GRID_BASE = Grid.GRID_SMALL.value
GRID_BASE = []

# สำหรับ countdown timer
start_time = 60  # 1 minutes
start_ticks = pygame.time.get_ticks()

def game_init(surface, path, player_alg1,player_alg2, en_alg, scale,grid,FPS = 15):
    # เริ่มต้นเกม และทำการโหลดภาพพื้นฐาน
    global font
    font = pygame.font.SysFont('Bebas', scale)

    global enemy_list
    global ene_blocks
    global player_1

    global player_list
    global player_blocks

    enemy_list = []
    ene_blocks = []
    player_list = []
    player_blocks = []
    global explosions
    global bombs
    
    bombs.clear()
    explosions.clear()
  

    global GRID_BASE
    GRID_BASE = grid
    w = len(GRID_BASE)
    h = len(GRID_BASE[0])

    en1_alg = en_alg[0]
    en2_alg = en_alg[1]

    if en1_alg is not Algorithm.NONE:
        # สร้าง ghost และกำหนดตำแหน่งเริ่มต้น
        en1 = Enemy(w-2, 1, en1_alg, 1)
        en1.load_animations('1', scale)
        enemy_list.append(en1)
        ene_blocks.append(en1)

    if en2_alg is not Algorithm.NONE:
        en2 = Enemy(1, h-2, en2_alg,2)
        en2.load_animations('2', scale)
        enemy_list.append(en2)
        ene_blocks.append(en2)

    # if en3_alg is not Algorithm.NONE:
    #     en3 = Enemy(w-2, 1, en3_alg,3)
    #     en3.load_animations('3', scale)
    #     enemy_list.append(en3)
    #     ene_blocks.append(en3)

    # กำหนดวิธีการเล่นของผู้เล่น และกำหนดตำแหน่งเริ่มต้นของผู้เล่น
    player_1 = None
    if player_alg1 is Algorithm.PLAYER:
        player_1 = PlayerKeyboard(1,1,1,player_alg1)
        player_1.load_animations(scale)
        player_blocks.append(player_1)
    elif player_alg1 is not Algorithm.NONE:
        player_bot = YourPlayer(1, 1, 1, player_alg1)
        player_bot.load_animations('1', scale)
        player_list.append(player_bot)
        player_blocks.append(player_bot)
    else:
        player_1.life = False
        
    if player_alg2 is not Algorithm.NONE:
        player_bot = YourPlayer(2,w-2, h-2, player_alg2)
        player_bot.load_animations('2', scale)
        player_list.append(player_bot)
        player_blocks.append(player_bot)
    else:
        player_1.life = False

    grass_img = pygame.image.load('images/terrain/grass.png')
    grass_img = pygame.transform.scale(grass_img, (scale, scale))

    block_img = pygame.image.load('images/terrain/block.png')
    block_img = pygame.transform.scale(block_img, (scale, scale))

    box_img = pygame.image.load('images/terrain/box.png')
    box_img = pygame.transform.scale(box_img, (scale, scale))
    bomb_images = []
    for i in range(6):
        if i < 3:
            bomb_img = pygame.image.load(f'images/bomb/bomb_p1_{i+1}.png')
        else:
            bomb_img = pygame.image.load(f'images/bomb/bomb_p2_{i-2}.png')
        bomb_img = pygame.transform.scale(bomb_img, (scale, scale))
        bomb_images.append(bomb_img)

    
    explosion1_img = pygame.image.load('images/explosion/1.png')
    explosion1_img = pygame.transform.scale(explosion1_img, (scale, scale))

    explosion2_img = pygame.image.load('images/explosion/2.png')
    explosion2_img = pygame.transform.scale(explosion2_img, (scale, scale))

    explosion3_img = pygame.image.load('images/explosion/3.png')
    explosion3_img = pygame.transform.scale(explosion3_img, (scale, scale))

    terrain_images = [grass_img, block_img, box_img, grass_img]
    explosion_images = [explosion1_img, explosion2_img, explosion3_img]


    main(surface, scale, path, terrain_images, bomb_images, explosion_images, FPS)


def draw(s, grid, tile_size, show_path, game_ended, terrain_images, bomb_images, explosion_images,seconds):
    s.fill(BACKGROUND_COLOR)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            s.blit(terrain_images[grid[i][j]], (i * tile_size, j * tile_size, tile_size, tile_size))

    
    for x in bombs:
        if x.bomber.player_id == 1:
            s.blit(bomb_images[x.frame], (x.pos_x * tile_size, x.pos_y * tile_size, tile_size, tile_size))
        else:
            s.blit(bomb_images[x.frame+3], (x.pos_x * tile_size, x.pos_y * tile_size, tile_size, tile_size))

    for y in explosions:
        for x in y.sectors:
            s.blit(explosion_images[y.frame], (x[0] * tile_size, x[1] * tile_size, tile_size, tile_size))
    if player_1 and player_1.life:
        s.blit(player_1.animation[player_1.direction][player_1.frame],
               (player_1.pos_x * (tile_size / 4), player_1.pos_y * (tile_size / 4), tile_size, tile_size))

    for en in enemy_list:
        if en.life:
            s.blit(en.animation[en.direction][en.frame],
                   (en.pos_x * (tile_size / 4), en.pos_y * (tile_size / 4), tile_size, tile_size))
            if show_path:
                if en.algorithm == Algorithm.DFS:
                    for sek in en.path:
                        pygame.draw.rect(s, (0, 255, 0, 240),
                                         [sek[0] * tile_size, sek[1] * tile_size, tile_size, tile_size], 1)
                else:
                    for sek in en.path:
                        pygame.draw.rect(s, (255, 0, 255, 240),
                                         [sek[0] * tile_size, sek[1] * tile_size, tile_size, tile_size], 1)
    for pl in player_list:
        if pl.life:
            s.blit(pl.animation[pl.direction][pl.frame],
                   (pl.pos_x * (tile_size / 4), pl.pos_y * (tile_size / 4), tile_size, tile_size))
            if show_path:
                if pl.algorithm == Algorithm.DFS:
                    for sek in pl.path:
                        pygame.draw.rect(s, (255, 0, 0, 240),
                                         [sek[0] * tile_size, sek[1] * tile_size, tile_size, tile_size], 1)
                else:
                    for sek in pl.path:
                        pygame.draw.rect(s, (255, 0, 255, 240),
                                         [sek[0] * tile_size, sek[1] * tile_size, tile_size, tile_size], 1)
    display_scores(s)
    display_debug_icon(s)
    update_time(s, seconds)
    # if game_ended:
    #     tf = font.render("Press ESC to go back to menu", False, (153, 153, 255))
    #     s.blit(tf, (10, 10))

    pygame.display.update()


def generate_map(grid):
    # สร้างกล่องเปล่าไว้ใน map
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[i]) - 1):
            if grid[i][j] != 0:
                continue
            #บางตำแหน่งจะไม่มีกล่องเปล่า เช่น มุมของ map เพราะจะให้ ผู้เล่นและ ghost มีทางหนี
            elif (i < 4 or i > len(grid) - 5) and (j < 4 or j > len(grid[i]) - 5):
                continue
            # 30% chance to place a box วางกล่องเปล่าไว้
            if random.randint(0, 9) < 3:
                grid[i][j] = 2

    return


def main(s, tile_size, show_path, terrain_images, bomb_images, explosion_images,FPS = 15):

    grid = [row[:] for row in GRID_BASE]
    generate_map(grid)
    
    clock = pygame.time.Clock()

    # global start_time
    
    running = True
    game_ended = False
    last_time = 0
    debuging = False
    while running:
        
        # สำหรับปรับ frame rate
        dt = clock.tick(FPS)
        if not debuging:
            # สำหรับ ghost และ player ที่ไม่ใช่ keyboard
            for en in enemy_list:
                if en.pos_x < 0 or en.pos_y < 0 or en.pos_x >= len(grid)* Enemy.TILE_SIZE or en.pos_y >= len(grid[0])* Enemy.TILE_SIZE:
                    running = False
                    print("enemy out of bounds")
                    print(en.pos_x, en.pos_y)
                    print(int(en.pos_x/Enemy.TILE_SIZE), int(en.pos_y/Enemy.TILE_SIZE))
                    print(en.movement_path)
                    print(en.path)
                en.make_move(grid, bombs, explosions,player_blocks ,ene_blocks)

            for pl in player_list:
                if pl.pos_x < 0 or pl.pos_y < 0:
                    running = False
                    print("Player out of bounds")
                    print(pl.pos_x, pl.pos_y)
                    print(int(pl.pos_x/Player.TILE_SIZE), int(pl.pos_y/Player.TILE_SIZE))
                    print(pl.movement_path)
                    print(pl.path)
                pl.make_move(grid, bombs, explosions, player_blocks,ene_blocks)

            # สำหรับ keyboard
            if player_1 and player_1.algorithm is Algorithm.PLAYER:
                keys = pygame.key.get_pressed()
                temp = player_1.direction
                movement = False
                if keys[pygame.K_DOWN]:
                    temp = 0
                    player_1.move(0, 1, grid, player_blocks)
                    movement = True
                elif keys[pygame.K_RIGHT]:
                    temp = 1
                    player_1.move(1, 0, grid, player_blocks)
                    movement = True
                elif keys[pygame.K_UP]:
                    temp = 2
                    player_1.move(0, -1, grid, player_blocks)
                    movement = True
                elif keys[pygame.K_LEFT]:
                    temp = 3
                    player_1.move(-1, 0, grid, player_blocks)
                    movement = True
                if temp != player_1.direction:
                    player_1.frame = 0
                    player_1.direction = temp
                if movement:
                    if player_1.frame == 2:
                        player_1.frame = 0
                    else:
                        player_1.frame += 1

            # Countdown timer
            seconds = start_time - (pygame.time.get_ticks() - start_ticks) // 1000
            if seconds < 0:
                running = False
                if player_1 and player_1.algorithm is Algorithm.PLAYER:
                    player_1.life = False
                    print("Time's up!")
                    print("Player 1 score: ", player_1.score)
                    print("Player 2 score: ", player_list[0].score)
                    break
                else:
                    print("Time's up!")
                    print("Player 1 score: ", player_list[0].score)
                    print("Player 2 score: ", player_list[1].score)
                    print(check_winner())
                    break
            elif seconds != last_time:
                last_time = seconds
                
            draw(s, grid, tile_size, show_path, game_ended, terrain_images, bomb_images, explosion_images,seconds)
            

            if not game_ended:
                game_ended = check_end_game()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                sys.exit(0)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                x, y = e.pos
                if 320 < x < 360 and 445 < y < 480:
                    debuging = not debuging
                    if debuging:
                        print("Debug mode enabled")
                    else:
                        print("Debug mode disabled")
            elif player_1 and e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    if player_1.set_bomb == player_1.bomb_limit or not player_1.life:
                        continue
                    temp_bomb = player_1.plant_bomb(grid)
                    bombs.append(temp_bomb)
                    grid[temp_bomb.pos_x][temp_bomb.pos_y] = 3
                    player_1.set_bomb += 1
                elif e.key == pygame.K_ESCAPE:
                    running = False

        update_bombs(grid, dt)

    explosions.clear()
    enemy_list.clear()
    ene_blocks.clear()
    player_list.clear()
    player_blocks.clear()



def update_bombs(grid, dt):
    for b in bombs:
        b.update(dt)
        if b.time < 1:
            b.bomber.set_bomb -= 1
            grid[b.pos_x][b.pos_y] = 0
            exp_temp = Explosion(b.pos_x, b.pos_y, b.range)
            exp_temp.explode(grid, bombs, b)
            exp_temp.clear_sectors(grid)
            explosions.append(exp_temp)
    if player_1 and player_1 not in player_list:
        player_1.check_death(explosions)
    for en in enemy_list:
        en.check_death(explosions)
    for pl in player_list:
        pl.check_death(explosions)
    for e in explosions:
        e.update(dt)
        if e.time < 1:
            explosions.remove(e)


def check_end_game():
    if player_1 and not player_1.life:
        return True

    for en in enemy_list:
        if en.life:
            return False
    
    for en in player_list:
        if en.life:
            return False

    return True

def check_winner():
    if player_list[0].score > player_list[1].score:
        return "Player 1 wins!"
    elif player_list[0].score < player_list[1].score:
        return "Player 2 wins!"
    else:
        if player_list[0].step > player_list[1].step:
            return "Player 1 wins!"
        elif player_list[0].step < player_list[1].step:
            return "Player 2 wins!"
        else:
            return "It's a draw!"

def update_time(screen,seconds):
    # Display timer
    timer_text = font.render(f"Time Left: {seconds}", True, (255, 255, 255))
    screen.blit(timer_text, (150, 450))

def display_scores(screen):
    y_offset = 450  # ตำแหน่ง Y สำหรับแสดงชื่อผู้เล่น
    x_offset = 10   # ตำแหน่ง X สำหรับแสดงชื่อผู้เล่น
    if player_1:
        text = font.render(f"P1 : {player_1.score}", False, P1_COLOR)
        screen.blit(text, (x_offset, y_offset))
    for pl in player_list:
        if pl.player_id == 1:
            text = font.render(f"P1 : {pl.score}", False, P1_COLOR)
            screen.blit(text, (x_offset, y_offset))
        else:
            y_offset += 30
            text = font.render(f"P2 : {pl.score}", False, P2_COLOR)
            screen.blit(text, (x_offset, y_offset))
    
def display_debug_icon(screen):
    y_offset = 445  # ตำแหน่ง Y 
    x_offset = 330   # ตำแหน่ง X 
    image_path = 'images/icon/'
    play_icon = pygame.image.load(image_path + 'play.png')
    # Display icons on screen
    screen.blit(play_icon, (x_offset, y_offset))
