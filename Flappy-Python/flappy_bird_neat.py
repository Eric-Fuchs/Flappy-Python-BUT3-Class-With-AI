
# Flappy Bird — NEAT v2 (entrainement infini, réseau costaud, rendu vectoriel)
# Auteur: Tom
# Dépendances: pygame, neat-python
#   pip install pygame neat-python
#
# Lancement:
#   python flappy_bird_neat_v2.py
#
# Menu:
#   [1] Entraîner l'IA (infini, ESC pour arrêter proprement la génération en cours)
#   [2] Jouer manuellement (Espace = saut, ESC = menu)
#   [3] Voir le champion (lit winner.pkl)
#   [S] Changer de skin
#   [ESC] Quitter

import os
import sys
import random
import pickle
from typing import List

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)  # faible latence
except Exception:
    print("Pygame manquant. Installe-le avec: pip install pygame")
    raise

try:
    import neat
    from neat.reporting import BaseReporter
except Exception:
    print("Le paquet 'neat-python' est manquant. Installe-le avec: pip install neat-python")
    raise

# ============== Fenêtre & globals ==============
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 500, 800
FLOOR_Y = 730
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird — NEAT v2")

STAT_FONT = pygame.font.SysFont("consolas", 26)
BIG_FONT = pygame.font.SysFont("consolas", 42, bold=True)

# Skins: (bg, pipe, rim, bird, ui)
SKINS = [
    ((30,30,45),  (90,200,90),  (50,130,60),  (240,220,70),  (235,235,235)),  # classic
    ((14,14,24),  (60,130,220), (36,80,150),  (250,250,255), (230,240,255)),  # nocturne
    ((20,12,40),  (255,0,128),  (80,0,80),    (0,255,200),   (255,200,255)),  # synthwave
    ((240,220,180),(200,140,60),(130,80,30),  (80,40,0),     (20,20,20)),     # desert
    ((18,18,18),  (200,200,200),(120,120,120),(240,240,240), (250,250,250)),  # mono
]
SKIN_IDX = 0

#============= Texutes ==============
BG_IMG = pygame.transform.scale(pygame.image.load("assets/bg.png").convert(), (WIDTH, HEIGHT))
BIRD_IMG = pygame.image.load("assets/bird1.png").convert_alpha()
MENU_BG_IMG = pygame.transform.scale(
    pygame.image.load(os.path.join("assets", "bg1.png")).convert(),
    (WIDTH, HEIGHT))
# Physique / gameplay
GRAVITY       = 0.5
JUMP_VELOCITY = -8.5
PIPE_GAP      = 200
PIPE_DISTANCE = 300
PIPE_VEL      = 5

# Entraînement / affichage
gen = 0
STOP_TRAINING = False

# ============== Utils ==============
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def make_net(genome, config):
    # Choisit feedforward ou recurrent selon config
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        return neat.nn.RecurrentNetwork.create(genome, config)

class UISyncReporter(BaseReporter):
    """Synchronise le compteur de génération 'gen' affiché à l'écran avec celui de NEAT."""
    def start_generation(self, generation):
        global gen
        gen = generation
SND_FLAP  = pygame.mixer.Sound("sounds/se_go_ball_gotcha.wav")
SND_POINT = pygame.mixer.Sound("sounds/SE164_001.wav")
SND_HIT   = pygame.mixer.Sound("sounds/gameover.mp3")
SND_DIE   = pygame.mixer.Sound("sounds/gameover.mp3")
for s in (SND_FLAP, SND_POINT, SND_HIT, SND_DIE):
    s.set_volume(0.35)

# ============== Entités ==============
class Bird:
    def __init__(self, x: int, y: int, color=(240,220,70)):
        self.x = float(x)
        self.y = float(y)
        self.vel = 0.0
        self.radius = 16
        self.sprite = pygame.transform.smoothscale(BIRD_IMG, (self.radius*2, self.radius*2))
        self.color = color
        self.tilt = 0.0

    def jump(self):
        self.vel = JUMP_VELOCITY
        try: SND_FLAP.play()
        except: pass

    def move(self):
        self.vel += GRAVITY
        dy = self.vel
        if dy > 16: dy = 16
        self.y += dy
        self.tilt = 25 if dy < 0 else max(-90, self.tilt - 3.5)

    def draw(self, win):
        rot = pygame.transform.rotate(self.sprite, self.tilt)
        rect = rot.get_rect(center=(int(self.x), int(self.y)))
        win.blit(rot, rect.topleft)

  




class Pipe:
    def __init__(self, x: int, gap_center: int, gap_size: int, color=(90,200,90), rim=(50,130,60)):
        self.x = float(x)
        self.gap_center = int(gap_center)
        self.gap_size = int(gap_size)
        self.width = 80
        self.passed = False
        self.color = color
        self.rim = rim

    @property
    def top(self): return self.gap_center - self.gap_size//2
    @property
    def bottom(self): return self.gap_center + self.gap_size//2

    def move(self):
        self.x -= PIPE_VEL

    def draw(self, win):
        top = self.top; bottom = self.bottom
        pygame.draw.rect(win, self.color, pygame.Rect(int(self.x), 0, self.width, max(0, top)))
        pygame.draw.rect(win, self.color, pygame.Rect(int(self.x), bottom, self.width, max(0, HEIGHT-bottom)))
        rim_th = 10
        pygame.draw.rect(win, self.rim, pygame.Rect(int(self.x)-6, top - rim_th, self.width+12, rim_th))
        pygame.draw.rect(win, self.rim, pygame.Rect(int(self.x)-6, bottom, self.width+12, rim_th))

class Base:
    def __init__(self, y: int):
        self.y = y
        self.x1 = 0
        self.x2 = WIDTH
    def move(self):
        self.x1 -= PIPE_VEL; self.x2 -= PIPE_VEL
        if self.x1 + WIDTH < 0: self.x1 = self.x2 + WIDTH
        if self.x2 + WIDTH < 0: self.x2 = self.x1 + WIDTH
    def draw(self, win, color=(150,120,80)):
        pygame.draw.rect(win, color, pygame.Rect(int(self.x1), self.y, WIDTH, HEIGHT-self.y))
        pygame.draw.rect(win, color, pygame.Rect(int(self.x2), self.y, WIDTH, HEIGHT-self.y))

# ============== Rendu ==============
def draw_window(win, birds: List[Bird], pipes: List[Pipe], base: Base, score: int, gen_num: int, alive: int, skin_idx: int):
    bg, pipe_c, rim_c, bird_c, ui = SKINS[skin_idx]
    win.blit(BG_IMG, (0, 0))
    for p in pipes: p.draw(win)
    base.draw(win)
    score_surf = STAT_FONT.render(f"Score: {score}", True, ui)
    gen_surf   = STAT_FONT.render(f"Gen: {gen_num}", True, ui)
    alive_surf = STAT_FONT.render(f"Alive: {alive}", True, ui)
    win.blit(score_surf, (WIDTH - score_surf.get_width() - 10, 10))
    win.blit(gen_surf, (10, 10))
    win.blit(alive_surf, (10, 40))
    for b in birds: b.draw(win)
    pygame.display.update()

def draw_menu(win, skin_idx: int):
    bg, _, _, _, ui = SKINS[skin_idx]
    win.blit(MENU_BG_IMG, (0, 0))
    title = BIG_FONT.render("FLAPPY BIRD — NEAT v2", True, ui)
    win.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2 - 160)))
    m1 = STAT_FONT.render("[1] Entraîner l'IA (infini)", True, ui)
    m2 = STAT_FONT.render("[2] Jouer manuellement", True, ui)
    m3 = STAT_FONT.render("[3] Voir le champion", True, ui)
    m4 = STAT_FONT.render("[S] Changer de skin   [ESC] Quitter", True, ui)
    for i, s in enumerate([m1, m2, m3, m4]):
        win.blit(s, s.get_rect(center=(WIDTH//2, HEIGHT//2 - 40 + i*46)))
    pygame.display.update()

def draw_banner(win, text: str, skin_idx: int):
    _, _, _, _, ui = SKINS[skin_idx]
    banner = BIG_FONT.render(text, True, ui)
    win.blit(banner, banner.get_rect(center=(WIDTH//2, 90)))
    pygame.display.update()

# ============== Génération tuyaux ==============
def new_pipe_x(prev_x): return prev_x + PIPE_DISTANCE
def random_gap_center():
    margin = 100
    return random.randint(margin, FLOOR_Y - margin)

# ============== Évaluation (NEAT) ==============
def eval_genomes(genomes, config):
    global SKIN_IDX, STOP_TRAINING, PIPE_GAP, PIPE_VEL

    nets, ge, birds = [], [], []
    bg, pipe_c, rim_c, bird_c, ui = SKINS[SKIN_IDX]

    for _, g in genomes:
        net = make_net(g, config)
        nets.append(net)
        birds.append(Bird(120, HEIGHT//2, color=bird_c))
        g.fitness = 0.0
        ge.append(g)

    base = Base(FLOOR_Y)
    pipes = [Pipe(350, random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c)]
    score = 0

    clock = pygame.time.Clock()
    run = True
    while run and len(birds) > 0 and not STOP_TRAINING:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                STOP_TRAINING = True  # stop après cette génération

        if len(birds) == 0:
            break

        # Choix du tuyau cible
        pipe_ind = 0
        if len(pipes) > 1 and all(b.x > pipes[0].x + pipes[0].width for b in birds):
            pipe_ind = 1
        target = pipes[pipe_ind]

        # Déplacements + décisions
        dead_idx = set()
        for i in range(len(birds)):
            b = birds[i]
            b.move()
            ge[i].fitness += 0.05  # survivre

            # Features (10)
            dx  = (target.x + target.width/2) - b.x
            dy  = (target.gap_center - b.y)
            ttc = dx / max(1.0, PIPE_VEL) / 60.0  # ~secondes
            if len(pipes) > pipe_ind + 1:
                nxt = pipes[pipe_ind + 1]
                dx2 = (nxt.x + nxt.width/2) - b.x
                dy2 = (nxt.gap_center - b.y)
            else:
                dx2 = WIDTH; dy2 = 0.0

            inputs = (
                b.y / HEIGHT,                                # 1
                max(-15.0, min(15.0, b.vel)) / 15.0,         # 2
                clamp01(dx / WIDTH),                          # 3
                dy / HEIGHT,                                  # 4
                clamp01(ttc),                                  # 5
                target.gap_size / HEIGHT,                     # 6
                target.top / HEIGHT,                          # 7
                target.bottom / HEIGHT,                       # 8
                clamp01(dx2 / WIDTH),                         # 9
                dy2 / HEIGHT,                                 # 10
            )

            out = nets[i].activate(inputs)
            if out[0] > 0.5:
                b.jump()
                ge[i].fitness -= 0.002  # éviter le spam de flaps

            # bonus de centrage vers le gap
            err = abs(dy) / max(1.0, (target.gap_size / 2.0))
            ge[i].fitness += max(0.0, 0.05 * (1.0 - min(1.0, err)))

        # Tuyaux / collisions (collecte, puis suppression en fin de frame)
        add_pipe = False
        rem_pipes = []

        for p in pipes:
            for i, b in enumerate(birds):
                if p.x < b.x + b.radius < p.x + p.width:
                    if b.y - b.radius < p.top or b.y + b.radius > p.bottom:
                        ge[i].fitness -= 1.0
                        dead_idx.add(i)
                if b.y + b.radius >= FLOOR_Y or b.y - b.radius <= 0:    
                    if i not in dead_idx:  # évite de jouer 2x le même frame
                        try:
                            SND_HIT.play()
                            SND_DIE.play()
                        except:
                            pass
                    ge[i].fitness -= 1.0
                    dead_idx.add(i)

            if not p.passed and any(b.x > p.x + p.width for b in birds):
                p.passed = True
                add_pipe = True

            p.move()
            if p.x + p.width < 0:
                rem_pipes.append(p)

        if add_pipe:    
            score += 1
            try: SND_POINT.play()
            except: pass
            for g in ge: g.fitness += 5.0
            # curriculum
            if score % 5 == 0:
                PIPE_GAP = max(140, PIPE_GAP - 10)
            if score % 10 == 0:
                PIPE_VEL = min(9, PIPE_VEL + 0.2)
            last_x = max(pipes[-1].x, WIDTH + 50)
            pipes.append(Pipe(new_pipe_x(last_x), random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c))

        for r in rem_pipes:
            if r in pipes: pipes.remove(r)

        if dead_idx:
            for idx in sorted(dead_idx, reverse=True):
                birds.pop(idx); nets.pop(idx); ge.pop(idx)

        base.move()
        draw_window(WIN, birds, pipes, base, score, gen, len(birds), SKIN_IDX)

# ============== Entraînement infini + checkpoints ==============
def train_ai(config_path, skin_idx=0, save_winner=True, ckpt_every=10):
    global SKIN_IDX, STOP_TRAINING
    SKIN_IDX = skin_idx
    STOP_TRAINING = False

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-XXXX')  # pour reprendre
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(UISyncReporter())
    p.add_reporter(neat.Checkpointer(generation_interval=ckpt_every, filename_prefix="neat-checkpoint-"))

    best_ever = None
    best_fit = float("-inf")

    while not STOP_TRAINING:
        p.run(eval_genomes, 1)
        if STOP_TRAINING:
            break

        pop_vals = [g for g in p.population.values() if g.fitness is not None]
        if pop_vals:
            current_best = max(pop_vals, key=lambda g: g.fitness)
            if current_best.fitness > best_fit:
                best_fit = current_best.fitness
                best_ever = current_best
                if save_winner:
                    with open("winner.pkl", "wb") as f:
                        pickle.dump(best_ever, f)

# ============== Champion (replay) ==============
def watch_champion(config_path, winner_path="winner.pkl", skin_idx=0):
    global SKIN_IDX
    SKIN_IDX = skin_idx
    if not os.path.exists(winner_path):
        return

    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    net = make_net(winner, config)

    bg, pipe_c, rim_c, bird_c, ui = SKINS[SKIN_IDX]
    base = Base(FLOOR_Y)
    bird = Bird(120, HEIGHT//2, color=bird_c)
    pipes = [Pipe(350, random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c)]
    score = 0

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].width:
            pipe_ind = 1
        target = pipes[pipe_ind]

        bird.move()
        dx  = (target.x + target.width/2) - bird.x
        dy  = (target.gap_center - bird.y)
        ttc = dx / max(1.0, PIPE_VEL) / 60.0
        if len(pipes) > pipe_ind + 1:
            nxt = pipes[pipe_ind + 1]
            dx2 = (nxt.x + nxt.width/2) - bird.x
            dy2 = (nxt.gap_center - bird.y)
        else:
            dx2 = WIDTH; dy2 = 0.0

        inputs = (
            bird.y / HEIGHT,
            max(-15.0, min(15.0, bird.vel)) / 15.0,
            clamp01(dx / WIDTH),
            dy / HEIGHT,
            clamp01(ttc),
            target.gap_size / HEIGHT,
            target.top / HEIGHT,
            target.bottom / HEIGHT,
            clamp01(dx2 / WIDTH),
            dy2 / HEIGHT,
        )
        if net.activate(inputs)[0] > 0.5:
            bird.jump()

        add_pipe = False
        rem = []
        for p in pipes:
            if p.x < bird.x + bird.radius < p.x + p.width:
                if bird.y - bird.radius < p.top or bird.y + bird.radius > p.bottom:
                    run = False
            if not p.passed and bird.x > p.x + p.width:
                p.passed = True; add_pipe = True
            p.move()
            if p.x + p.width < 0: rem.append(p)
        if add_pipe:
            score += 1
            last_x = max(pipes[-1].x, WIDTH + 50)
            pipes.append(Pipe(new_pipe_x(last_x), random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c))
        for r in rem:
            if r in pipes: pipes.remove(r)

        base.move()
        draw_window(WIN, [bird], pipes, base, score, gen, 1, SKIN_IDX)

# ============== Mode manuel ==============
def play_manual(skin_idx=0):
    global SKIN_IDX
    SKIN_IDX = skin_idx
    bg, pipe_c, rim_c, bird_c, ui = SKINS[SKIN_IDX]

    base = Base(FLOOR_Y)
    bird = Bird(120, HEIGHT//2, color=bird_c)
    pipes = [Pipe(350, random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c)]
    score = 0
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: bird.jump()
                if event.key == pygame.K_ESCAPE: run = False

        bird.move()
        add_pipe = False
        rem = []
        for p in pipes:
            if p.x < bird.x + bird.radius < p.x + p.width:
                if bird.y - bird.radius < p.top or bird.y + bird.radius > p.bottom:
                    run = False
            if not p.passed and bird.x > p.x + p.width:
                p.passed = True; add_pipe = True
            p.move()
            if p.x + p.width < 0: rem.append(p)

        if add_pipe:
            score += 1
            last_x = max(pipes[-1].x, WIDTH + 50)
            pipes.append(Pipe(new_pipe_x(last_x), random_gap_center(), PIPE_GAP, color=pipe_c, rim=rim_c))
        for r in rem:
            if r in pipes: pipes.remove(r)

        base.move()
        draw_window(WIN, [bird], pipes, base, score, gen, 1, SKIN_IDX)

# ============== Config NEAT (toujours écrite) ==============
DEFAULT_NEAT_CONFIG = """[NEAT]
fitness_criterion          = max
fitness_threshold          = 50
pop_size                   = 150
reset_on_extinction        = False

[DefaultGenome]
activation_default         = tanh
activation_options         = tanh sigmoid relu gauss
activation_mutate_rate     = 0.03

aggregation_default        = sum
aggregation_options        = sum max
aggregation_mutate_rate    = 0.0

# 10 entrées, 1 sortie
num_inputs                 = 10
num_outputs                = 1
num_hidden                 = 0

# True = feed-forward, False = récurrent autorisé
feed_forward               = True
initial_connection         = full_direct

# bias
bias_init_mean             = 0.0
bias_init_stdev            = 1.0
bias_max_value             = 30.0
bias_min_value             = -30.0
bias_mutate_power          = 0.6
bias_mutate_rate           = 0.7
bias_replace_rate          = 0.1

# response (compat neat-python variantes)
response_init_mean         = 0.0
response_init_stdev        = 1.0
response_max_value         = 30.0
response_min_value         = -30.0
response_mutate_power      = 0.5
response_mutate_rate       = 0.7
response_replace_rate      = 0.1

# weights
weight_init_mean           = 0.0
weight_init_stdev          = 1.0
weight_max_value           = 30.0
weight_min_value           = -30.0
weight_mutate_power        = 0.8
weight_mutate_rate         = 0.85
weight_replace_rate        = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

node_add_prob              = 0.3
node_delete_prob           = 0.2
conn_add_prob              = 0.6
conn_delete_prob           = 0.3
enabled_default            = True
enabled_mutate_rate        = 0.01

[DefaultSpeciesSet]
compatibility_threshold    = 3.0

[DefaultStagnation]
species_fitness_func       = max
max_stagnation             = 20
species_elitism            = 2

[DefaultReproduction]
elitism                    = 2
survival_threshold         = 0.25
"""

def write_neat_config_always(path: str):
    # On écrase toujours pour éviter des configs divergentes
    with open(path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_NEAT_CONFIG)

# ============== Main menu loop ==============
def main():
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    write_neat_config_always(config_path)  # on force une config propre à chaque run

    clock = pygame.time.Clock()
    skin = 0
    while True:
        clock.tick(60)
        draw_menu(WIN, skin)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_1:
                    draw_banner(WIN, "Entraînement en cours... (ESC pour stopper)", skin)
                    train_ai(config_path, skin_idx=skin, save_winner=True, ckpt_every=10)
                    draw_banner(WIN, "Entraînement stoppé. [3] pour voir le champion.", skin)
                if event.key == pygame.K_2:
                    play_manual(skin_idx=skin)
                if event.key == pygame.K_3:
                    if os.path.exists("winner.pkl"):
                        watch_champion(config_path, "winner.pkl", skin_idx=skin)
                    else:
                        draw_banner(WIN, "Pas de champion (entraîne d'abord).", skin)
                if event.key == pygame.K_s:
                    skin = (skin + 1) % len(SKINS)

if __name__ == "__main__":
    main()
