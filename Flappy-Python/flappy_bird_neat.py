# Flappy Bird — NEAT v2 (textures base+pipes, anti-stack 2–3, boutons, skins, graph matplotlib)
# Auteur: Tom
# Dépendances: pygame, neat-python, matplotlib
#   pip install pygame neat-python matplotlib

import os
import sys
import random
import pickle
import time
import csv
from typing import List
from multiprocessing import Process  # popup graphe sans bloquer le jeu

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

# Matplotlib (popup graphe)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    print("Matplotlib non dispo: pip install matplotlib (le jeu continue sans le graphe).")

# ============== Fenêtre & globals ==============
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 500, 800
FLOOR_Y = 730  # haut de la base
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird — NEAT v2")

STAT_FONT = pygame.font.SysFont("consolas", 22)
BIG_FONT  = pygame.font.SysFont("consolas", 42, bold=True)

# Skins: (bg, pipe, rim, bird, ui)  (couleurs UI ; les tuyaux sont texturés PNG)
SKINS = [
    ((30,30,45),  (90,200,90),  (50,130,60),  (240,220,70),  (235,235,235)),  # classic
    ((14,14,24),  (60,130,220), (36,80,150),  (250,250,255), (230,240,255)),  # nocturne
    ((20,12,40),  (255,0,128),  (80,0,80),    (0,255,200),   (255,200,255)),  # synthwave
    ((240,220,180),(200,140,60),(130,80,30),  (80,40,0),     (20,20,20)),     # desert
    ((18,18,18),  (200,200,200),(120,120,120),(240,240,240), (250,250,250)),  # mono
]
SKIN_IDX = 0

#============= Backgrounds multiples ==============
def _safe_load_bg(path):
    try:
        img = pygame.image.load(path).convert()
        return pygame.transform.scale(img, (WIDTH, HEIGHT))
    except Exception:
        return None

BG_FILES = [
    "assets/bg.png",
    "assets/bg1.png",
    "assets/bg2.png",
    "assets/bg_desert.png",
]
BG_IMGS = [img for p in BG_FILES if (img := _safe_load_bg(p)) is not None]
if not BG_IMGS:
    surf = pygame.Surface((WIDTH, HEIGHT)); surf.fill((30,30,45)); BG_IMGS = [surf]
BG_IDX = 0
BG_IMG = BG_IMGS[BG_IDX]
MENU_BG_IMG = BG_IMGS[BG_IDX]

def apply_background(idx):
    global BG_IDX, BG_IMG, MENU_BG_IMG
    BG_IDX = idx % len(BG_IMGS)
    BG_IMG = BG_IMGS[BG_IDX]
    MENU_BG_IMG = BG_IMGS[BG_IDX]

def cycle_background(step=1):
    apply_background(BG_IDX + step)

#============= Textures sprites ==============
# Oiseau
BIRD_IMG = pygame.image.load("assets/bird1.png").convert_alpha()

# Tuyaux (PNG) — on scale à une largeur cible pour cohérence avec le gameplay
PIPE_RAW = pygame.image.load("assets/pipe-green.png").convert_alpha()
PIPE_TARGET_W = 80  # largeur cohérente avec ton ancienne valeur
pipe_scale = PIPE_TARGET_W / PIPE_RAW.get_width()
PIPE_BOTTOM_IMG = pygame.transform.smoothscale(
    PIPE_RAW, (PIPE_TARGET_W, int(PIPE_RAW.get_height() * pipe_scale))
)
PIPE_TOP_IMG = pygame.transform.flip(PIPE_BOTTOM_IMG, False, True)

# Base (PNG) — on scale à la largeur de l'écran et à la hauteur exacte de la base
BASE_RAW = pygame.image.load("assets/base.png").convert_alpha()
BASE_H = max(1, HEIGHT - FLOOR_Y)
BASE_IMG = pygame.transform.smoothscale(BASE_RAW, (WIDTH, BASE_H))

# Sons
SND_FLAP  = pygame.mixer.Sound("sounds/se_go_ball_gotcha.wav")
SND_POINT = pygame.mixer.Sound("sounds/sfx_point.mp3")
SND_HIT   = pygame.mixer.Sound("sounds/are_you_serious.mp3")
SND_DIE   = pygame.mixer.Sound("sounds/are_you_serious.mp3")
SND_BG    = pygame.mixer.Sound("sounds/bg_sound2.mp3")
for s in (SND_FLAP, SND_POINT, SND_HIT, SND_DIE, SND_BG):
    s.set_volume(0.7)

def play_menu_music():
    SND_BG.set_volume(0.25)
    SND_BG.play(-1)

# ===== Gameplay (timer spacing + anti-stack) =====
FPS               = 60
GRAVITY           = 0.5
JUMP_VELOCITY     = -8.5
PIPE_VEL          = 5

TIME_BETWEEN_PIPES= 0.78  # cadence (~0.72–0.85)
SPAWN_OFFSET_X    = 70    # spawn juste hors-écran
MIN_PIPE_SPACING_PX = 220 # anti-stack: distance min entre colonnes successives (≈ Flappy)

# Base gap par défaut (sera varié)
PIPE_GAP          = 180

# Entraînement / affichage
gen = 0
STOP_TRAINING = False

# Sessions
SESS_LOG = []            # {"mode": "manual"|"auto", "score": int}
SESS_CSV = "sessions_summary.csv"
SESSION_AUTO_BEST_SCORE = 0

#============= UI: Boutons esthétiques ==============
def draw_pill(surface, rect, bg=(0,0,0,130), border=(255,255,255), border_w=1, radius=14, shadow=True):
    if shadow:
        sh = pygame.Surface((rect.width+10, rect.height+10), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0,0,0,90), sh.get_rect())
        surface.blit(sh, (rect.x-5, rect.y-5))
    pill = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(pill, bg, pygame.Rect(0,0,rect.width,rect.height), border_radius=radius)
    pygame.draw.rect(pill, border, pygame.Rect(0,0,rect.width,rect.height), width=border_w, border_radius=radius)
    surface.blit(pill, rect.topleft)

def draw_button(surface, rect, text, ui_color, hover=False):
    base_alpha = 130 if not hover else 190
    bg_col = (20, 20, 30, base_alpha)
    draw_pill(surface, rect, bg=bg_col, border=ui_color, border_w=1, radius=16, shadow=True)
    label = STAT_FONT.render(text, True, ui_color)
    lx = rect.x + (rect.w - label.get_width()) // 2
    ly = rect.y + (rect.h - label.get_height()) // 2
    surface.blit(label, (lx, ly))

def get_menu_button_rects():
    W, H = 260, 48
    gap = 14
    cx, start_y = WIDTH//2 - W//2, HEIGHT//2 - (H*6 + gap*5)//2 + 20
    names = ["train", "play", "champ", "skin", "bg", "quit"]
    rects = {}
    for i, name in enumerate(names):
        rects[name] = pygame.Rect(cx, start_y + i*(H+gap), W, H)
    return rects

# === Bouton Graph (relooké) ===
BTN_W, BTN_H = 150, 36
GRAPH_BTN_RECT = pygame.Rect(WIDTH - BTN_W - 12, 10, BTN_W, BTN_H)

def is_graph_button(pos): return GRAPH_BTN_RECT.collidepoint(pos)
def draw_graph_button(surface, ui_color):
    hover = GRAPH_BTN_RECT.collidepoint(pygame.mouse.get_pos())
    draw_button(surface, GRAPH_BTN_RECT, "📈 Graph (G)", ui_color, hover=hover)

def draw_centered_score(surface, score, ui_color):
    txt = BIG_FONT.render(f"{score}", True, ui_color)
    pad_x, pad_y = 24, 8
    W_, H_ = txt.get_width() + pad_x*2, txt.get_height() + pad_y*2
    rect = pygame.Rect((WIDTH - W_)//2, 14, W_, H_)
    draw_pill(surface, rect, bg=(0,0,0,140), border=ui_color, border_w=1, radius=18, shadow=True)
    surface.blit(txt, (rect.x + pad_x, rect.y + pad_y))

def handle_graph_request():
    show_last_10_sessions_matplotlib()

# CSV sessions
def record_session(mode: str, best_score: int):
    entry = {"mode": mode, "score": int(best_score)}
    SESS_LOG.append(entry)
    new_file = not os.path.exists(SESS_CSV)
    try:
        with open(SESS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "mode", "session_index", "max_score"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), mode, len(SESS_LOG), int(best_score)])
    except Exception as e:
        print("session log error:", e)
    if len(SESS_LOG) % 10 == 0:
        show_last_10_sessions_matplotlib()

# ============== Utils ==============
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def make_net(genome, config):
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        return neat.nn.RecurrentNetwork.create(genome, config)

class UISyncReporter(BaseReporter):
    def start_generation(self, generation):
        global gen
        gen = generation

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
    """Tuyaux avec sprites PNG (haut retourné + bas). Collision logique via gap top/bottom."""
    def __init__(self, x: int, gap_center: int, gap_size: int):
        self.x = float(x)
        self.gap_center = int(gap_center)
        self.gap_size = int(gap_size)
        self.passed = False
        # images
        self.top_img = PIPE_TOP_IMG
        self.bot_img = PIPE_BOTTOM_IMG
        self.width   = self.bot_img.get_width()
        self.top_h   = self.top_img.get_height()
        self.bot_h   = self.bot_img.get_height()

    @property
    def top(self):    return self.gap_center - self.gap_size//2
    @property
    def bottom(self): return self.gap_center + self.gap_size//2

    def move(self):
        self.x -= PIPE_VEL

    def draw(self, win):
        # positionne l'extrémité du tuyau vers le gap
        top_y = self.top - self.top_h     # le bas du tuyau top arrive à self.top
        bot_y = self.bottom               # le haut du tuyau bottom part à self.bottom
        win.blit(self.top_img, (int(self.x), int(top_y)))
        win.blit(self.bot_img, (int(self.x), int(bot_y)))

class Base:
    """Base texturée PNG qui défile en boucle sous le FLOOR_Y."""
    def __init__(self, y: int):
        self.y = y
        self.img = BASE_IMG
        self.w   = self.img.get_width()
        self.x1  = 0
        self.x2  = self.w

    def move(self):
        self.x1 -= PIPE_VEL
        self.x2 -= PIPE_VEL
        if self.x1 + self.w <= 0:
            self.x1 = self.x2 + self.w
        if self.x2 + self.w <= 0:
            self.x2 = self.x1 + self.w

    def draw(self, win):
        win.blit(self.img, (int(self.x1), self.y))
        win.blit(self.img, (int(self.x2), self.y))

# ============== Rendu ==============
def draw_window(win, birds: List[Bird], pipes: List[Pipe], base: Base, score: int, gen_num: int, alive: int, skin_idx: int):
    _, _, _, _, ui = SKINS[skin_idx]
    win.blit(BG_IMG, (0, 0))
    for p in pipes: p.draw(win)
    base.draw(win)
    # Score centré
    draw_centered_score(win, score, ui)
    # Infos à gauche
    gen_surf   = STAT_FONT.render(f"Gen: {gen_num}", True, ui)
    alive_surf = STAT_FONT.render(f"Alive: {alive}", True, ui)
    win.blit(gen_surf, (10, 10))
    win.blit(alive_surf, (10, 36))
    # Bouton graph
    draw_graph_button(win, ui)
    for b in birds: b.draw(win)
    pygame.display.update()

def draw_menu(win, skin_idx: int):
    _, _, _, _, ui = SKINS[skin_idx]
    win.blit(MENU_BG_IMG, (0, 0))
    title = BIG_FONT.render("FLAPPY BIRD — NEAT v2", True, ui)
    win.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2 - 190)))
    rects = get_menu_button_rects()
    labels = {
        "train": "🚀 Entraîner l'IA (1)",
        "play" : "🎮 Jouer (2)",
        "champ": "👑 Champion (3)",
        "skin" : "🎨 Skin (S)",
        "bg"   : "🖼️ Background (B)",
        "quit" : "❌ Quitter (ESC)",
    }
    mouse = pygame.mouse.get_pos()
    for k, r in rects.items():
        draw_button(win, r, labels[k], ui, hover=r.collidepoint(mouse))
    draw_graph_button(win, ui)
    hint = STAT_FONT.render(f"Sessions jouées: {len(SESS_LOG)} (graphe toutes les 10)", True, ui)
    win.blit(hint, (12, HEIGHT - 30))
    pygame.display.update()

def draw_banner(win, text: str, skin_idx: int):
    _, _, _, _, ui = SKINS[skin_idx]
    banner = BIG_FONT.render(text, True, ui)
    overlay = pygame.Surface((WIDTH, 120), pygame.SRCALPHA); overlay.fill((0,0,0,110))
    win.blit(overlay, (0, 50))
    win.blit(banner, banner.get_rect(center=(WIDTH//2, 110)))
    pygame.display.update()

# ============== Génération tuyaux variés (gap sûr) ==============
SAFE_PAD         = 80
GAP_MIN          = 120
GAP_MAX          = 230
MAX_CENTER_STEP  = 42
GAP_JITTER_PROB  = 0.85
GAP_JITTER_DELTA = 26
GAP_STRONG_PROB  = 0.25
GAP_STRONG_BOOST = 22

_center_state = None
_gap_state    = None

def _safe_center(center:int, gap:int) -> int:
    half = gap // 2
    lo   = SAFE_PAD + half
    hi   = FLOOR_Y - SAFE_PAD - half
    return max(lo, min(hi, center))

def reset_gap_sequence():
    global _center_state, _gap_state
    _center_state = None
    _gap_state    = None

def next_pipe_params():
    global _center_state, _gap_state
    if _center_state is None:
        base_gap      = min(max(PIPE_GAP, GAP_MIN), GAP_MAX)
        _gap_state    = base_gap
        _center_state = _safe_center(FLOOR_Y // 2, _gap_state)
        return _center_state, _gap_state
    step          = random.randint(-MAX_CENTER_STEP, MAX_CENTER_STEP)
    candidate_ctr = _center_state + step
    gap = _gap_state
    if random.random() < GAP_JITTER_PROB:
        delta = random.randint(-GAP_JITTER_DELTA, GAP_JITTER_DELTA)
        if random.random() < GAP_STRONG_PROB:
            delta += random.choice([-GAP_STRONG_BOOST, GAP_STRONG_BOOST])
        gap = max(GAP_MIN, min(GAP_MAX, gap + delta))
    center = _safe_center(candidate_ctr, gap)
    _center_state = center
    _gap_state    = gap
    return center, gap

def prime_pipes(count=3):
    """Crée quelques tuyaux espacés à droite (anti-stack garanti)."""
    reset_gap_sequence()
    spacing = max(MIN_PIPE_SPACING_PX, int(PIPE_VEL * FPS * TIME_BETWEEN_PIPES))
    x0 = WIDTH + SPAWN_OFFSET_X
    pipes = []
    for k in range(count):
        c, g = next_pipe_params()
        pipes.append(Pipe(x0 + k*spacing, c, g))
    return pipes

def spawn_pipe(pipes):
    """Ajoute un tuyau à droite, en respectant un espacement minimum vs le dernier."""
    c, g = next_pipe_params()
    if pipes:
        base_x = max(WIDTH + SPAWN_OFFSET_X, pipes[-1].x + MIN_PIPE_SPACING_PX)
    else:
        base_x = WIDTH + SPAWN_OFFSET_X
    pipes.append(Pipe(base_x, c, g))

# ============== Graphe Matplotlib ==============
def _plot_last10_process(data, total_len):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    scores = [d["score"] for d in data]
    xs = list(range(len(data)))
    labels = [str(total_len - (len(data)-1) + i) for i in range(len(data))]
    auto_idx = [i for i,d in enumerate(data) if d["mode"] == "auto"]
    man_idx  = [i for i,d in enumerate(data) if d["mode"] == "manual"]
    auto_scores = [scores[i] for i in auto_idx]
    man_scores  = [scores[i] for i in man_idx]
    plt.figure("Scores — 10 dernières sessions (ligne + points)")
    plt.clf()
    plt.plot(xs, scores, "-o", linewidth=2, markersize=6, label="Score", color="#1f77b4")
    if auto_idx:
        plt.scatter(auto_idx, auto_scores, s=60, c="c", marker="o", edgecolors="black", zorder=3, label="Auto")
    if man_idx:
        plt.scatter(man_idx,  man_scores,  s=60, c="gold", marker="o", edgecolors="black", zorder=3, label="Manuel")
    plt.xticks(xs, labels, rotation=0)
    plt.xlabel("Session"); plt.ylabel("Score max"); plt.title("Évolution des 10 derniers scores")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    try: plt.show(block=True)
    except Exception: pass

def show_last_10_sessions_matplotlib():
    if not SESS_LOG:
        return
    data = SESS_LOG[-10:]
    p = Process(target=_plot_last10_process, args=(data, len(SESS_LOG)))
    p.daemon = False
    p.start()

# ============== Évaluation (NEAT) ==============
def eval_genomes(genomes, config):
    global SKIN_IDX, STOP_TRAINING, SESSION_AUTO_BEST_SCORE

    nets, ge, birds = [], [], []
    _, _, _, bird_c, _ = SKINS[SKIN_IDX]

    for _, g in genomes:
        net = make_net(g, config)
        nets.append(net)
        birds.append(Bird(120, HEIGHT//2, color=bird_c))
        g.fitness = 0.0
        ge.append(g)

    base  = Base(FLOOR_Y)
    pipes = prime_pipes(count=3)
    score = 0
    SESSION_AUTO_BEST_SCORE = 0

    clock = pygame.time.Clock()
    spawn_acc = 0.0
    run = True
    while run and len(birds) > 0 and not STOP_TRAINING:
        dt = clock.tick(FPS) / 1000.0
        spawn_acc += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_graph_button(event.pos): handle_graph_request()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g: handle_graph_request()
                if event.key == pygame.K_s: SKIN_IDX = (SKIN_IDX + 1) % len(SKINS)
                if event.key == pygame.K_b: cycle_background(+1)
                if event.key == pygame.K_ESCAPE: STOP_TRAINING = True

        if len(birds) == 0: break

        # cible
        pipe_ind = 0
        if len(pipes) > 1 and all(b.x > pipes[0].x + pipes[0].width for b in birds):
            pipe_ind = 1
        target = pipes[pipe_ind]

        dead_idx = set()
        for i in range(len(birds)):
            b = birds[i]
            b.move()
            ge[i].fitness += 0.05

            dx  = (target.x + target.width/2) - b.x
            dy  = (target.gap_center - b.y)
            ttc = dx / max(1.0, PIPE_VEL) / FPS
            if len(pipes) > pipe_ind + 1:
                nxt = pipes[pipe_ind + 1]
                dx2 = (nxt.x + nxt.width/2) - b.x
                dy2 = (nxt.gap_center - b.y)
            else:
                dx2 = WIDTH; dy2 = 0.0

            inputs = (
                b.y / HEIGHT,
                max(-15.0, min(15.0, b.vel)) / 15.0,
                clamp01(dx / WIDTH),
                dy / HEIGHT,
                clamp01(ttc),
                target.gap_size / HEIGHT,
                target.top / HEIGHT,
                target.bottom / HEIGHT,
                clamp01(dx2 / WIDTH),
                dy2 / HEIGHT,
            )
            if nets[i].activate(inputs)[0] > 0.5:
                b.jump()
                ge[i].fitness -= 0.002

            err = abs(dy) / max(1.0, (target.gap_size / 2.0))
            ge[i].fitness += max(0.0, 0.05 * (1.0 - min(1.0, err)))

        # Pipes : collisions + score
        add_point = False
        rem_pipes = []

        for p in pipes:
            for i, b in enumerate(birds):
                if p.x < b.x + b.radius < p.x + p.width:
                    if b.y - b.radius < p.top or b.y + b.radius > p.bottom:
                        if i not in dead_idx:
                            try: SND_HIT.play(); SND_DIE.play()
                            except: pass
                        ge[i].fitness -= 1.0
                        dead_idx.add(i)
                if b.y + b.radius >= FLOOR_Y or b.y - b.radius <= 0:
                    if i not in dead_idx:
                        try: SND_HIT.play(); SND_DIE.play()
                        except: pass
                    ge[i].fitness -= 1.0
                    dead_idx.add(i)

            if not p.passed and any(b.x > p.x + p.width for b in birds):
                p.passed = True
                add_point = True

            p.move()
            if p.x + p.width < 0: rem_pipes.append(p)

        if add_point:
            score += 1
            SESSION_AUTO_BEST_SCORE = max(SESSION_AUTO_BEST_SCORE, score)
            try: SND_POINT.play()
            except: pass
            for g in ge: g.fitness += 5.0

        for r in rem_pipes:
            if r in pipes: pipes.remove(r)

        if dead_idx:
            for idx in sorted(dead_idx, reverse=True):
                birds.pop(idx); nets.pop(idx); ge.pop(idx)

        # Spawn régulier (cadencé) — anti-stack: x basé sur dernier pipe
        while spawn_acc >= TIME_BETWEEN_PIPES:
            spawn_pipe(pipes)
            spawn_acc -= TIME_BETWEEN_PIPES

        base.move()
        draw_window(WIN, birds, pipes, base, score, gen, len(birds), SKIN_IDX)

# ============== Entraînement infini + checkpoints ==============
def train_ai(config_path, skin_idx=0, save_winner=True, ckpt_every=10):
    global SKIN_IDX, STOP_TRAINING, SESSION_AUTO_BEST_SCORE
    SKIN_IDX = skin_idx
    STOP_TRAINING = False
    SESSION_AUTO_BEST_SCORE = 0

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(UISyncReporter())
    p.add_reporter(neat.Checkpointer(generation_interval=ckpt_every, filename_prefix="neat-checkpoint-"))

    best_fit = float("-inf")

    while not STOP_TRAINING:
        p.run(eval_genomes, 1)
        if STOP_TRAINING: break
        pop_vals = [g for g in p.population.values() if g.fitness is not None]
        if pop_vals:
            current_best = max(pop_vals, key=lambda g: g.fitness)
            if current_best.fitness > best_fit:
                best_fit = current_best.fitness
                if save_winner:
                    with open("winner.pkl", "wb") as f:
                        pickle.dump(current_best, f)

    record_session("auto", SESSION_AUTO_BEST_SCORE)

# ============== Champion (replay) ==============
def watch_champion(config_path, winner_path="winner.pkl", skin_idx=0):
    global SKIN_IDX
    SKIN_IDX = skin_idx
    if not os.path.exists(winner_path): return

    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    net = make_net(winner, config)

    _, _, _, bird_c, _ = SKINS[SKIN_IDX]
    base  = Base(FLOOR_Y)
    bird  = Bird(120, HEIGHT//2, color=bird_c)
    pipes = prime_pipes(count=3)
    score = 0

    clock = pygame.time.Clock()
    spawn_acc = 0.0
    run = True
    while run:
        dt = clock.tick(FPS) / 1000.0
        spawn_acc += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_graph_button(event.pos): handle_graph_request()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g: handle_graph_request()
                if event.key == pygame.K_s: SKIN_IDX = (SKIN_IDX + 1) % len(SKINS)
                if event.key == pygame.K_b: cycle_background(+1)
                if event.key == pygame.K_ESCAPE: run = False

        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].width:
            pipe_ind = 1
        target = pipes[pipe_ind]

        bird.move()
        dx  = (target.x + target.width/2) - bird.x
        dy  = (target.gap_center - bird.y)
        ttc = dx / max(1.0, PIPE_VEL) / FPS
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

        add_point = False
        rem = []
        for p in pipes:
            if p.x < bird.x + bird.radius < p.x + p.width:
                if bird.y - bird.radius < p.top or bird.y + bird.radius > p.bottom:
                    run = False
            if not p.passed and bird.x > p.x + p.width:
                p.passed = True; add_point = True
            p.move()
            if p.x + p.width < 0: rem.append(p)

        if add_point:
            score += 1
            try: SND_POINT.play()
            except: pass

        for r in rem:
            if r in pipes: pipes.remove(r)

        while spawn_acc >= TIME_BETWEEN_PIPES:
            spawn_pipe(pipes)
            spawn_acc -= TIME_BETWEEN_PIPES

        base.move()
        draw_window(WIN, [bird], pipes, base, score, gen, 1, SKIN_IDX)

# ============== Mode manuel ==============
def play_manual(skin_idx=0):
    global SKIN_IDX
    SKIN_IDX = skin_idx
    _, _, _, bird_c, _ = SKINS[SKIN_IDX]

    base  = Base(FLOOR_Y)
    bird  = Bird(120, HEIGHT//2, color=bird_c)
    pipes = prime_pipes(count=3)
    score = 0

    clock = pygame.time.Clock()
    spawn_acc = 0.0
    run = True
    while run:
        dt = clock.tick(FPS) / 1000.0
        spawn_acc += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_graph_button(event.pos): handle_graph_request()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: bird.jump()
                if event.key == pygame.K_g:     handle_graph_request()
                if event.key == pygame.K_s:     SKIN_IDX = (SKIN_IDX + 1) % len(SKINS)
                if event.key == pygame.K_b:     cycle_background(+1)
                if event.key == pygame.K_ESCAPE: run = False

        bird.move()
        add_point = False
        rem = []

        for p in pipes:
            if p.x < bird.x + bird.radius < p.x + p.width:
                if bird.y - bird.radius < p.top or bird.y + bird.radius > p.bottom:
                    try: SND_HIT.play(); SND_DIE.play()
                    except: pass
                    run = False
            if not p.passed and bird.x > p.x + p.width:
                p.passed = True; add_point = True
            p.move()
            if p.x + p.width < 0: rem.append(p)

        if bird.y + bird.radius >= FLOOR_Y or bird.y - bird.radius <= 0:
            try: SND_HIT.play(); SND_DIE.play()
            except: pass
            run = False

        if add_point:
            score += 1
            try: SND_POINT.play()
            except: pass

        for r in rem:
            if r in pipes: pipes.remove(r)

        while spawn_acc >= TIME_BETWEEN_PIPES:
            spawn_pipe(pipes)
            spawn_acc -= TIME_BETWEEN_PIPES

        base.move()
        draw_window(WIN, [bird], pipes, base, score, gen, 1, SKIN_IDX)

    record_session("manual", score)

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

feed_forward               = True
initial_connection         = full_direct

bias_init_mean             = 0.0
bias_init_stdev            = 1.0
bias_max_value             = 30.0
bias_min_value             = -30.0
bias_mutate_power          = 0.6
bias_mutate_rate           = 0.7
bias_replace_rate          = 0.1

response_init_mean         = 0.0
response_init_stdev        = 1.0
response_max_value         = 30.0
response_min_value         = -30.0
response_mutate_power      = 0.5
response_mutate_rate       = 0.7
response_replace_rate      = 0.1

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
    with open(path, "w", encoding="utf-8") as f:
        f.write(DEFAULT_NEAT_CONFIG)

# ============== Main menu loop ==============
def main():
    play_menu_music()
    local_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    write_neat_config_always(config_path)

    clock = pygame.time.Clock()
    skin = 0
    while True:
        clock.tick(FPS)
        draw_menu(WIN, skin)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if is_graph_button(event.pos):
                    handle_graph_request()
                rects = get_menu_button_rects()
                if rects["train"].collidepoint(event.pos):
                    draw_banner(WIN, "Entraînement en cours... (ESC pour stopper)", skin)
                    train_ai(config_path, skin_idx=skin, save_winner=True, ckpt_every=10)
                    draw_banner(WIN, "Entraînement stoppé. [3] pour voir le champion.", skin)
                elif rects["play"].collidepoint(event.pos):
                    play_manual(skin_idx=skin)
                elif rects["champ"].collidepoint(event.pos):
                    if os.path.exists("winner.pkl"):
                        watch_champion(config_path, "winner.pkl", skin_idx=skin)
                    else:
                        draw_banner(WIN, "Pas de champion (entraîne d'abord).", skin)
                elif rects["skin"].collidepoint(event.pos):
                    skin = (skin + 1) % len(SKINS)
                elif rects["bg"].collidepoint(event.pos):
                    cycle_background(+1)
                elif rects["quit"].collidepoint(event.pos):
                    pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_g:
                    handle_graph_request()
                if event.key == pygame.K_b:
                    cycle_background(+1)
                if event.key == pygame.K_s:
                    skin = (skin + 1) % len(SKINS)
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

if __name__ == "__main__":
    main()