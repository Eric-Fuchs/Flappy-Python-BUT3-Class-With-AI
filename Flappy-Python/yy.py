import pygame
import random
import sys

# === Paramètres du jeu ===
LARGEUR, HAUTEUR = 400, 600
GRAVITE = 0.5
SAUT = -8
VITESSE_TUYAUX = 3
LARGEUR_TUYAU = 60
ECART = 150
RAYON = 12
ESPACEMENT_TUYAUX = 200  # distance horizontale entre tuyaux

# === Paramètres GA ===
POP_SIZE = 20
MUTATION_RATE = 0.2
GENERATIONS = 50

pygame.init()
Ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption("Flappy Bird - GA + Manuel (Boutons + Score)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18, bold=True)
font_big = pygame.font.SysFont("Arial", 26, bold=True)
font_title = pygame.font.SysFont("Arial", 32, bold=True)

# === Classe UI: Button ===
class Button:
    def __init__(self, rect, text, on_click=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.on_click = on_click
        self.base_color = (40, 120, 255)
        self.hover_color = (60, 140, 255)
        self.text_color = (255, 255, 255)
        self.border_color = (255, 255, 255)

    def draw(self, surface):
        mx, my = pygame.mouse.get_pos()
        hover = self.rect.collidepoint(mx, my)
        color = self.hover_color if hover else self.base_color
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, self.border_color, self.rect, width=2, border_radius=12)
        label = font_big.render(self.text, True, self.text_color)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.on_click:
                self.on_click()

# === Classe Bot (GA) ===
class Bot:
    def __init__(self, threshold=None):
        self.x = 60
        self.y = HAUTEUR // 2
        self.v = 0.0
        self.alive = True
        self.score = 0              # nb de tuyaux passés
        self.frames = 0             # temps de survie
        self.fitness = 0.0
        # "ADN" = seuil de saut par rapport au centre du gap
        self.threshold = threshold if threshold is not None else random.uniform(-50, 50)
        self.front_idx = 0          # index du tuyau "en face"

    def _tuyau_en_face_idx(self, tuyaux):
        """Premier tuyau dont la colonne n'est pas encore passée par le bot."""
        for i, t in enumerate(tuyaux):
            if self.x - RAYON <= t["x"] + LARGEUR_TUYAU:
                return i
        return max(0, len(tuyaux) - 1)

    def update(self, tuyaux):
        if not self.alive:
            return

        self.frames += 1

        # Physique
        self.v += GRAVITE
        if self.v > 16:
            self.v = 16
        self.y += self.v

        # Choisir le tuyau en face + score quand il change
        if tuyaux:
            new_front = self._tuyau_en_face_idx(tuyaux)
            if new_front > self.front_idx:
                self.score += 1  # a passé un tuyau
            self.front_idx = new_front

            p = tuyaux[self.front_idx]
            centre = (p["haut"] + p["bas"]) / 2.0
            # Politique "seuil": saute si trop bas par rapport au centre du gap
            if self.y > centre + self.threshold:
                self.v = SAUT

        # Collisions: plafond/sol
        if self.y - RAYON < 0 or self.y + RAYON > HAUTEUR:
            self.alive = False
            return

        # Collisions: tuyaux (collision rectangle simple)
        for t in tuyaux:
            if (self.x + RAYON > t["x"] and self.x - RAYON < t["x"] + LARGEUR_TUYAU):
                if self.y - RAYON < t["haut"] or self.y + RAYON > t["bas"]:
                    self.alive = False
                    return

    def draw(self, surface, color=(255, 220, 0)):
        if self.alive:
            pygame.draw.circle(surface, color, (self.x, int(self.y)), RAYON)

# === Fonctions Génétique ===
def crossover(p1, p2):
    """Croisement entre deux parents (moyenne + mutation)."""
    child_threshold = (p1.threshold + p2.threshold) / 2.0
    if random.random() < MUTATION_RATE:
        child_threshold += random.uniform(-20, 20)
    return Bot(child_threshold)

def next_generation(population):
    # Trier par fitness décroissant
    population.sort(key=lambda b: b.fitness, reverse=True)
    best = population[:max(2, POP_SIZE // 4)]  # top 25% (au moins 2)

    new_pop = []
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(best, 2)
        new_pop.append(crossover(p1, p2))

    return new_pop, best[0]

# === Tuyaux ===
def creer_tuyau(x_depart=None):
    h = random.randint(80, HAUTEUR - 220)
    x = x_depart if x_depart is not None else LARGEUR
    return {"x": x, "haut": h, "bas": h + ECART}

def initialiser_tuyaux():
    """Crée une file de tuyaux espacés au départ."""
    tuyaux = []
    x = LARGEUR + 80
    for _ in range(4):
        tuyaux.append(creer_tuyau(x))
        x += ESPACEMENT_TUYAUX
    return tuyaux

def entretenir_tuyaux(tuyaux):
    """Fait avancer, retire ceux sortis, et maintient une file régulière."""
    for t in tuyaux:
        t["x"] -= VITESSE_TUYAUX
    # Retire ceux sortis de l'écran (à gauche)
    while tuyaux and tuyaux[0]["x"] + LARGEUR_TUYAU < 0:
        tuyaux.pop(0)
    # Ajoute pour garder 5 tuyaux à l'écran
    while len(tuyaux) < 5 and (not tuyaux or tuyaux[-1]["x"] < LARGEUR + ESPACEMENT_TUYAUX):
        last_x = tuyaux[-1]["x"] if tuyaux else LARGEUR
        tuyaux.append(creer_tuyau(last_x + ESPACEMENT_TUYAUX))

# === Affichage commun ===
def dessiner_scene(tuyaux):
    Ecran.fill((135, 206, 250))
    for t in tuyaux:
        pygame.draw.rect(Ecran, (0, 200, 0), (t["x"], 0, LARGEUR_TUYAU, t["haut"]))
        pygame.draw.rect(Ecran, (0, 200, 0), (t["x"], t["bas"], LARGEUR_TUYAU, HAUTEUR - t["bas"]))

# === Mode Auto (GA) ===
def entrainer():
    generation = 0
    population = [Bot() for _ in range(POP_SIZE)]

    while generation < GENERATIONS:
        generation += 1
        tuyaux = initialiser_tuyaux()

        # Simulation jusqu'à mort de tous les bots ou ESC
        while any(bot.alive for bot in population):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return  # retour menu

            entretenir_tuyaux(tuyaux)

            for bot in population:
                bot.update(tuyaux)

            # Affichage + HUD avec score (meilleur courant)
            dessiner_scene(tuyaux)
            for bot in population:
                bot.draw(Ecran)
            alive_count = sum(b.alive for b in population)
            best_current = max((b.score for b in population), default=0)
            hud = font.render(
                f"Auto | Gen {generation} | Alive: {alive_count} | Best score: {best_current} | ESC: menu",
                True, (0, 0, 0)
            )
            Ecran.blit(hud, (10, 10))
            pygame.display.flip()
            clock.tick(60)

        # Fitness : priorité au nombre de tuyaux passés, puis frames
        for bot in population:
            bot.fitness = bot.score * 1000 + bot.frames

        # Nouvelle génération
        population, best = next_generation(population)
        print(f"Gen {generation:02d} finie | Best pipes: {best.score} | Frames: {best.frames} | "
              f"Fitness: {best.fitness:.0f} | Threshold: {best.threshold:.2f}")

    # Fin de l'entraînement -> retour menu
    return

# === Mode Manuel ===
def jouer_manuel():
    x = 60
    y = HAUTEUR // 2
    v = 0.0
    score = 0
    front_idx = 0
    alive = True

    tuyaux = initialiser_tuyaux()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return  # retour menu
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if alive:
                        v = SAUT
                    else:
                        # restart partie
                        x = 60; y = HAUTEUR // 2; v = 0.0; score = 0; front_idx = 0; alive = True
                        tuyaux = initialiser_tuyaux()

        if alive:
            # Physique
            v += GRAVITE
            if v > 16: v = 16
            y += v

            # Tuyau en face + score
            new_front = front_idx
            for i, t in enumerate(tuyaux):
                if x - RAYON <= t["x"] + LARGEUR_TUYAU:
                    new_front = i
                    break
            if new_front > front_idx:
                score += 1
            front_idx = new_front

            # Collisions
            if y - RAYON < 0 or y + RAYON > HAUTEUR:
                alive = False
            else:
                t = tuyaux[front_idx]
                if (x + RAYON > t["x"] and x - RAYON < t["x"] + LARGEUR_TUYAU):
                    if y - RAYON < t["haut"] or y + RAYON > t["bas"]:
                        alive = False

            # Entretien des tuyaux
            entretenir_tuyaux(tuyaux)

        # Affichage
        dessiner_scene(tuyaux)
        color = (255, 220, 0) if alive else (180, 180, 180)
        pygame.draw.circle(Ecran, color, (x, int(y)), RAYON)

        # HUD score
        hud = font.render(f"Manuel | Score: {score}  (SPACE/UP: saut, ESC: menu)", True, (0, 0, 0))
        Ecran.blit(hud, (10, 10))

        if not alive:
            over = font.render("Perdu ! SPACE/UP: recommencer | ESC: menu", True, (0, 0, 0))
            Ecran.blit(over, (10, 35))

        pygame.display.flip()
        clock.tick(60)

# === Menu principal (Boutons) ===
def menu():
    # Crée les boutons
    btn_w = 240; btn_h = 50
    x_center = LARGEUR // 2 - btn_w // 2
    y0 = 260
    btn_auto = Button((x_center, y0,            btn_w, btn_h), "Auto (GA)", on_click=lambda: entrainer())
    btn_man  = Button((x_center, y0 + 70,       btn_w, btn_h), "Manuel",    on_click=lambda: jouer_manuel())
    btn_quit = Button((x_center, y0 + 140,      btn_w, btn_h), "Quitter",   on_click=lambda: (pygame.quit(), sys.exit()))

    buttons = [btn_auto, btn_man, btn_quit]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            for b in buttons:
                b.handle_event(event)

        # Affichage menu
        Ecran.fill((25, 25, 35))
        title = font_title.render("Flappy — Menu", True, (240, 240, 240))
        subtitle = font.render("Choisis un mode :", True, (220, 220, 220))
        Ecran.blit(title, title.get_rect(center=(LARGEUR//2, 180)))
        Ecran.blit(subtitle, subtitle.get_rect(center=(LARGEUR//2, 215)))

        for b in buttons:
            b.draw(Ecran)

        pygame.display.flip()
        clock.tick(60)

# === Lancement ===
if __name__ == "__main__":
    menu()
