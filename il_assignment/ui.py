"""Local Pygame collection UI; no server, account, or network service required."""
import chess
import numpy as np
from .domains import MOVES, MOVE_IDS


class CollectionAborted(Exception):
    pass


class CollectorUI:
    def __init__(self, env):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 760))
        pygame.display.set_caption("Learning from Demonstrations | Collection studio")
        self.font = pygame.font.SysFont("dejavusans", 18)
        self.small = pygame.font.SysFont("dejavusans", 14)
        self.large = pygame.font.SysFont("dejavusans", 30, bold=True)
        self.clock = pygame.time.Clock()
        self.env = env
        self.selected = None
        self.promotion = chess.QUEEN
        self.control = np.array([-1., 0.], np.float32)
        self.notice = ""
        self.progress = ""

    def text(self, value, xy, color=(228, 234, 240), font=None):
        self.screen.blit((font or self.font).render(str(value), True, color), xy)

    def events(self):
        events = self.pg.event.get()
        if any(e.type == self.pg.QUIT or
               (e.type == self.pg.KEYDOWN and e.key == self.pg.K_ESCAPE) for e in events):
            raise CollectionAborted()
        return events

    def ready(self, message):
        self.control[:] = [-1, 0]
        self.selected = None
        self.notice = message
        while True:
            self.draw()
            self.text("ENTER: start episode    ESC: stop (completed episodes are saved)", (28, 720))
            self.pg.display.flip()
            if any(e.type == self.pg.KEYDOWN and e.key == self.pg.K_RETURN for e in self.events()):
                self.notice = ""
                return
            self.clock.tick(30)

    def draw(self):
        p = self.pg
        self.screen.fill((19, 33, 48))
        self.text(self.env.spec.name.upper(), (26, 17), (221, 188, 126), self.large)
        self.text(self.progress, (28, 57))
        if self.env.spec.name.startswith("lander"):
            frame = self.env.env.render()
            if frame is not None:
                surface = p.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                self.screen.blit(p.transform.smoothscale(surface, (660, 440)), (24, 100))
            self.text(f"Return {self.env.total_reward:7.1f} | physics steps {self.env.physics_steps}/600*", (28, 555))
            self.text("*Configured cap may differ. Actions repeat for 2 physics steps (25 Hz).", (28, 582), font=self.small)
            if self.env.spec.continuous:
                self.text("CONTINUOUS CONTROL", (720, 110), (221, 188, 126))
                box = p.Rect(725, 160, 230, 230)
                p.draw.rect(self.screen, (53, 74, 91), box)
                p.draw.line(self.screen, (110, 132, 148), (840, 160), (840, 390))
                point = (int(725 + (self.control[1] + 1) * 115), int(160 + (1 - self.control[0]) * 115))
                p.draw.circle(self.screen, (233, 191, 113), point, 8)
                for i, s in enumerate(["Hold mouse in pad:", "up = main throttle", "left/right = side throttle", "Or W/S: ramp main", "A/D: ramp side", "SPACE: main engine off", f"main={self.control[0]:+.2f}", f"side={self.control[1]:+.2f}"]):
                    self.text(s, (720, 412 + i * 25), font=self.small)
            else:
                for i, s in enumerate(["DISCRETE CONTROL", "UP: main engine (2)", "LEFT: left jet (1)", "RIGHT: right jet (3)", "No key: coast (0)", "Main has priority."]):
                    self.text(s, (715, 130 + i * 35), font=self.small)
        else:
            self._board()
        self.text(self.notice[:105], (28, 659), (245, 204, 128), self.small)
        self.text("H: accept teacher suggestion (recorded as assisted) | ESC: abort current episode", (28, 690), font=self.small)
        p.display.flip()

    def _board(self):
        p, env = self.pg, self.env
        _, mask = env.observe()
        for sq in chess.SQUARES:
            x, y = 32 + chess.square_file(sq) * 64, 110 + (7 - chess.square_rank(sq)) * 64
            rect = p.Rect(x, y, 64, 64)
            base = (222, 218, 201) if (chess.square_file(sq) + chess.square_rank(sq)) % 2 else (111, 137, 140)
            p.draw.rect(self.screen, base, rect)
            if self.selected == sq:
                p.draw.rect(self.screen, (251, 188, 56), rect, 5)
            piece = env.board.piece_at(sq)
            if piece:
                fill = (245, 242, 228) if piece.color else (35, 40, 52)
                p.draw.circle(self.screen, fill, (x + 32, y + 31), 24)
                label = self.large.render(piece.symbol().upper(), True, (28, 38, 45) if piece.color else (245, 242, 229))
                self.screen.blit(label, label.get_rect(center=(x + 32, y + 31)))
            self.text(chess.square_name(sq), (x + 2, y + 47), (20, 29, 40), self.small)
        lines = [f"WHITE TURN {env.moves + 1}/{env.prefix_moves}", f"Decision: {env.phase.upper()}", "You always play White.", "Click source, then destination.", "Promotion: Q / R / B / N", f"Selected: {chess.piece_name(self.promotion)}", ""]
        lines += ["Black replies automatically.", "Only legal moves are accepted.", "After the prefix, Stockfish takes over."]
        for i, line in enumerate(lines):
            self.text(line, (582, 115 + i * 28), (221, 188, 126) if i < 2 else (227, 233, 238), self.small)
        self.text(env.last_event[:105], (30, 630), font=self.small)

    def action(self):
        return self._lander_action() if self.env.spec.name.startswith("lander") else self._board_action()

    def _lander_action(self):
        p = self.pg
        events = self.events()
        if any(e.type == p.KEYDOWN and e.key == p.K_h for e in events):
            return self.env.expert_action(), "assisted"
        keys = p.key.get_pressed()
        if self.env.spec.continuous:
            mx, my = p.mouse.get_pos()
            if p.mouse.get_pressed()[0] and 725 <= mx <= 955 and 160 <= my <= 390:
                self.control[:] = [1 - (my - 160) / 115, (mx - 725) / 115 - 1]
            else:
                self.control[0] += .08 * (keys[p.K_w] - keys[p.K_s])
                if keys[p.K_a] or keys[p.K_d]:
                    self.control[1] += .12 * (keys[p.K_d] - keys[p.K_a])
                else:
                    self.control[1] *= .8
                if keys[p.K_SPACE]:
                    self.control[0] = -1
                np.clip(self.control, -1, 1, out=self.control)
            action = self.control.copy()
        else:
            action = 2 if keys[p.K_UP] else 1 if keys[p.K_LEFT] else 3 if keys[p.K_RIGHT] else 0
        self.draw()
        self.clock.tick(25)
        return action, "human"

    def _board_action(self):
        p, env = self.pg, self.env
        while True:
            self.draw()
            for event in self.events():
                if event.type == p.KEYDOWN:
                    if event.key == p.K_h:
                        self.selected = None
                        return env.expert_action(), "assisted"
                    promotion = {p.K_q: chess.QUEEN, p.K_r: chess.ROOK, p.K_b: chess.BISHOP, p.K_n: chess.KNIGHT}
                    if event.key in promotion:
                        self.promotion = promotion[event.key]
                if event.type != p.MOUSEBUTTONDOWN or event.button != 1:
                    continue
                x, y = event.pos
                if not (32 <= x < 544 and 110 <= y < 622):
                    continue
                sq = chess.square((x - 32) // 64, 7 - (y - 110) // 64)
                piece = env.board.piece_at(sq)
                if self.selected is None or (piece and piece.color == chess.WHITE):
                    self.selected = sq if piece and piece.color == chess.WHITE else None
                    continue
                source_piece = env.board.piece_at(self.selected)
                promote = self.promotion if source_piece and source_piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7) else None
                move = chess.Move(self.selected, sq, promotion=promote)
                action = MOVE_IDS.get(move)
                if action is not None and env.observe()[1][action]:
                    self.selected = None
                    self.notice = ""
                    return action, "human"
                self.notice = "That move is not an available request. Choose another destination."
            self.clock.tick(30)

    def close(self):
        self.pg.quit()
