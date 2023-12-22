import pygame
import sys


class Game:
    def __init__(self, board, selected_square, black, white, blue, font, screen):
        self.board = board
        self.selected_square = selected_square
        self.black = black
        self.white = white
        self.blue = blue
        self.font = font
        self.screen = screen

    def game_ended(self):
        for i in range(3):
            if self.board[i] == ['x', 'x', 'x']:
                return 10
            elif self.board[i] == ['o', 'o', 'o']:
                return -10

        for i in range(3):
            col = []
            for j in range(3):
                col.append(self.board[j][i])

            if col == ['x', 'x', 'x']:
                return 10
            elif col == ['o', 'o', 'o']:
                return -10

        diag = []
        for i in range(3):
            diag.append(self.board[i][i])

        diag2 = []
        for i in range(1, 4):
            diag2.append(self.board[i - 1][-i])

        if diag == ['x', 'x', 'x'] or diag2 == ['x', 'x', 'x']:
            return 10
        elif diag == ['o', 'o', 'o'] or diag2 == ['o', 'o', 'o']:
            return -10

        full = True
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '*':
                    return None

        if full:
            return 0

    def draw_objects(self, *tasks):
        for task in tasks:
            if task == "board":
                for i in range(3):
                    pygame.draw.line(self.screen, self.black, [0, i * 100], [300, i * 100], 5)
                    pygame.draw.line(self.screen, self.black, [i * 100, 0], [i * 100, 300], 5)
            elif task == "text":
                for i, line in enumerate(self.board):
                    for j, n in enumerate(line):
                        if n != '*':
                            position = (j * 100 + 25, i * 100 + 25)
                            text = str(n)

                            self.screen.blit(self.font.render(text, True, self.black), position)
            elif task == "square":
                upper_left = (self.selected_square[0] * 100, self.selected_square[1] * 100)
                upper_right = ((self.selected_square[0] + 1) * 100, self.selected_square[1] * 100)
                lower_left = (self.selected_square[0] * 100, (self.selected_square[1] + 1) * 100)
                lower_right = ((self.selected_square[0] + 1) * 100, (self.selected_square[1] + 1) * 100)

                pygame.draw.line(self.screen, self.blue, upper_left, lower_left, 5)
                pygame.draw.line(self.screen, self.blue, upper_left, upper_right, 5)
                pygame.draw.line(self.screen, self.blue, upper_right, lower_right, 5)
                pygame.draw.line(self.screen, self.blue, lower_right, lower_left, 5)

    def minimax_util(self, depth, turn, alpha, beta):
        if self.game_ended() is not None:
            return self.game_ended()

        if turn == 'x':
            # maximize
            best_score = -100
            for i, j in self.get_moves():
                self.board[i][j] = 'x'
                best_score = max(best_score, self.minimax_util(depth + 1, 'o', alpha, beta))
                self.board[i][j] = '*'

                if best_score > beta:
                    break
                
                alpha = max(alpha, best_score)
            return best_score
        else:
            # minimize
            best_score = 100
            for i, j in self.get_moves():
                self.board[i][j] = 'o'
                best_score = min(best_score, self.minimax_util(depth + 1, 'x', alpha, beta))
                self.board[i][j] = '*'
                
                if best_score < alpha:
                    break

                beta = max(beta, best_score)
            return best_score

    def minimax(self, alpha, beta):
        move = None
        best = 100
        for a in range(3):
            for b in range(3):
                if game.board[a][b] == '*':
                    game.board[a][b] = 'o'
                    score = self.minimax_util(0, 'x', alpha, beta)
                    game.board[a][b] = '*'

                    if score < alpha:
                        break

                    beta = max(beta, score)

                    if score <= best:
                        best = score
                        move = (a, b)
        game.board[move[0]][move[1]] = 'o'

    def get_moves(self):
        possible_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '*':
                    possible_moves.append((i, j))
        return possible_moves

    def handle_return_code(self):
        return_code = self.game_ended()
        if return_code == 10:
            sys.exit()
        elif return_code == 0:
            sys.exit()
        elif return_code == -10:
            sys.exit()


pygame.font.init()
game = Game(
    board=[['*', '*', '*'], ['*', '*', '*'], ['*', '*', '*']],
    selected_square=(-1, -1),
    black=(0, 0, 0),
    white=(255, 255, 255),
    blue=(0, 0, 255),
    font=pygame.font.SysFont("dejavuserif", 50),
    screen=pygame.display.set_mode((300, 300)),
)

game.screen.fill(game.white)
game.draw_objects("board", "text")

var = True
while var:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            game.minimax(-1000, 1000)
            var = False
            break

while True:
    game.draw_objects("board", "text", "square")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if not (0 <= pos[1] <= 600):
                pos = None

            if pos is not None:
                game.selected_square = (pos[0] // 100, pos[1] // 100)
                game.screen.fill(game.white)
                game.draw_objects("board", "text", "square")

        if event.type == pygame.KEYDOWN:
            prev = game.board[game.selected_square[1]][game.selected_square[0]]
            if event.key == pygame.K_RETURN and prev == '*':
                # human move
                game.board[game.selected_square[1]][game.selected_square[0]] = 'x'
                print(game.handle_return_code())

                # ai move
                game.minimax(-1000, 1000)
                print(game.handle_return_code())

            game.draw_objects("board", "text", "square")

    pygame.display.update()
