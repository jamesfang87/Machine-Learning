import heapq as prioq
import copy


class State:
    def __init__(self, board, parent, cost):
        self.board = board
        self.parent = parent
        self.cost = cost
        self.heuristic = self.calc_manhattan_dist()

    def calc_manhattan_dist(self):
        """
        Calculates a heuristic score for a certain game state
        :return: the heuristic score
        """
        score = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    continue

                correct_pos = divmod(self.board[i][j] - 1, 3)
                score += abs(i - correct_pos[0]) + abs(j - correct_pos[1])

        return score

    def print(self, step):
        if step != 0:
            print(f'\nStep {step}: ')
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    print('_', end=' ')
                else:
                    print(self.board[i][j], end=' ')
            print()

    # custom comparator (like in for std::sort in C++)
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def reconstruct_sol_for_output(state):
    steps = 0
    moves = []
    while state.parent is not None:
        moves.append(state)
        state = state.parent
        steps += 1

    moves.reverse()
    for i in range(steps):
        moves[i].print(i + 1)

    print(f'\nTotal number of steps: {steps}')


class Solver:
    def __init__(self, unexamined_states=None, examined_states=None, starting_state=None):
        self.unexamined_states = unexamined_states
        self.examined_states = examined_states
        self.starting_state = starting_state
        self.cur_state = None

    def find_blank_spot(self):
        for i in range(3):
            for j in range(3):
                if self.cur_state.board[i][j] == 0:
                    return i, j

    def find_possible_moves(self):
        x, y = self.find_blank_spot()
        possible_moves = []
        # check if moving the blank left is possible
        if x > 0:
            possible_moves.append('left')
        # check if moving the blank right is possible
        if x < 2:
            possible_moves.append('right')
        # check if moving the blank up is possible
        if y < 2:
            possible_moves.append('up')
        # check if moving the blank down is possible
        if y > 0:
            possible_moves.append('down')

        return possible_moves

    def change_board(self, move):
        new_board = copy.deepcopy(self.cur_state.board)
        x, y = self.find_blank_spot()
        if move == 'left':
            new_board[x][y], new_board[x - 1][y] = new_board[x - 1][y], new_board[x][y]
        elif move == 'right':
            new_board[x][y], new_board[x + 1][y] = new_board[x + 1][y], new_board[x][y]
        elif move == 'up':
            new_board[x][y], new_board[x][y + 1] = new_board[x][y + 1], new_board[x][y]
        elif move == 'down':
            new_board[x][y], new_board[x][y - 1] = new_board[x][y - 1], new_board[x][y]

        return new_board

    def solve(self):
        # initialize data structures to hold the examined and unexamined states
        self.examined_states = set()  # all visited states
        self.unexamined_states = []  # prio queue (all none visited states)

        # add the initial state to the examined states
        self.starting_state = State(board=self.starting_state, parent=None, cost=0)
        self.unexamined_states.append(self.starting_state)
        print('Starting state: ')
        self.starting_state.print(0)

        while self.unexamined_states:
            # choose the state with the lowest heuristic score
            self.cur_state = prioq.heappop(self.unexamined_states)

            # if we have reached the goal state (the state is equivalent to the goal state)
            # reconstruct the solution end the search
            if self.cur_state.calc_manhattan_dist() == 0:
                return reconstruct_sol_for_output(self.cur_state)
            else:
                # otherwise, add it to the examined states
                self.examined_states.add(self.cur_state)

                # find all possible moves from this position
                possible_moves = self.find_possible_moves()
                for move in possible_moves:
                    # change the board based on the move
                    changed_board = self.change_board(move)

                    # create a new state
                    new_state = State(changed_board, self.cur_state, self.cur_state.cost + 1)

                    # check if we have already visited the new state
                    # if not, add it to our unexamined states
                    if new_state not in self.examined_states:
                        prioq.heappush(self.unexamined_states, new_state)

        print("Failed to Solve")


solver = Solver(starting_state=[[1, 2, 3], [4, 5, 0], [7, 8, 6]])
solver.solve()
