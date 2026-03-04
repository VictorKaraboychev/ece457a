import math
import random
import copy

INF = 10**9
BOARD_SIZE = 4

DIRECTIONS = [
    (-1,-1), (-1,0), (-1,1),
    (0,-1),          (0,1),
    (1,-1),  (1,0),  (1,1)
]

def triangular_depth(n):
    return int((math.sqrt(8*n + 1) - 1) // 2)


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def generate_moves(board, player):
    moves = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            stack = board[r][c]
            if stack * player <= 0:
                continue

            count = abs(stack)

            for dr, dc in DIRECTIONS:
                remaining = count
                k = 1
                nr, nc = r + dr, c + dc
                path = []

                while in_bounds(nr, nc) and remaining > 0:
                    # Cannot land on opponent stacks (would cancel stones); only empty or friendly
                    if board[nr][nc] * player < 0:
                        break
                    drop = min(k, remaining)
                    path.append((nr, nc, drop))
                    remaining -= drop

                    k += 1
                    nr += dr
                    nc += dc

                # Valid move if we used all stones, or put remainder in last cell (when next step would be OOB)
                if path:
                    if remaining > 0:
                        # Dump remaining stones in the last stack
                        lr, lc, last_drop = path[-1]
                        path[-1] = (lr, lc, last_drop + remaining)
                    moves.append((r, c, path))

    return moves


def apply_move(board, move, player):
    r, c, path = move
    new_board = copy.deepcopy(board)

    new_board[r][c] = 0

    for nr, nc, k in path:
        new_board[nr][nc] += player * k

    return new_board

# Weights based on the max number of moves from a given position
POSITION_WEIGHTS = [
    [3, 5, 5, 3],
    [5, 8, 8, 5],
    [5, 8, 8, 5],
    [3, 5, 5, 3],
]


def evaluate(board, player):

    my_moves = generate_moves(board, player)
    opp_moves = generate_moves(board, -player)

    if not my_moves:
        return -INF
    if not opp_moves:
        return INF

    mobility = len(my_moves) - len(opp_moves)
    central = 0
    stack_potential = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = board[r][c]
            if val == 0:
                continue

            owner = 1 if val > 0 else -1
            count = abs(val)

            weight = POSITION_WEIGHTS[r][c] * count
            depth = triangular_depth(count)

            if owner == player:
                central += weight
                stack_potential += depth
            else:
                central -= weight
                stack_potential -= depth

    return 15 * mobility + 2 * central + 8 * stack_potential


def minimax(board, depth, alpha, beta, root_player, current_player, node_count=None):
    if node_count is not None:
        node_count[0] += 1

    moves = generate_moves(board, current_player)

    if depth == 0 or not moves:
        return evaluate(board, root_player), None

    maximizing = (current_player == root_player)
    best_move = None

    if maximizing:
        max_eval = -INF
        for move in moves:
            new_board = apply_move(board, move, current_player)
            eval_score, _ = minimax(
                new_board, depth-1, alpha, beta,
                root_player, -current_player, node_count
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break

        return max_eval, best_move

    else:
        min_eval = INF
        for move in moves:
            new_board = apply_move(board, move, current_player)
            eval_score, _ = minimax(
                new_board, depth-1, alpha, beta,
                root_player, -current_player, node_count
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return min_eval, best_move


def root_pvs(board, depth, root_player, node_count=None):
    """
    Principal Variation Search at root: first move full window, remaining moves
    tried with null window [beta, beta+1]; re-search with full window if fail-high.
    Returns (value, best_move).
    """
    moves = generate_moves(board, root_player)
    if not moves:
        return evaluate(board, root_player), None
    if depth == 0:
        return evaluate(board, root_player), None

    alpha, beta = -INF, INF
    best_move = None

    for i, move in enumerate(moves):
        new_board = apply_move(board, move, root_player)
        if i == 0:
            score, _ = minimax(
                new_board, depth - 1, alpha, beta,
                root_player, -root_player, node_count
            )
        else:
            score, _ = minimax(
                new_board, depth - 1, beta, beta + 1,
                root_player, -root_player, node_count
            )
            if score >= beta:
                score, _ = minimax(
                    new_board, depth - 1, alpha, beta,
                    root_player, -root_player, node_count
                )
        if score > alpha:
            alpha = score
            best_move = move
    return alpha, best_move

class MinimaxAgent:
    def __init__(self, player, depth=4):
        self.player = player
        self.depth = depth
        self.last_nodes = 0
        self.last_value = None

    def select_move(self, board):
        node_count = [0]
        value, move = minimax(
            board,
            self.depth,
            -INF,
            INF,
            self.player,
            self.player,
            node_count
        )
        self.last_nodes = node_count[0]
        self.last_value = value
        return move


class PVSAgent:
    def __init__(self, player, depth=4):
        self.player = player
        self.depth = depth
        self.last_nodes = 0
        self.last_value = None

    def select_move(self, board):
        node_count = [0]
        value, move = root_pvs(
            board, self.depth, self.player, node_count
        )
        self.last_nodes = node_count[0]
        self.last_value = value
        return move

class RandomAgent:
    def __init__(self, player):
        self.player = player

    def select_move(self, board):
        moves = generate_moves(board, self.player)
        if not moves:
            return None
        return random.choice(moves)


def print_board(board):
    for row in board:
        print(["{:3d}".format(x) for x in row])
    print()


def play_game(agent_black, agent_white, verbose=True):
    board = [
        [0, 0, 0, 10],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-10, 0, 0, 0],
    ]

    current_player = 1  # Black starts
    move_count = 0

    while True:
        if verbose:
            print_board(board)

        agent = agent_black if current_player == 1 else agent_white
        move = agent.select_move(board)

        if verbose and hasattr(agent, "last_nodes") and move is not None:
            print(f"  [depth={agent.depth}, nodes={agent.last_nodes}, value={agent.last_value}]")

        if move is None:
            winner = "White" if current_player == 1 else "Black"
            if verbose:
                print(f"{winner} wins!")
                print(f"Number of moves: {move_count}")
            return winner, move_count

        if move_count >= 100:
            if verbose:
                print("Draw (move limit reached).")
                print(f"Number of moves: {move_count}")
            return None, move_count

        board = apply_move(board, move, current_player)
        move_count += 1
        current_player *= -1


def run_n_games(n, agent_black, agent_white):
    """Run n games (Black vs White), return wins, losses, ties, and list of move counts when Black wins."""
    wins = losses = ties = 0
    win_move_counts = []
    for i in range(n):
        result, move_count = play_game(agent_black, agent_white, verbose=False)
        if result == "Black":
            wins += 1
            win_move_counts.append(move_count)
        elif result == "White":
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, win_move_counts

if __name__ == "__main__":
    black_agent = PVSAgent(player=1, depth=4)
    white_agent = RandomAgent(player=-1)
    
    play_game(black_agent, white_agent, verbose=True)
    
    # n_games = 100
    # print(f"Running {n_games} games (Minimax vs Random)...")
    # wins, losses, ties, win_move_counts = run_n_games(n_games, black_agent, white_agent)
    # avg_rounds_to_win = sum(win_move_counts) / len(win_move_counts) if win_move_counts else 0
    # print(f"\nResults (Black):")
    # print(f"  Wins:   {wins}")
    # print(f"  Losses: {losses}")
    # print(f"  Ties:   {ties}")
    # print(f"  Avg rounds to win: {avg_rounds_to_win:.1f}")
    