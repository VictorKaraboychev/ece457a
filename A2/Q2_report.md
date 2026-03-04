# Assignment 2 — Question 2: Conga Game (Deliverables)

## 1. Problem Representation [10 pts]

**State:** A 4×4 board. Each cell holds an integer: positive = Black (Player 1) stones, negative = White (Player 2) stones, zero = empty.

- **Initial state:** Black has 10 stones at (1,4), White has 10 at (4,1) in 1-based indexing; in code (0-indexed): Black at `board[0][3] = 10`, White at `board[3][0] = -10`.

**Actions:** For the current player, a move is a triple `(r, c, path)`:
- `(r, c)` = source cell with at least one of the player’s stones.
- `path` = list of `(row, col, drop)` along one of 8 directions (horizontal, vertical, diagonal). Stones are distributed as 1 in the first cell, 2 in the next, 3 in the next, etc.; any remainder is placed in the last cell. No step may land on an opponent-occupied cell (only empty or friendly).

**Terminal state:** The current player has no legal moves → the other player wins.

**Turn order:** Black (Player 1) moves first; players alternate.

---

## 2. Solution Description: Algorithm / Pseudocode [20 pts]

### Minimax with alpha–beta pruning

```
function MINIMAX(board, depth, α, β, root_player, current_player):
    increment node_count
    moves ← GENERATE_MOVES(board, current_player)

    if depth = 0 or moves is empty:
        return (EVALUATE(board, root_player), null)

    if current_player = root_player then  // maximizing
        max_eval ← −∞, best_move ← null
        for each move in moves:
            new_board ← APPLY_MOVE(board, move, current_player)
            (eval_score, _) ← MINIMAX(new_board, depth−1, α, β, root_player, −current_player)
            if eval_score > max_eval then max_eval ← eval_score, best_move ← move
            α ← max(α, eval_score)
            if β ≤ α then break
        return (max_eval, best_move)
    else  // minimizing
        min_eval ← +∞, best_move ← null
        for each move in moves:
            new_board ← APPLY_MOVE(board, move, current_player)
            (eval_score, _) ← MINIMAX(new_board, depth−1, α, β, root_player, −current_player)
            if eval_score < min_eval then min_eval ← eval_score, best_move ← move
            β ← min(β, eval_score)
            if β ≤ α then break
        return (min_eval, best_move)
```

### Move generation (sketch)

- For each cell with the current player’s stones and each of the 8 directions:
  - Step along that direction; at each step drop min(k, remaining) stones (k = 1, 2, 3, …).
  - If the next cell is out of bounds, put all remaining stones in the last in-bounds cell.
  - If any step would land on an opponent cell, stop that direction (no move through or onto opponent).

### Evaluation function (one of several considered)

- Terminal: no moves for current player → −∞ for root; no moves for opponent → +∞ for root.
- Otherwise: weighted sum of **mobility** (number of legal moves), **position** (centre-weighted), and **stack potential** (triangular depth of stacks). Coefficients used: e.g. 15×mobility + 2×central + 8×stack_potential.

### Null-window search (Principal Variation Search)

In standard alpha–beta we search each move with the full window [α, β]. A **null window** is a zero-width window, e.g. [β, β+1] at a max node: we only ask “is this move’s value ≥ β?”. The search returns a value; if it is **≥ β** we get a **fail-high** (the move is at least as good as our current best) and we then **re-search** that move with the full window [α, β] to get the exact score. If the null-window search returns **< β**, we know the move is worse than the current best and we **skip** it—no re-search. With good move ordering, many moves fail low in the null-window search, so we do fewer full-window searches and explore fewer nodes. This idea is used at the root in the **PVS (Principal Variation Search)** agent: the first root move is searched with full window; every other root move is first searched with null window and re-searched only on fail-high.

---

## 3. Hand-Worked Example (Search Strategy) [15 pts]

We trace minimax with alpha–beta on a small Conga position, similar to a tic-tac-toe style analysis.

### 3.1 Position and Conventions

- **Board (4×4):** Black (Player 1) has stones in the top-right area; White (Player 2) in the bottom-left. Root player is **Black**; we use **depth 2** (Black moves, then White, then we evaluate).
- **Evaluation:** From Black’s perspective. Positive = good for Black; negative = good for White. Terminal (no moves) gives +∞ (Black wins) or −∞ (White wins).
- **Notation:** Moves are labeled (e.g. **A**, **B** for Black; **A1**, **A2** for White replies after Black’s move A).

### 3.2 Example Tree (Depth 2)

Suppose at the root Black has two legal moves:

- **A:** Spread a stack right (e.g. 1 stone, then 2 in the next cell).
- **B:** Spread the same stack down.

After **A**, White has two replies: **A1** (e.g. move a white stack up) and **A2** (move it diagonally).  
After **B**, White has two replies: **B1** and **B2**.

Assume the **evaluation function** (mobility + position + stack potential) gives these values at the leaves (after Black’s move then White’s move; from Black’s view):

| Leaf | Eval |
|------|------|
| After A, A1 | 5  |
| After A, A2 | −10 |
| After B, B1 | 2  |
| After B, B2 | 6  |

### 3.3 Backing Up (Minimax)

- **White nodes (min):**
  - After A: min(5, −10) = **−10** → move A yields **−10** for Black.
  - After B: min(2, 6) = **2** → move B yields **2** for Black.
- **Root (Black, max):** max(−10, 2) = **2** → **best move is B** with value **2**.

So under pure minimax, Black chooses **B** and the backed-up value is **2**.

### 3.4 Alpha–Beta Pruning

Initial: **α = −∞**, **β = +∞** (root is a max node).

**1. Root explores move A.**  
Calls the min node (White to move) with **(α, β) = (−∞, +∞)**.

- **White tries A1:** eval = 5. Min node sets β = 5. (α, β) = (−∞, 5).
- **White tries A2:** eval = −10. Min = −10. So move A returns **−10** to the root.

Root: value from A is −10 → **α = max(−∞, −10) = −10**.  
Root still has **(α, β) = (−10, +∞)**.

**2. Root explores move B.**  
Calls the min node (White to move) with **(α, β) = (−10, +∞)**.

- **White tries B1:** eval = 2. Min node sets β = 2. So **(α, β) = (−10, 2)**.
- **White tries B2:** eval = 6. So 6 ≥ β (2). The min node would take min(2, 6) = 2, so the value from B is still 2. Here we do **not** prune B2 because we must compute the min; but we see that after B1 the min is at most 2, and B2 gives 6, so min stays 2. (Pruning would occur if we had more White moves and one of them gave a value ≤ α: we could then prune the rest of White’s moves at this node.)

So B returns **2** to the root.

Root: value from B is 2 → **α = max(−10, 2) = 2**.  
Final: **(α, β) = (2, +∞)**. **Best move: B, value 2.**

**3. Where pruning can happen**

In a **deeper** tree (e.g. depth 3), the same idea gives real pruning. Suppose we explore **B** first and B returns 2, so the root sets **α = 2**. Then for move **A** we call the min node (White) with **(α, β) = (2, +∞)**. White tries **A1**, which returns 5 → so β = 5 and **(α, β) = (2, 5)**. Now White tries **A2**. At depth 3, **A2** would be a *max* node (Black’s reply). We pass **(α, β) = (2, 5)** into A2. If **A2’s first child** (a leaf or a min node’s result) returns **6**, then 6 ≥ β, so we **prune** the rest of A2’s children: we do not need to evaluate them, because the min node above will take at most 6 and already has β = 5. So we save work by not expanding A2’s other branches.

**Summary:** Alpha–beta prunes branches that cannot change the root decision. With good move ordering (e.g. trying the current best move first), we often prune many nodes and still get the same best move and value as full minimax. In the full 4×4 Conga implementation, the same algorithm runs at depth 4 with full move generation and evaluation.

---

## 4. Random Agent: Logic and Performance [10 pts]

**Logic (algorithm):**
- On each turn, get the list of legal moves for the current player.
- If the list is empty, return no move (game over).
- Otherwise, choose one move uniformly at random with `random.choice(moves)` and return it.

**Performance index:** Win rate of Minimax (depth 4) as Black vs Random as White over 100 games, with a 100-move cap per game.

**Results (100 games, Minimax as Black vs Random as White):**

| Result | Count |
|--------|-------|
| Wins   | 79    |
| Losses| 0     |
| Ties   | 21    |
| **Avg rounds to win** | **23.6** |

*(Ties = games that hit the 100-move limit. Average rounds to win is over winning games only.)*

**Observations:**
- The Minimax agent wins the large majority of games against Random and never lost in this run.
- When Minimax wins, it typically does so in about 24 moves on average, consistent with the assignment note that a rational agent can defeat the Random agent in as little as 30 moves.
- Random creates weak positions (e.g. spreading stones poorly), so Minimax restricts its mobility and forces a win or a long game; the 21 ties are games that reached the move cap without a winner.

To reproduce: run `python A2/q2.py batch`.

---

## 5. Sample Output (Depth and Nodes Explored) [15 pts]

The program prints, for each Minimax move:
- `[depth=d, nodes=n, value=v]` where d = search depth, n = number of nodes explored in the minimax tree for that move, and v = root value.

Example (excerpt):

```
  [depth=4, nodes=371, value=-80.0]
Legal moves: [...]
...
  [depth=4, nodes=2985, value=-4.0]
...
White wins!
Number of moves: 42
```

Full sample output is produced by running:

```bash
python A2/q2.py
```

---

## 6. Time and Memory Complexity [10 pts]

**Time (per minimax call from root):**
- Let b = branching factor (number of legal moves per state), d = depth.
- Without pruning: O(b^d) nodes; each node does O(1) work plus move generation and evaluation, which are O(board size + number of moves).
- With alpha–beta, best case (perfect ordering): O(b^(d/2)); worst case (no pruning): O(b^d). In practice, pruning reduces nodes significantly (e.g. 10³–10⁴ per move at depth 4).

**Memory:**
- Recursion depth d gives O(d) stack frames.
- Each frame holds a board copy in the loop (or we copy per child); so per path O(d × board_size²) = O(d). Overall memory is O(d × 16) for board copies along one branch, plus move lists. No transposition table is used in the basic version.

---

## 7. Evaluation Functions Considered

Several evaluation ideas were considered and one (or a combination) is used in the code:

1. **Mobility:** `len(my_moves) - len(opp_moves)` — favours having more options and restricting the opponent.
2. **Centrality:** Sum of (stone count × position weight) for each cell; centre cells have higher weight (e.g. 8) than corners (3).
3. **Stack potential:** Sum of triangular_depth(stack) per stack; larger stacks can spread further and create more threats.

The implemented function combines these with weights (e.g. 15, 2, 8) so that mobility dominates but position and stack size still matter. Terminal conditions (win/loss) are handled by returning ±∞.

---

## 8. Bonus: Iterative Deepening and Null-Window Alpha-Beta (PVS) [10 pts]

Two optional improvements are implemented and can be selected at run time.

### 8.1 Null-Window Alpha-Beta (Principal Variation Search)

**Idea:** At the root, the first move is searched with the full window [α, β]. Each subsequent move is first searched with a **null window** [β, β+1]. A null-window search only answers “is this move’s value ≥ β?”. If the result is ≥ β (fail-high), the move is good enough that we re-search it with the full window to get the exact value. If the result is < β, we skip that move (it’s not better than our current best). With good move ordering, many moves fail low in the null-window search and are never re-searched, reducing the number of nodes.

**Implementation:** `root_pvs()` and `PVSAgent` in the code. The root iterates over moves; the first uses `minimax(..., alpha, beta)`; the rest use `minimax(..., beta, beta+1)` and, if the score is ≥ β, `minimax(..., alpha, beta)` again.

**Usage:** `python A2/q2.py pvs` (single game) or `python A2/q2.py batch pvs` (100 games).

### 8.2 Iterative Deepening

**Idea:** Run minimax at depth 1, then 2, …, up to a maximum depth. After each depth, we have a best move. Use that move as the **first** move to try at the next depth so that alpha–beta sees a strong move first and prunes more. Iterative deepening also gives a natural time-management policy (e.g. stop when time runs out and use the last completed depth’s best move).

**Implementation:** `iterative_deepening()` and `IterativeDeepeningAgent`. For d = 1, 2, …, max_depth we run minimax at depth d with root moves ordered so the previous depth’s best move is first. The final best move and value come from depth max_depth.

**Optional:** Iterative deepening can be combined with PVS at each depth (null-window for root moves after the first). Use `IterativeDeepeningAgent(..., use_pvs=True)` or run `python A2/q2.py idpvs`.

**Usage:** `python A2/q2.py id` (iterative deepening only), `python A2/q2.py idpvs` (iterative deepening + PVS), or `python A2/q2.py batch id` / `python A2/q2.py batch idpvs` for batch runs.

### Summary

| Agent   | Command / option      | Description                                  |
|---------|------------------------|----------------------------------------------|
| Minimax | `q2.py` or `q2.py batch` | Standard alpha–beta, fixed depth 4.          |
| PVS     | `q2.py pvs` or `q2.py batch pvs` | Null-window at root only.                   |
| ID      | `q2.py id`             | Iterative deepening, depth 1..4.             |
| ID+PVS  | `q2.py idpvs`          | Iterative deepening with PVS at each depth.  |

---

## 9. Report Quality and Completeness [10 pts]

- Problem representation, algorithm, and pseudocode are given above.
- A hand-worked minimax/alpha–beta example illustrates the search strategy.
- Random agent logic and a performance index (win rate, move count) are described.
- Sample output includes depth and nodes explored as required.
- Time and memory complexity are discussed.
- The code implements: correct Conga rules (spread 1, 2, 3… with remainder in last cell; no landing on opponent), minimax with alpha–beta, node counting, and a 100-move cap with move count printed at the end.
- Bonus (Section 8): iterative deepening and null-window (PVS) alpha–beta are implemented and selectable via command line.
