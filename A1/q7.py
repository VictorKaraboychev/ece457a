import heapq
from collections import deque
from setup import maze, start, end, display


# Get neighboring cells (up, down, left, right)
def get_neighbors(maze, pos):
    # Check if a position is valid (within bounds and not a wall)
    def is_valid(pos):
        row, col = pos
        if row < 0 or row >= len(maze) or col < 0 or col >= len(maze[0]):
            return False
        # 0 = passable, 1 = wall, 2 = start/end (also passable)
        return maze[row][col] != 1

    row, col = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_pos = (row + dr, col + dc)
        if is_valid(new_pos):
            neighbors.append(new_pos)
    return neighbors


def bfs(maze, start, end):
    # Initialize queue with start position
    queue = deque([tuple(start)])
    visited = {tuple(start)}
    came_from = {}

    start_tuple = tuple(start)
    end_tuple = tuple(end)

    while queue:
        current = queue.popleft()

        # Check if we reached the goal
        if current == end_tuple:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(list(start_tuple))
            path.reverse()
            return path

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    # No path found
    return None


def dfs(maze, start, end):
    # Initialize stack with start position
    stack = [tuple(start)]
    visited = {tuple(start)}
    came_from = {}

    start_tuple = tuple(start)
    end_tuple = tuple(end)

    while stack:
        current = stack.pop()

        # Check if we reached the goal
        if current == end_tuple:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(list(start_tuple))
            path.reverse()
            return path

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    # No path found
    return None


def astar(maze, start, end):
    def heuristic(a, b):
        # manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Initialize data structures
    open_set = []  # Priority queue: (f_score, g_score, position)
    heapq.heappush(open_set, (0, 0, tuple(start)))

    came_from = {}  # Track path
    g_score = {tuple(start): 0}  # Cost from start
    f_score = {tuple(start): heuristic(start, end)}  # Estimated total cost

    closed_set = set()

    start_tuple = tuple(start)
    end_tuple = tuple(end)

    while open_set:
        # Get node with lowest f_score
        current_f, current_g, current = heapq.heappop(open_set)

        # Skip if already processed with a better path
        if current in closed_set:
            continue

        closed_set.add(current)

        # Check if we reached the goal
        if current == end_tuple:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(list(start_tuple))
            path.reverse()
            return path

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor in closed_set:
                continue

            # Calculate tentative g_score
            tentative_g = g_score[current] + 1  # Each step costs 1

            # If this path to neighbor is better, record it
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))

    # No path found
    return None


bfs_path = bfs(maze, start, end)
dfs_path = dfs(maze, start, end)
astar_path = astar(maze, start, end)

print(f"BFS Path Cost: {len(bfs_path)}")
display(maze, bfs_path)
print(f"DFS Path Cost: {len(dfs_path)}")
display(maze, dfs_path)
print(f"A* Path Cost: {len(astar_path)}")
display(maze, astar_path)
