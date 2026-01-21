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

    while queue:
        current = queue.popleft()

        # Check if we reached the goal
        if current == tuple(end):
            # Reconstruct path
            path = [list(start)]
            while current in came_from:
                path.insert(1, list(current))
                current = came_from[current]
            return path, visited

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    # No path found
    return None, visited


def dfs(maze, start, end):
    # Initialize stack with start position
    stack = [tuple(start)]
    visited = {tuple(start)}
    came_from = {}

    while stack:
        current = stack.pop()

        # Check if we reached the goal
        if current == tuple(end):
            # Reconstruct path
            path = [list(start)]
            while current in came_from:
                path.insert(1, list(current))
                current = came_from[current]
            return path, visited

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

    # No path found
    return None, visited


def astar(maze, start, end):
    def heuristic(a, b):
        # manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    priority_queue = []
    heapq.heappush(priority_queue, (0, 0, tuple(start)))
    visited = {tuple(start)}

    came_from = {}  # Track path
    g_score = {tuple(start): 0}  # Cost from start

    while priority_queue:
        current_f, current_g, current = heapq.heappop(priority_queue)

        # Check if we reached the goal
        if current == tuple(end):
            # Reconstruct path
            path = [list(start)]
            while current in came_from:
                path.insert(1, list(current))
                current = came_from[current]
            return path, visited

        # Explore neighbors
        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                # Calculate tentative g_score
                tentative_g = g_score[current] + 1  # Each step costs 1

                # If this path to neighbor is better, record it
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(
                        priority_queue, (f_score, tentative_g, neighbor)
                    )

    # No path found
    return None, visited


bfs_path, bfs_visited = bfs(maze, start, end)
dfs_path, dfs_visited = dfs(maze, start, end)
astar_path, astar_visited = astar(maze, start, end)

print(f"BFS Path Cost: {len(bfs_path)} Expanded: {len(bfs_visited)}")
display(maze, bfs_path, bfs_visited)
print(f"DFS Path Cost: {len(dfs_path)} Expanded: {len(dfs_visited)}")
display(maze, dfs_path, dfs_visited)
print(f"A* Path Cost: {len(astar_path)} Expanded: {len(astar_visited)}")
display(maze, astar_path, astar_visited)
