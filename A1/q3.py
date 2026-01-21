def get_next_states(state):
    x, y = state
    next_states = []
    # Fill the first bucket completely
    if x < capacity[0]:
        next_states.append((capacity[0], y))
    # Fill the second bucket completely
    if y < capacity[1]:
        next_states.append((x, capacity[1]))
        
    # Empty the first bucket completely
    if x > 0:
        next_states.append((0, y))
    # Empty the second bucket completely
    if y > 0:
        next_states.append((x, 0))
    
    # Pour from first bucket to second bucket
    if x > 0 and y < capacity[1]:
        transfer = min(x, capacity[1] - y)
        next_states.append((x - transfer, y + transfer))
    
    # Pour from second bucket to first bucket
    if y > 0 and x < capacity[0]:
        transfer = min(y, capacity[0] - x)
        next_states.append((x + transfer, y - transfer))
    
    return next_states

capacity = (4, 3)

to_visit = [(0, 0)]
visited_states = set()


while to_visit:
    current_state = to_visit.pop(0)
    visited_states.add(current_state)
    for next_state in get_next_states(current_state):
      x1, y1 = current_state
      x2, y2 = next_state
      print(f"{x1},{y1} {x2},{y2}")
      if next_state not in visited_states:
        to_visit.append(next_state)
