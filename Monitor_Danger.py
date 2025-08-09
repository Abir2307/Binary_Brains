from sklearn.base import defaultdict
from datetime import datetime
danger_memory = defaultdict(list)
def check_persistent_danger(block_id, is_danger, current_time=None, duration_sec=120, min_occurrences=5): # Check if a block is in persistent danger
    if current_time is None:
        current_time = datetime.now()
    if is_danger:
        danger_memory[block_id].append(current_time)
        danger_memory[block_id] = [t for t in danger_memory[block_id]
                                   if (current_time - t).total_seconds() < duration_sec]
        if len(danger_memory[block_id]) >= min_occurrences:
            return True
    else:
        danger_memory[block_id] = []
    return False