import cv2
DANGER_THRESHOLD = 5  # Threshold for danger level in blocks
# --- Utility functions (from utils.py) ---
def get_block_id(x, y, frame_width, frame_height, num_blocks_x=3, num_blocks_y=3):
    """Calculates the block ID based on coordinates."""
    block_width = frame_width // num_blocks_x
    block_height = frame_height // num_blocks_y
    block_x = int(x // block_width)
    block_y = int(y // block_height)
    block_id = block_y * num_blocks_x + block_x
    return block_id

def draw_blocks_and_info(frame, block_counts, frame_width, frame_height, num_blocks_x=3, num_blocks_y=3):
    """Draws grid and counts on the frame."""
    block_width = frame_width // num_blocks_x
    block_height = frame_height // num_blocks_y

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            block_id = by * num_blocks_x + bx
            x1 = bx * block_width
            y1 = by * block_height
            x2 = x1 + block_width
            y2 = y1 + block_height

            count = block_counts.get(block_id, 0)
            danger = count > DANGER_THRESHOLD
            print(f"[BLOCK DEBUG] block_id={block_id}, count={count}, danger={danger}")

            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text_color = (0, 0, 0) if not danger else (0, 0, 255)
            font_scale = 0.9
            thickness = 2

            Block_id = f"Block {block_id}"
            text = f"Now: {count}"
            cv2.putText(frame, Block_id, (x1 + 5, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, text, (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return frame

def draw_detections(frame, results, confidence_threshold):
    """Draws bounding boxes and labels on the frame."""
    if not results:
        return frame, 0, []
    
    people_count = 0
    confidences = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > confidence_threshold:
                    people_count += 1
                    confidences.append(conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if conf > 0.8:
                        color = (0, 255, 0)
                    elif conf > 0.6:
                        color = (0, 255, 255) 
                    else:
                        color = (0, 165, 255) 
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"Person {conf:.1%}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 30
                    
                    cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), 
                                (x1 + label_size[0] + 10, label_y), color, -1)
                    
                    cv2.putText(frame, label, (x1 + 5, label_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    number_label = f"#{people_count}"
                    cv2.putText(frame, number_label, (x1 + 5, y2 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame, people_count, confidences
