import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import pandas as pd
import bcrypt
import sqlite3
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from collections import defaultdict
from sms_notifier import notify_admin_force_sent, notify_volunteer, notify_user_summary
from lstm_predictor import predict_block_future
import matplotlib.pyplot as plt
from kde_heatmap import generate_kde
import torch
import torch.nn as nn

# === Database Setup ===
DB_FILE = "crowd_management.db"
# A default prediction threshold
DANGER_THRESHOLD = 5
# Global CSS theme injector
def inject_global_css():
    st.markdown(
        f"""
        <style>
        body, [data-testid='stAppViewContainer'] {{
            background: #181c24;
            color: #f1f1f1;
        }}
        body::before, [data-testid='stAppViewContainer']::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: url('static/Crowd.jpg') center center/cover no-repeat;
            opacity: 0.3;
            z-index: -1;
        }}
        h1.title {{
            color: #7c5cff;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.2em;
            text-align: center;
            letter-spacing: 1px;
        }}
        h2.subtitle {{
            color: #00d4ff;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5em;
            text-align: center;
        }}
        .Built-by {{
            color: #9aa0b4;
            font-size: 1.1rem;
            text-align: center;
            margin-bottom: 2em;
        }}
        .login-bottom {{
            position: fixed;
            left: 0; right: 0; bottom: 0;
            width: 100%;
            background: #222;
            padding: 1.2em 0 1.2em 0;
            text-align: center;
            box-shadow: 0 -2px 16px rgba(0,0,0,0.12);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_db():
    """Initializes the SQLite database and creates the users table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        # Create the users table with phone number
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT,
                password_hash TEXT,
                role TEXT,
                phone TEXT PRIMARY KEY
            )
        ''')
        
        # Create the control_room table based on your models.py
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS control_room (
                name TEXT,
                phone TEXT PRIMARY KEY,
                role TEXT,
                block_id TEXT
            )
        ''')
        
        # Create the block_counts table with UNIQUE constraint on (timestamp, block_id)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS block_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                block_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                UNIQUE(timestamp, block_id)
            )
        ''')

        # Check if default admin/user exists
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            default_users = [
                ('admin', bcrypt.hashpw('password'.encode('utf-8'), bcrypt.gensalt()), 'admin', '+917439644593'),
                ('user', bcrypt.hashpw('password'.encode('utf-8'), bcrypt.gensalt()), 'user', '+917439576360')
            ]
            cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", default_users)
            
            default_control_room_admin = ('admin', '+917439644593', 'admin', 'N/A')
            cursor.execute("INSERT INTO control_room VALUES (?, ?, ?, ?)", default_control_room_admin)
            
            conn.commit()
    st.session_state.db_initialized = True

if 'db_initialized' not in st.session_state:
    init_db()

# --- User Management (with SQLite) ---
def register_user(username, password, role):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            st.error("Username already exists.")
            return False
        phone = st.session_state.get('register_phone', None)
        if not phone:
            st.error("Phone number is required.")
            return False
        cursor.execute("SELECT phone FROM users WHERE phone = ?", (phone,))
        if cursor.fetchone():
            st.error(f"A user with phone {phone} already exists.")
            return False
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (username, hashed_password, role, phone))
        conn.commit()
        st.success("User registered successfully! Please log in.")
        return True

def authenticate_user(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        
        if not user_data:
            st.error("Username not found.")
            return False
            
        hashed_password = user_data[0]
        role = user_data[1]
        
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = role
            st.success("Logged in successfully!")
            st.rerun()
            return True
        else:
            st.error("Incorrect password.")
            return False

def logout():
    # Clear session and temp data on logout
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    # Clear danger memory and any other global/session data
    global danger_memory
    danger_memory.clear()
    for key in ["all_centers", "block_counts", "video_stopped", "sms_status", "top_blocks"]:
        if key in st.session_state:
            del st.session_state[key]
    if os.path.exists("temp_files"):
        for f in os.listdir("temp_files"):
            try:
                os.remove(os.path.join("temp_files", f))
            except Exception:
                pass
    # Truncate block_counts table for all users on logout
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM block_counts")
        conn.commit()
    st.rerun()

# --- Utility functions (from utils.py) ---
def get_block_id(x, y, frame_width, frame_height, num_blocks_x=3, num_blocks_y=3):
    """Calculates the block ID."""
    block_width = frame_width // num_blocks_x
    block_height = frame_height // num_blocks_y
    block_x = int(x // block_width)
    block_y = int(y // block_height)
    block_id = block_y * num_blocks_x + block_x
    return block_id

def draw_blocks_and_info(frame, block_counts, frame_width, frame_height,
                        num_blocks_x=3, num_blocks_y=3):
    """Draws grid, counts on the frame."""
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

            # LSTM prediction for this block
            try:
                pred_count, pred_danger = predict_block_future(block_id, model_path='lstm_block.pt', threshold=DANGER_THRESHOLD)
                pred_text = f"Pred: {pred_count:.1f}" if pred_count is not None else "Pred: N/A"
            except Exception as e:
                print(f"[LSTM ERROR] Block {block_id}: {e}")
                pred_text = "Pred: Err"

            color = (0, 0, 255) if danger else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Use green for normal, red for danger
            text_color = (0, 255, 0) if not danger else (0, 0, 255)
            font_scale = 0.9
            thickness = 2

            Block_id = f"Block {block_id}"
            text = f"Now: {count}"
            cv2.putText(frame, Block_id, (x1 + 5, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, text, (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, pred_text, (x1 + 5, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
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

def save_counts_to_db(timestamp, block_id, count):
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO block_counts (timestamp, block_id, count) VALUES (?, ?, ?)",
                       (ts_str, block_id, count))
        conn.commit()

# --- Page Navigation ---
def show_login_page():
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            authenticate_user(username, password)
    
    st.markdown("---")
    if st.button("Don't have an account? Register here."):
        st.session_state.view = 'register'
        st.rerun()

def show_register_page():
    st.header("Register")
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        phone = st.text_input("Phone Number")
        submit_button = st.form_submit_button("Register")
        if submit_button:
            st.session_state['register_phone'] = phone
            # Always register as role 'user'
            register_user(username, password, 'user')
    
    st.markdown("---")
    if st.button("Already have an account? Login here."):
        st.session_state.view = 'login'
        st.rerun()

@st.cache_resource
def load_yolo_model():
    """Loadingmodel with caching for better performance"""
    try:
        model = YOLO('yolo11n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video_realtime(video_path, model, confidence_threshold, user_role):
    cap = cv2.VideoCapture(video_path)
    tracker = DeepSort()
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    col_video, col_stats = st.columns([2.5, 1], gap="large")

    with col_video:
        st.markdown("### 🎥 Live Video with Detection")
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        kde_placeholder = st.empty()

    with col_stats:
        st.markdown("### 🎛️ Controls")
        stop_button = st.button("⏹️ Stop", key="stop")
        st.markdown("### 📊 Live Detection Stats")
        stats_placeholder = st.empty()
    total_detections = 0
    start_time = time.time()
    prev_max_id = 0

    # Track how many times each block is dangerous (red)
    block_red_counts = {i: 0 for i in range(9)}
    print(f"[DEBUG] Initialized block_red_counts: {block_red_counts}")

    # For KDE heatmap under video
    all_centers = []
    kde_placeholder = st.empty()

    # --- SMS notification logic ---
    red_detection_count = 0
    RED_DETECTION_THRESHOLD = 3  # Number of CRITICAL frames before SMS fires
    sms_sent = False
    sms_status_message = ""

    # --- Sample only 1 frame per second ---
    # OpenCV's CAP_PROP_DURATION is not always available, so use total_frames/fps
    duration_sec = int(total_frames / fps) if (fps > 0 and total_frames > 0) else 0
    if duration_sec == 0:
        duration_sec = total_frames if total_frames > 0 else 1
    selected_frame_indices = [int(i * fps) for i in range(duration_sec)]
    frame_idx = 0
    frame_count = 0
    notified_blocks = set()
    # Track how many times each block is dangerous (red)
    block_red_counts = {i: 0 for i in range(9)}
    # For user summary SMS
    video_stopped_early = False
    if 'video_stopped' not in st.session_state:
        st.session_state.video_stopped = False
    print(f"[DEBUG] Starting video processing, video_stopped={st.session_state.video_stopped}, block_red_counts initialized")
    while cap.isOpened() and frame_count < len(selected_frame_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < selected_frame_indices[frame_count]:
            frame_idx += 1
            continue
        if stop_button:
            video_stopped_early = True
            st.session_state.video_stopped = True
            break
        frame_start_time = time.time()
        results = model(frame, verbose=False, conf=confidence_threshold)
        confidences = []
        detection_bboxes = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0:
                        confidences.append(conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection_bboxes.append((x1, y1, x2, y2))
        annotated_frame = results[0].plot()
        detections_deep_sort = []
        for x1, y1, x2, y2 in detection_bboxes:
            detections_deep_sort.append(([x1, y1, x2-x1, y2-y1], 1, 'person'))
        tracks = tracker.update_tracks(detections_deep_sort, frame=annotated_frame)
        unique_ids = set()
        block_counts = {}
        centers = []
        max_id_this_frame = prev_max_id
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            try:
                track_id = int(track.track_id)
            except Exception:
                continue
            unique_ids.add(track_id)
            if track_id > max_id_this_frame:
                max_id_this_frame = track_id
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
            block_id = get_block_id(cx, cy, frame.shape[1], frame.shape[0])
            # Sanitize block_id to be in range 0-8
            if block_id < 0:
                block_id = 0
            elif block_id > 8:
                block_id = 8
            block_counts[block_id] = block_counts.get(block_id, 0) + 1
        sms_sent = False
        # Draw track ID and collect centers inside the track loop
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            try:
                track_id = int(track.track_id)
            except Exception:
                continue
            unique_ids.add(track_id)
            if track_id > max_id_this_frame:
                max_id_this_frame = track_id
            cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
            block_id = get_block_id(cx, cy, frame.shape[1], frame.shape[0])
            block_counts[block_id] = block_counts.get(block_id, 0) + 1
            cv2.putText(annotated_frame, f"ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            centers.append((cx, cy))
        all_centers.extend(centers)
        total_detections += len(unique_ids)
        if len(all_centers) > 2:
            unique_points = set(all_centers)
            if len(unique_points) > 2:
                xs, ys = zip(*unique_points)
                if (max(xs) != min(xs)) and (max(ys) != min(ys)):
                    count_data = [(x, y, 1) for (x, y) in all_centers]
                    try:
                        fig = generate_kde(count_data, width=2, height=1.2)
                        ax = fig.gca()
                        ax.set_xlabel('X', fontsize=10)
                        ax.set_ylabel('Y', fontsize=10)
                        # Find block with highest count
                        if block_counts:
                            max_block_id = max(block_counts, key=lambda k: block_counts[k])
                            max_block_count = block_counts[max_block_id]
                        else:
                            max_block_id = None
                            max_block_count = None
                        # Always show heatmap and block info in the same slot below video
                        with kde_placeholder.container():
                            st.markdown("#### Heatmap")
                            st.pyplot(fig)
                            if max_block_id is not None:
                                st.markdown(f"<div style='text-align:center; font-size:1.2em; margin-top:-1em;'><b>Block with Highest Count: {max_block_id} ({max_block_count})</b></div>", unsafe_allow_html=True)
                        plt.close(fig)
                    except Exception as e:
                        with kde_placeholder.container():
                            st.warning(f"Heatmap not shown (data degenerate): {e}")
        # Save block counts to DB for every block (0-8), including zero
        for block_id in range(9):
            count = block_counts.get(block_id, 0)
            save_counts_to_db(datetime.now(), block_id, count)
        # Show only current block counts
        annotated_frame = draw_blocks_and_info(annotated_frame, block_counts, frame.shape[1], frame.shape[0])
        people_count = len(unique_ids)
        prev_max_id = max_id_this_frame
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        progress = (frame_count+1) / len(selected_frame_indices) if len(selected_frame_indices) > 0 else 0
        progress_bar.progress(min(progress, 1.0))
        progress_text.text(f"Progress: {frame_count+1}/{len(selected_frame_indices)} frames ({progress:.1%})")
        elapsed_time = time.time() - start_time
        current_fps = (frame_count+1) / elapsed_time if elapsed_time > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        if people_count >= 15:
            density_level = "🔴 CRITICAL"
            red_detection_count += 1
        else:
            density_level = "🔵 LOW" if people_count < 2 else ("🟢 MEDIUM" if people_count < 5 else "🟡 HIGH")
            red_detection_count = 0  # Reset if not critical

        # Fire SMS to admin and volunteers after threshold CRITICAL frames (only for admin)
        if user_role == "admin":
            if density_level == "🔴 CRITICAL" and red_detection_count >= RED_DETECTION_THRESHOLD and not sms_sent:
                # Find dangerous blocks (block_counts > DANGER_THRESHOLD)
                current_dangerous_blocks = set(block_id for block_id, count in block_counts.items() if count > DANGER_THRESHOLD)
                print(f"[DEBUG] notified_blocks: {sorted(list(notified_blocks))}")
                print(f"[DEBUG] current_dangerous_blocks: {sorted(list(current_dangerous_blocks))}")
                new_dangerous_blocks = list(current_dangerous_blocks - notified_blocks)
                if new_dangerous_blocks:
                    # Update block_red_counts for dangerous blocks
                    for block_id in new_dangerous_blocks:
                        block_red_counts[block_id] += 1
                    print(f"[DEBUG] Updated block_red_counts: {block_red_counts}")

                    print(f"[SMS] Sending SMS to admin for blocks: {new_dangerous_blocks}")
                    success, error = notify_admin_force_sent(new_dangerous_blocks)
                    if success:
                        sms_status_message = f"✅ SMS sent to admin for blocks: {new_dangerous_blocks}"
                        print(f"[SMS] Success: {sms_status_message}")
                        notified_blocks.update(new_dangerous_blocks)
                    else:
                        sms_status_message = f"❌ SMS sending failed: {error}"
                        print(f"[SMS] Error: {sms_status_message}")
                    # Notify volunteers assigned to these blocks
                    with sqlite3.connect(DB_FILE) as conn:
                        cursor = conn.cursor()
                        for block_id in new_dangerous_blocks:
                            cursor.execute("SELECT name, phone FROM control_room WHERE role='volunteer' AND block_id=?", (str(block_id),))
                            volunteers = cursor.fetchall()
                            for name, phone in volunteers:
                                print(f"[SMS] Notifying volunteer {name} ({phone}) for block {block_id}")
                                notify_volunteer(name, phone, block_id)
                    sms_sent = True
                else:
                    sms_status_message = "⚠️ No new dangerous blocks detected, SMS not sent."
                    print(f"[SMS] Info: {sms_status_message}")
        # Track red block counts for user summary
        for block_id, count in block_counts.items():
            if count > DANGER_THRESHOLD:
                block_red_counts[block_id] += 1
        # Remove blocks from notified_blocks if they are no longer dangerous
        current_dangerous_blocks = set(block_id for block_id, count in block_counts.items() if count > DANGER_THRESHOLD)
        notified_blocks.intersection_update(current_dangerous_blocks)
        if user_role == "admin":
            print(f"[DEBUG] updated notified_blocks: {sorted(list(notified_blocks))}")
        with stats_placeholder.container():
            st.error("🔴 LIVE DETECTION ACTIVE")
            st.caption(f"Processing Frame: {frame_count+1}")
            st.metric("👥 Distinct ID People", people_count, help="Distinct current people in frame")
            st.metric("📈 Total", total_detections, help="Total detections")
            st.metric("FPS", f"{current_fps:.1f}", help="Processing speed")
            st.metric("Confidence", f"{avg_confidence:.1%}", help="Average confidence")
            st.subheader("Crowd Level")
            if "CRITICAL" in density_level:
                st.error(f"**{density_level}**")
            elif "HIGH" in density_level:
                st.warning(f"**{density_level}**")
            elif "MEDIUM" in density_level:
                st.success(f"**{density_level}**")
            else:
                st.info(f"**{density_level}**")
            if confidences:
                st.write(f"**Range:** {min(confidences):.1%} - {max(confidences):.1%}")
            st.success("✅ Model Active")
            if sms_status_message:
                if sms_status_message.startswith("✅"):
                    st.success(sms_status_message)
                elif sms_status_message.startswith("❌"):
                    st.error(sms_status_message)
                else:
                    st.info(sms_status_message)
        time_to_sleep = max(0, 1.0 - (time.time() - frame_start_time))
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        frame_count += 1
        frame_idx += 1

    cap.release()

    total_elapsed_time = time.time() - start_time
    final_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

    st.success(f"✅ Processing complete! {frame_count} frames, {total_detections} total detections")
    st.info(f" Average FPS: {final_fps:.1f} | Total detections: {total_detections}")

    # Send summary SMS to user (not admin/volunteer) after video ends or is stopped
    if user_role == "user":
        # Print final block_red_counts if video stopped or ended
        if video_stopped_early or st.session_state.video_stopped:
            debug_msg = f"[DEBUG] Final block_red_counts for user {st.session_state.username}: {dict(block_red_counts)}"
            print(debug_msg)
            st.write(debug_msg)
            st.session_state.video_stopped = False
        # Get top 3 most dangerous blocks by red count
        top_blocks = sorted(block_red_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_blocks = [block_id for block_id, count in top_blocks if count > 0]
        # Always show top 3 dangerous blocks in sidebar for user
        with st.sidebar:
            st.subheader("Top 3 Dangerous Blocks")
            if top_blocks:
                st.write(f"Top 3 dangerous blocks: {top_blocks}")
            else:
                st.info("No dangerous blocks detected during this session.")

        if top_blocks:
            print(f"[USER SMS DEBUG] Attempting to send SMS. Username: {st.session_state.username}, Top blocks: {top_blocks}")
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT phone FROM users WHERE username=? AND role='user'", (st.session_state.username,))
                user_row = cursor.fetchone()
                print(f"[USER SMS DEBUG] user_row from DB: {user_row}")
                if user_row:
                    user_phone = user_row[0]
                    print(f"[USER SMS DEBUG] Found user_phone: {user_phone}")
                    try:
                        notify_user_summary(user_phone, top_blocks)
                        print(f"[USER SMS] SMS sent to user {st.session_state.username} ({user_phone}) for top dangerous blocks: {top_blocks}")
                        st.success(f"SMS sent to {st.session_state.username} for top dangerous blocks: {top_blocks}")
                    except Exception as e:
                        print(f"[USER SMS ERROR] Exception during SMS send: {e}")
                        st.error(f"SMS sending failed: {e}")
                else:
                    print(f"[USER SMS] User phone number not found in users table for {st.session_state.username}")
                    st.warning("User phone number not found in users table.")

def main():
    inject_global_css()
    if 'username' not in st.session_state:
        st.session_state.username = None

    st.title(f"Welcome, {st.session_state.username} - Crowd Management Dashboard")

    with st.sidebar:
        st.header("👤 Controls")
        if st.button("Logout"):
            logout()
        st.markdown("---")
        # Only allow admin to see volunteer registration
        if st.session_state.role == "admin":
            st.subheader("Volunteer Registration")
            with st.form("volunteer_form"):
                name = st.text_input("Volunteer Name")
                phone = st.text_input("Phone Number")
                block_id = st.text_input("Block ID (e.g., 0, 1, 2, ...,8)")
                submit_volunteer = st.form_submit_button("Register Volunteer")
                if submit_volunteer:
                    if not name or not phone or not block_id:
                        st.error("All fields are required.")
                    else:
                        with sqlite3.connect(DB_FILE) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT phone FROM control_room WHERE phone = ?", (phone,))
                            if cursor.fetchone():
                                st.error(f"A volunteer with phone {phone} already exists.")
                            else:
                                cursor.execute("INSERT INTO control_room (name, phone, role, block_id) VALUES (?, ?, ?, ?)", (name, phone, 'volunteer', block_id))
                                conn.commit()
                                st.success(f"Volunteer '{name}' registered for block {block_id}!")

    with st.spinner("Loading model..."):
        model = load_yolo_model()

    if model is None:
        st.error("❌ Failed to load model. Please check your installation.")
        return

    st.success("✅ Model loaded successfully!")

    st.markdown("## 📤 Upload Video for Detection")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to start real-time people detection"
    )

    # Confidence threshold slider always visible
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        key='conf_thresh'
    )

    # Clear session/global data if a new file is uploaded
    if uploaded_file is not None:
        # Clear session and temp data on new upload
        global danger_memory
        danger_memory.clear()
        for key in ["all_centers", "block_counts", "video_stopped", "sms_status", "top_blocks"]:
            if key in st.session_state:
                del st.session_state[key]
        if os.path.exists("temp_files"):
            for f in os.listdir("temp_files"):
                try:
                    os.remove(os.path.join("temp_files", f))
                except Exception:
                    pass
        # Truncate block_counts table for all users on new video upload
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM block_counts")
            conn.commit()
        # Set a directory for temporary files
        os.makedirs("temp_files", exist_ok=True)
        tempfile.tempdir = "temp_files"

        with open(os.path.join(tempfile.tempdir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
        temp_video_path = os.path.join(tempfile.tempdir, uploaded_file.name)

        st.markdown("Video Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Filename", uploaded_file.name)
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024*1024)
            st.metric("💾 Size", f"{file_size:.1f} MB")
        with col3:
            st.metric("📹 Format", uploaded_file.name.split('.')[-1].upper())

        if st.button("Start Real-Time Detection", type="primary", use_container_width=True):
            st.markdown("### 🎬 Real-Time Detection")
            with st.spinner("🔄 Starting real-time detection..."):
                try:
                    process_video_realtime(temp_video_path, model, confidence_threshold, st.session_state.role)
                except Exception as e:
                    st.error(f"❌ Error during video processing: {e}")
                finally:
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
    else:
        st.info("Please upload a video to start.")

# Main app entry point
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    if 'view' not in st.session_state:
        st.session_state.view = 'login'

    # Title and subtitle for crowd management
    st.markdown('<h1 class="title">Crowd Management Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Real-Time Crowd Monitoring & Danger Prediction</h2>', unsafe_allow_html=True)
    st.markdown('<div class="Built-by">Built by Binary Brains</div>', unsafe_allow_html=True)

    # Show login/register at the bottom
    st.markdown('<div class="login-bottom">', unsafe_allow_html=True)
    if st.session_state.view == 'login':
        show_login_page()
    elif st.session_state.view == 'register':
        show_register_page()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    main()
