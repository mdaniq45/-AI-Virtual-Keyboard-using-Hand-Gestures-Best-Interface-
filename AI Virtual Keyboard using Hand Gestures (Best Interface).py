import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define keyboard layout with Backspace & Space
keyboard_keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "⌫"],  # ⌫ is Backspace
    ["Z", "X", "C", "V", "B", "N", "M", "SPACE"]
]

# Define key positions
key_size = 100  # Key size
key_spacing = 15  # Space between keys
offset_x, offset_y = 50, 250  # Keyboard position

keys_pos = []
for row_idx, row in enumerate(keyboard_keys):
    for col_idx, key in enumerate(row):
        x = offset_x + col_idx * (key_size + key_spacing)
        y = offset_y + row_idx * (key_size + key_spacing)
        keys_pos.append((key, x, y))

# Variable to store typed text
typed_text = ""
last_pressed_key = None
last_press_time = 0
press_cooldown = 0.5  # Prevent multiple key presses

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand Tracking
    results = hands.process(rgb_frame)
    
    # Draw text display box
    cv2.rectangle(frame, (50, 50), (900, 150), (50, 50, 50), -1)  # Dark grey box for display
    cv2.putText(frame, typed_text, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    # Draw Keyboard UI with colorful keys
    for key, x, y in keys_pos:
        if key.isalpha():
            key_color = (200, 100, 255)  # Purple for letters
        elif key == "⌫":
            key_color = (255, 50, 50)  # Red for backspace
        elif key == "SPACE":
            key_color = (255, 200, 100)  # Orange for space
        else:
            key_color = (100, 255, 100)  # Green for others

        # Draw key with rounded edges
        cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), key_color, -1, cv2.LINE_AA)
        cv2.putText(frame, key, (x + 30, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)

            # Draw fingertip tracker
            cv2.circle(frame, (index_x, index_y), 25, (0, 255, 0), -1)

            # Check if fingertip is over any key
            for key, x, y in keys_pos:
                if x < index_x < x + key_size and y < index_y < y + key_size:
                    current_time = time.time()

                    # Prevent multiple key presses
                    if key != last_pressed_key or (current_time - last_press_time > press_cooldown):
                        last_pressed_key = key
                        last_press_time = current_time

                        # Highlight pressed key
                        cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, key, (x + 30, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

                        # Handle backspace key properly
                        if key == "⌫":
                            if typed_text:
                                typed_text = typed_text[:-1]  # Remove last character
                                pyautogui.press("backspace")  # Send backspace keystroke
                        elif key == "SPACE":
                            typed_text += " "
                            pyautogui.press("space")
                        else:
                            typed_text += key
                            pyautogui.press(key.lower())

    # Display the keyboard
    cv2.imshow("AI Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
