import threading as th
import queue
import time
import math

import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import pygame as pg

def camera_worker(q):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    last_gesture = None
    is_playing = False
    is_track_started = False
    pg.mixer.init()
    pg.mixer.music.load("ALBLAK 52 - +7(952)812.mp3")
    while True:
        current_gesture = None
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        hands_states = {}
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                status = get_fingers_status(hand_landmarks)
                hands_states[hand_label] = status
                pass

        current_gesture = None
        
  
        if hands_states.get("Left") == [1, 1, 1, 0, 0] and hands_states.get("Right") == [1, 1, 1, 1, 1]:
             current_gesture = "52"

        elif hands_states.get("Left") == [1, 1, 1, 1, 1]:
             current_gesture = "LEFT_PALM"

        if current_gesture != last_gesture and current_gesture is not None:
            
            if current_gesture == "52":
                 pg.mixer.music.play() 
                 is_playing = True
                 is_track_started = True
                 print("ЖЕСТ 52: Музыка запущена с начала!")
                 
            elif current_gesture == "LEFT_PALM":
                 if not is_track_started:
                     pg.mixer.music.play()
                     is_playing = True
                     is_track_started = True
                     print("Ладонь: Музыка запущена (Первый старт)")
                 else:
                     if is_playing:
                          pg.mixer.music.pause()
                          is_playing = False
                          print("Ладонь: Музыка на паузе")
                     else:
                          pg.mixer.music.unpause()
                          is_playing = True
                          print("Ладонь: Музыка играет дальше")
        last_gesture = current_gesture
        final_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(final_rgb)
        q.put(img_pil)


def get_fingers_status(hand_landmarks):
    fingers = []
    wrist = hand_landmarks.landmark[0] 
    thumb_tip = hand_landmarks.landmark[4]
    thumb_joint = hand_landmarks.landmark[3]
    pinky_base = hand_landmarks.landmark[17]
    
    d_thumb_tip = math.hypot(thumb_tip.x - pinky_base.x, thumb_tip.y - pinky_base.y)
    d_thumb_joint = math.hypot(thumb_joint.x - pinky_base.x, thumb_joint.y - pinky_base.y)
    
    fingers.append(1 if d_thumb_tip > d_thumb_joint else 0)
    for i in range(8, 21, 4):
        tip = hand_landmarks.landmark[i]
        joint = hand_landmarks.landmark[i - 2]
        d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_joint = math.hypot(joint.x - wrist.x, joint.y - wrist.y)
        fingers.append(1 if d_tip > d_joint else 0)
        
    return fingers

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MediaPipe Coursework")
        self.geometry("700x550")
        ctk.set_appearance_mode("dark")
        self.q = queue.Queue()
        self.worker = th.Thread(target=camera_worker, args=(self.q,), daemon=True)
        self.worker.start()
        self.label = ctk.CTkLabel(self, text="Ожидание камеры...", font=("Arial", 20))
        self.label.pack(pady=50)
        self.check_queue()

    def check_queue(self):
        try:
            img_pil = self.q.get_nowait()
            ctk_img = ctk.CTkImage(light_image=img_pil, size=(640, 480))
            self.image = ctk_img
            self.label.configure(image=ctk_img, text="")
        except queue.Empty:
            pass
        self.after(10, self.check_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()
