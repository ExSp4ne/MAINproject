import threading
import queue
import time
import math
import os
from typing import List, Dict, Any, Optional

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp
import pygame as pg
import pyautogui
import numpy as np
import pystray
from pystray import MenuItem as item

# ==========================================
# КОНСТАНТЫ (Настройки системы)
# ==========================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MOUSE_DEADZONE = 3
SCROLL_DEADZONE = 15
SMOOTHING_FACTOR = 0.3
VOLUME_SMOOTHING = 0.1
SKIP_COOLDOWN = 1.0
MOUSE_PAD_X = 120
MOUSE_PAD_Y = 100

class AudioController:
    """Класс для управления воспроизведением музыки через pygame."""
    def __init__(self):
        pg.mixer.init()
        self.is_playing: bool = False
        self.is_track_started: bool = False

    def load_and_play(self, track_path: str) -> None:
        try:
            pg.mixer.music.load(track_path)
            pg.mixer.music.play()
            self.is_playing = True
            self.is_track_started = True
        except Exception as e:
            print(f"Ошибка загрузки трека: {e}")

    def pause(self) -> None:
        if self.is_playing:
            pg.mixer.music.pause()
            self.is_playing = False

    def unpause(self) -> None:
        if not self.is_playing and self.is_track_started:
            pg.mixer.music.unpause()
            self.is_playing = True

    def rewind_or_play(self) -> None:
        if self.is_playing:
            pg.mixer.music.rewind()
        else:
            pg.mixer.music.play()
            self.is_playing = True
            self.is_track_started = True

    def set_volume(self, volume: float) -> None:
        pg.mixer.music.set_volume(volume)

    def is_busy(self) -> bool:
        return pg.mixer.music.get_busy()


class HandDetector:
    """Класс для инкапсуляции логики MediaPipe."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, image_rgb: np.ndarray) -> Any:
        return self.hands.process(image_rgb)

    def draw(self, image_rgb: np.ndarray, hand_landmarks: Any) -> None:
        self.mp_draw.draw_landmarks(image_rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        self.hands.close()

    @staticmethod
    def get_fingers_status(hand_landmarks: Any) -> List[int]:
        fingers = []
        wrist = hand_landmarks.landmark[0]
        t_tip, t_joint = hand_landmarks.landmark[4], hand_landmarks.landmark[3]
        pinky_base = hand_landmarks.landmark[17]
        
        d_t_tip = math.hypot(t_tip.x - pinky_base.x, t_tip.y - pinky_base.y)
        d_t_joint = math.hypot(t_joint.x - pinky_base.x, t_joint.y - pinky_base.y)
        fingers.append(1 if d_t_tip > d_t_joint else 0)
        
        for i in [8, 12, 16, 20]:
            tip = hand_landmarks.landmark[i]
            joint = hand_landmarks.landmark[i - 2]
            dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_joint = math.hypot(joint.x - wrist.x, joint.y - wrist.y)
            fingers.append(1 if dist_tip > dist_joint else 0)
        return fingers


class CameraWorker(threading.Thread):
    """Поток для обработки видео с камеры и распознавания жестов."""
    def __init__(self, frame_queue: queue.Queue, shared_state: Dict[str, Any], state_lock: threading.Lock):
        super().__init__(daemon=True)
        self.q = frame_queue
        self.state = shared_state
        self.lock = state_lock
        
        self.audio = AudioController()
        self.detector = HandDetector()
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        self.screen_w, self.screen_h = pyautogui.size()
        
        self.last_skip_time = 0.0
        self.last_gesture: Optional[str] = None
        self.left_pinch_active = False
        self.right_pinch_active = False
        self.prev_scroll_y: Optional[float] = None

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        with self.lock:
            self.audio.set_volume(self.state["volume"])

        try:
            while True:
                with self.lock:
                    if not self.state["is_running"]:
                        break  # Безопасный выход из цикла
                    
                    update_track = self.state["update_track"]
                    gestures_enabled = self.state["gestures_enabled"]
                    current_mode = self.state["current_mode"]

                # Управление плейлистом
                if update_track:
                    self._handle_track_update()

                # Автопереключение трека
                if self.audio.is_playing and self.audio.is_track_started and not self.audio.is_busy():
                    self._handle_auto_next_track()

                success, frame = cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                # Оптимизация: конвертируем в RGB один раз и рисуем прямо на нем
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector.process(image_rgb)

                if results.multi_hand_landmarks and gestures_enabled:
                    if current_mode == "player":
                        self._process_player_mode(results, image_rgb)
                    elif current_mode == "mouse":
                        self._process_mouse_mode(results, image_rgb, w, h)
                else:
                    self._reset_gesture_states()

                # Передача кадра в UI
                img_pil = Image.fromarray(image_rgb)
                if self.q.qsize() > 1:
                    try: self.q.get_nowait()
                    except queue.Empty: pass
                self.q.put(img_pil)

        except Exception as e:
            print(f"Поток камеры остановлен из-за ошибки: {e}")
        finally:
            cap.release()
            self.detector.close()

    def _handle_track_update(self) -> None:
        with self.lock:
            playlist = self.state["playlist"]
            idx = self.state["current_index"]
            
            if playlist and len(playlist) > idx:
                track_path = playlist[idx]
                self.audio.load_and_play(track_path)
                track_name = os.path.basename(track_path)
                self.state["playback_status"] = f"▶ {track_name}"
            else:
                self.state["playback_status"] = "❌ Ошибка загрузки трека"
            self.state["update_track"] = False

    def _handle_auto_next_track(self) -> None:
        with self.lock:
            if self.state["current_index"] < len(self.state["playlist"]) - 1:
                self.state["current_index"] += 1
                self.state["update_track"] = True
            else:
                self.audio.is_playing = False
                self.state["playback_status"] = "⏹ Плейлист завершен"

    def _process_player_mode(self, results: Any, image_rgb: np.ndarray) -> None:
        hands_states = {}
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            self.detector.draw(image_rgb, hand_landmarks)
            label = handedness.classification[0].label
            status = self.detector.get_fingers_status(hand_landmarks)
            hands_states[label] = status

            # Управление громкостью
            if label == "Left" and status == [0, 1, 0, 0, 0]:
                y_pos = hand_landmarks.landmark[8].y
                new_vol = max(0.0, min(1.0, 1.0 - y_pos))
                with self.lock:
                    self.state["volume"] = self.state["volume"] + (new_vol - self.state["volume"]) * VOLUME_SMOOTHING
                    self.audio.set_volume(self.state["volume"])

        current_gesture = self._detect_player_gesture(hands_states)

        if current_gesture != self.last_gesture and current_gesture is not None:
            self._execute_player_gesture(current_gesture)
        self.last_gesture = current_gesture

    def _detect_player_gesture(self, hands_states: Dict[str, List[int]]) -> Optional[str]:
        l_stat = hands_states.get("Left")
        r_stat = hands_states.get("Right")

        if l_stat == [1, 1, 1, 0, 0] and r_stat == [1, 1, 1, 1, 1]:
            return "52"
        if l_stat == [0, 0, 0, 0, 0]:
            return "PAUSE"
        if r_stat == [1, 1, 1, 1, 1] and l_stat != [1, 1, 1, 0, 0]:
            return "UNPAUSE"
        if r_stat == [1, 0, 0, 0, 0]:
            return "NEXT"
        if l_stat == [1, 0, 0, 0, 0]:
            return "PREV"
        return None

    def _execute_player_gesture(self, gesture: str) -> None:
        with self.lock:
            playlist = self.state["playlist"]
            idx = self.state["current_index"]
            current_time = time.time()

            if not playlist:
                self.state["playback_status"] = "⚠️ Добавьте треки"
                return

            if gesture == "52":
                self.audio.rewind_or_play()
                self.state["playback_status"] = f"▶ {os.path.basename(playlist[idx])}"
            elif gesture == "PAUSE" and self.audio.is_playing:
                self.audio.pause()
                self.state["playback_status"] = "⏸ Пауза"
            elif gesture == "UNPAUSE":
                self.audio.unpause()
                self.state["playback_status"] = f"▶ {os.path.basename(playlist[idx])}"
            elif gesture == "NEXT" and (current_time - self.last_skip_time > SKIP_COOLDOWN):
                if idx < len(playlist) - 1:
                    self.state["current_index"] += 1
                    self.state["update_track"] = True
                self.last_skip_time = current_time
            elif gesture == "PREV" and (current_time - self.last_skip_time > SKIP_COOLDOWN):
                if idx > 0:
                    self.state["current_index"] -= 1
                    self.state["update_track"] = True
                else:
                    self.audio.rewind_or_play()
                self.last_skip_time = current_time

    def _process_mouse_mode(self, results: Any, image_rgb: np.ndarray, w: int, h: int) -> None:
        cv2.rectangle(image_rgb, (MOUSE_PAD_X, MOUSE_PAD_Y), (w - MOUSE_PAD_X, h - MOUSE_PAD_Y), (0, 255, 150), 2)
        left_hand_visible = False

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            self.detector.draw(image_rgb, hand_landmarks)
            label = handedness.classification[0].label
            status = self.detector.get_fingers_status(hand_landmarks)

            if label == "Right":
                self._handle_mouse_movement(hand_landmarks, w, h)
                self._handle_mouse_clicks(status)
            
            if label == "Left":
                left_hand_visible = True
                self._handle_mouse_scroll(status, hand_landmarks, h)

        if not left_hand_visible:
            self.prev_scroll_y = None

    def _handle_mouse_movement(self, hand_landmarks: Any, w: int, h: int) -> None:
        idx_x = hand_landmarks.landmark[8].x * w
        idx_y = hand_landmarks.landmark[8].y * h

        scr_x = np.interp(idx_x, (MOUSE_PAD_X, w - MOUSE_PAD_X), (0, self.screen_w))
        scr_y = np.interp(idx_y, (MOUSE_PAD_Y, h - MOUSE_PAD_Y), (0, self.screen_h))

        cur_x, cur_y = pyautogui.position()
        
        if abs(scr_x - cur_x) > MOUSE_DEADZONE or abs(scr_y - cur_y) > MOUSE_DEADZONE:
            pyautogui.moveTo(
                cur_x + (scr_x - cur_x) * SMOOTHING_FACTOR,
                cur_y + (scr_y - cur_y) * SMOOTHING_FACTOR
            )

    def _handle_mouse_clicks(self, status: List[int]) -> None:
        # Левый клик (Средний + Указательный)
        if status[2] == 1 and status[1] == 1:
            if not self.left_pinch_active:
                pyautogui.click(button='left')
                self.left_pinch_active = True
        else:
            self.left_pinch_active = False

        # Правый клик (Мизинец + Указательный)
        if status[4] == 1 and status[1] == 1:
            if not self.right_pinch_active:
                pyautogui.click(button='right')
                self.right_pinch_active = True
        else:
            self.right_pinch_active = False

    def _handle_mouse_scroll(self, status: List[int], hand_landmarks: Any, h: int) -> None:
        if status[1] == 1 and status[2] == 0 and status[3] == 0:
            current_y = hand_landmarks.landmark[8].y * h
            if self.prev_scroll_y is not None:
                delta_y = self.prev_scroll_y - current_y
                scroll_amount = int(delta_y * 2.0)
                if abs(scroll_amount) > SCROLL_DEADZONE:
                    pyautogui.scroll(scroll_amount)
                    self.prev_scroll_y = current_y
            else:
                self.prev_scroll_y = current_y
        else:
            self.prev_scroll_y = None

    def _reset_gesture_states(self) -> None:
        self.last_gesture = None
        self.left_pinch_active = False
        self.right_pinch_active = False
        self.prev_scroll_y = None


def create_tray_icon_image() -> Image.Image:
    image = Image.new('RGB', (64, 64), color='#10B981')
    dc = ImageDraw.Draw(image)
    dc.rectangle((16, 16, 48, 48), fill='#1f2937')
    return image


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI Gesture Controller 🤖")
        self.geometry("900x800")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.protocol('WM_DELETE_WINDOW', self.hide_to_tray)
        
        # Потокобезопасное состояние
        self.state_lock = threading.Lock()
        self.shared_state = {
            "is_running": True,
            "gestures_enabled": True,
            "playlist": [],          
            "current_index": 0,      
            "update_track": False, 
            "current_mode": "player",
            "playback_status": "⏹ Ожидание треков",
            "volume": 0.5            
        }

        self._setup_ui()

        self.q = queue.Queue()
        self.after(500, self.start_worker)
        self.check_queue()

    def _setup_ui(self) -> None:
        self.tabview = ctk.CTkTabview(self, corner_radius=15)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        self.tab_main = self.tabview.add("🎛 Главная")
        self.tab_settings = self.tabview.add("⚙️ Настройки")
        
        self.setup_main_tab()
        self.setup_settings_tab()

    def start_worker(self) -> None:
        self.worker = CameraWorker(self.q, self.shared_state, self.state_lock)
        self.worker.start()

    def hide_to_tray(self) -> None:
        self.withdraw()
        image = create_tray_icon_image()
        menu = (item('Развернуть', self.show_from_tray), item('Выход', self.quit_app))
        self.tray_icon = pystray.Icon("name", image, "AI Controller", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_from_tray(self, icon: Any, item: Any) -> None:
        self.tray_icon.stop()
        self.after(0, self.deiconify)

    def quit_app(self, icon: Any = None, item: Any = None) -> None:
        if hasattr(self, 'tray_icon'):
            self.tray_icon.stop()
        
        with self.state_lock:
            self.shared_state["is_running"] = False
            
        # Ждем безопасного завершения воркера (максимум 2 секунды)
        if hasattr(self, 'worker') and self.worker.is_alive():
            self.worker.join(timeout=2.0)
            
        self.destroy()

    def setup_main_tab(self) -> None:
        self.mode_switcher = ctk.CTkSegmentedButton(
            self.tab_main, 
            values=["Управление плеером", "Управление мышью"],
            command=self.change_mode,
            font=("Arial", 16, "bold"), height=45
        )
        self.mode_switcher.set("Управление плеером")
        self.mode_switcher.pack(pady=(20, 10))

        self.status_frame = ctk.CTkFrame(self.tab_main, corner_radius=15)
        self.status_frame.pack(fill="x", padx=40, pady=10)
        
        self.lbl_status_title = ctk.CTkLabel(self.status_frame, text="Текущий трек:", font=("Arial", 14))
        self.lbl_status_title.pack(pady=(15, 0))
        
        with self.state_lock:
            status_text = self.shared_state["playback_status"]
            
        self.lbl_status_value = ctk.CTkLabel(self.status_frame, text=status_text, font=("Arial", 22, "bold"), text_color="#10B981")
        self.lbl_status_value.pack(pady=(5, 5))

        self.lbl_volume = ctk.CTkLabel(self.status_frame, text="🔊 Громкость: 50%", font=("Arial", 14, "bold"), text_color="#60A5FA")
        self.lbl_volume.pack(pady=(0, 15))

        self.instr_frame = ctk.CTkFrame(self.tab_main, corner_radius=15, fg_color="transparent", border_width=2)
        self.instr_frame.pack(fill="both", expand=True, padx=40, pady=10)

        self.lbl_instr_title = ctk.CTkLabel(self.instr_frame, text="Активные жесты", font=("Arial", 20, "bold"))
        self.lbl_instr_title.pack(pady=(20, 10))

        player_text = (
            "👊 Левый кулак: ПАУЗА\n"
            "🖐 Правая ладонь: ИГРАТЬ\n"
            "🤙 Жест '52': СТАРТ\n\n"
            "👍 Правая (Класс): СЛЕД. ТРЕК\n"
            "👍 Левая (Класс): ПРЕД. ТРЕК\n\n"
            "☝️ Левая (только указательный) + Вверх/Вниз: ГРОМКОСТЬ"
        )
        self.lbl_instr = ctk.CTkLabel(
            self.instr_frame, 
            text=player_text, 
            font=("Arial", 18), justify="center"
        )
        self.lbl_instr.pack(pady=10)
     

        # === ДОБАВИТЬ ЭТОТ БЛОК ===
        self.btn_quit = ctk.CTkButton(
            self.tab_main,
            text="Выключить приложение ⏻",
            fg_color="#EF4444",       # Красный цвет для кнопки выхода
            hover_color="#B91C1C",
            font=("Arial", 16, "bold"),
            height=45,
            command=self.quit_app     # Привязываем к существующему методу безопасного выхода
        )
        self.btn_quit.pack(side="bottom", pady=(10, 20), fill="x", padx=40)

    def change_mode(self, value: str) -> None:
        with self.state_lock:
            if value == "Управление плеером":
                self.shared_state["current_mode"] = "player"
                player_text = (
                    "👊 Левый кулак: ПАУЗА\n"
                    "🖐 Правая ладонь: ИГРАТЬ\n"
                    "🤙 Жест '52': СТАРТ\n\n"
                    "👍 Правая (Класс): СЛЕД. ТРЕК\n"
                    "👍 Левая (Класс): ПРЕД. ТРЕК\n\n"
                    "☝️ Левая (только указательный) + Вверх/Вниз: ГРОМКОСТЬ"
                )
                self.lbl_instr.configure(text=player_text)
                self.status_frame.pack(fill="x", padx=40, pady=10, before=self.instr_frame)
            else:
                self.shared_state["current_mode"] = "mouse"
                mouse_text = (
                    "ПОЗИЦИЯ: Указательный палец выпрямлен, остальные согнуты.\n\n"
                    "👆 Правая рука: ДВИЖЕНИЕ КУРСОРA\n"
                    "✌️ Правая (выпрямить средний): ЛЕВЫЙ КЛИК\n"
                    "🤙 Правая (оттопырить мизинец): ПРАВЫЙ КЛИК\n\n"
                    "↕️ Левая рука (двигать вверх/вниз): ПРОКРУТКА (Скролл)"
                )
                self.lbl_instr.configure(text=mouse_text)
                self.status_frame.pack_forget()

    def setup_settings_tab(self) -> None:
        theme_frame = ctk.CTkFrame(self.tab_settings, corner_radius=10)
        theme_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        self.lbl_theme = ctk.CTkLabel(theme_frame, text="🎨 Тема оформления:", font=("Arial", 14))
        self.lbl_theme.pack(side="left", padx=15, pady=15)
        
        self.theme_switcher = ctk.CTkSegmentedButton(theme_frame, values=["Тёмная", "Светлая"], command=self.change_theme)
        self.theme_switcher.set("Тёмная")
        self.theme_switcher.pack(side="right", padx=15)

        audio_frame = ctk.CTkFrame(self.tab_settings, corner_radius=10)
        audio_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_track = ctk.CTkLabel(audio_frame, text="🎵 Треков в плейлисте: 0", font=("Arial", 14))
        self.lbl_track.pack(side="left", padx=15, pady=15)
        
        ctk.CTkButton(audio_frame, text="📂 Добавить треки", font=("Arial", 14), command=self.browse_tracks).pack(side="right", padx=15)

        cam_frame = ctk.CTkFrame(self.tab_settings, corner_radius=10)
        cam_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.sw_gest = ctk.CTkSwitch(cam_frame, text="👁 Отслеживание жестов включено", font=("Arial", 14, "bold"), command=self.toggle_gestures)
        self.sw_gest.select()
        self.sw_gest.pack(anchor="w", padx=20, pady=10)
        
        self.video_label = ctk.CTkLabel(cam_frame, text="Загрузка камеры...")
        self.video_label.pack(pady=5, expand=True)

    def change_theme(self, new_theme: str) -> None:
        if new_theme == "Тёмная":
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")

    def browse_tracks(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Audio", "*.mp3 *.wav")])
        if paths:
            with self.state_lock:
                self.shared_state["playlist"] = list(paths)
                self.shared_state["current_index"] = 0
                self.shared_state["update_track"] = True
            self.lbl_track.configure(text=f"🎵 Треков в плейлисте: {len(paths)}")

    def toggle_gestures(self) -> None:
        with self.state_lock:
            self.shared_state["gestures_enabled"] = bool(self.sw_gest.get())

    def check_queue(self) -> None:
        if not self.winfo_exists():
            return
            
        try:
            img_pil = self.q.get_nowait()
            if hasattr(self, 'tabview') and self.tabview.winfo_exists() and self.tabview.get() == "⚙️ Настройки":
                ctk_img = ctk.CTkImage(light_image=img_pil, size=(CAMERA_WIDTH, CAMERA_HEIGHT))
                if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                    self.video_label.configure(image=ctk_img, text="")
        except queue.Empty:
            pass
            
        if hasattr(self, 'lbl_status_value') and self.lbl_status_value.winfo_exists():
            with self.state_lock:
                status_text = self.shared_state["playback_status"]
                vol_percent = int(self.shared_state["volume"] * 100)
                
            if len(status_text) > 35:
                status_text = status_text[:32] + "..."
            self.lbl_status_value.configure(text=status_text)
            self.lbl_volume.configure(text=f"🔊 Громкость: {vol_percent}%")
            
        self.after(15, self.check_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()