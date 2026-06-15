call env\Scripts\activate
pyinstaller --noconsole --onefile --icon=icon.ico --collect-datas mediapipe --collect-all customtkinter frontend.py
pause