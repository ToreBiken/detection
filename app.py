import subprocess
import sys

def install_requirements():
    """
    Проверяет и устанавливает зависимости из requirements.txt.
    """
    try:
        import pip
    except ImportError:
        print("Pip не установлен. Пожалуйста, установите pip и запустите скрипт снова.")
        sys.exit(1)

    print("Проверка и установка зависимостей из requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке зависимостей: {e}")
        sys.exit(1)

# Вызываем функцию установки зависимостей
install_requirements()

import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO
import torch  # Импортируем torch для работы с YOLOv5

# Инициализация Flask
app = Flask(__name__)

# Загрузка модели YOLO
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')

# Настройка камеры
camera = cv2.VideoCapture(0)  # 0 — первая веб-камера; замените на 1, если подключена вторая

def generate_frames():
    """
    Генерирует видеопоток с аннотациями YOLO.
    """
    while True:
        success, frame = camera.read()  # Считывание кадра с камеры
        if not success:
            break

        # Инференс с использованием YOLO
        results = model(frame)
        annotated_frame = results.render()[0]  # Получаем кадр с аннотациями

        # Конвертируем изображение в формат JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Генерация фрейма для Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    """
    Главная страница сервиса.
    """
    return "YOLO Webcam API is running!"

@app.route("/video_feed")
def video_feed():
    """
    Маршрут для видеопотока.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
