import streamlit as st
import cv2
import numpy as np
import torch  # Для работы с YOLOv5
from PIL import Image
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Загрузка модели YOLO
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github', force_reload=True)



# Функция для обработки изображений
def process_frame(frame):
    results = model(frame)
    annotated_frame = results.render()[0]  # Получаем кадр с аннотациями
    return annotated_frame


st.title("YOLO Webcam Detection with Streamlit")

# Включаем веб-камеру
camera = cv2.VideoCapture(0)

run_detection = st.button("Start Detection")

if run_detection:
    stframe = st.empty()  # Placeholder для отображения видео
    while True:
        success, frame = camera.read()  # Считывание кадра
        if not success:
            st.error("Ошибка при доступе к камере.")
            break

        # Применяем YOLO
        annotated_frame = process_frame(frame)

        # Преобразуем изображение в формат для Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_frame)

        # Отображаем результат
        stframe.image(img, use_column_width=True)
