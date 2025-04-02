import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks.python import vision
from annotate import Overlay

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

IMAGE_W = 1920
IMAGE_H = 1080

CAMERA_W = 1280
CAMERA_H = 720

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

latest_result = None
processing_frame = False

def process_frame(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, processing_frame
    latest_result = result
    processing_frame = False      
        
def stream():
    overlay = Overlay()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_frame,
        num_hands=2)

    with HandLandmarker.create_from_options(options) as landmarker:
        global latest_result, processing_frame
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue

                current_time = time.time()
                
                if not processing_frame:
                    processing_frame = True

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    landmarker.detect_async(mp_image, round(current_time * 1000))
                
                if latest_result:
                    annotated_image = overlay.draw_landmarks_on_image(frame, latest_result)
                else:
                    annotated_image = frame
                overlay.draw(annotated_image)
                #annotated_image = cv2.addWeighted(annotated_image, 0.5, overlay, 0.5, 0.0)
                cv2.imshow("Live Video", cv2.flip(annotated_image, 1))

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    stream()
