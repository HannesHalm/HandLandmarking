from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import pyautogui
import math

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_hands = solutions.hands

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        indexfinger = hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]

        if (indexfinger.x > 0 and indexfinger.y > 0 and handedness[0].category_name == 'Left'):
            width = int(1920 - hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 1920)
            height = int(hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 1080)
            #cv2.circle(annotated_image,(width,height), 25, (0,0,255), -1)
            pyautogui.moveTo(width, height)

        if distance(thumb.x, indexfinger.x, thumb.y, indexfinger.y) < 0.05 and handedness[0].category_name == 'Right':
            print("click")
            pyautogui.click()

    return annotated_image

def distance(x1, x2, y1, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)