from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import pyautogui
import math
import time

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

SCREEN_W = 1920
SCREEN_H = 1080

CAMERA_W = 1280
CAMERA_H = 720

mp_hands = solutions.hands

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
COLORS = [RED, GREEN, BLUE]

class Overlay:
  draw_mode = True
  cursor_mode = False
  overlay = []
  color = (0, 0, 255)
  selected_color = 0
  last_color_change = 0

  def draw_landmarks_on_image(self, rgb_image, detection_result):
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
        pinky = hand_landmarks[mp_hands.HandLandmark.PINKY_TIP]

        if (indexfinger.x > 0 and indexfinger.y > 0 and handedness[0].category_name == 'Left'):
            if self.draw_mode:
              x_color, y_color = int(indexfinger.x * CAMERA_W), int(indexfinger.y * CAMERA_H)
              self.overlay.append((x_color, y_color, self.color))
              
                #cv2.circle(annotated_image,(x_draw, y_draw), 10, (0,0,255), -1)
            if self.cursor_mode:
              width = int(SCREEN_W - hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * SCREEN_W)
              height = int(hand_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * SCREEN_H)
              pyautogui.moveTo(width, height, pyautogui.MINIMUM_DURATION, pyautogui.easeInQuad)

        if self.distance(thumb, indexfinger) < 0.05 and handedness[0].category_name == 'Right':
            print("click")
            if self.draw_mode:
              self.overlay.clear()
            if self.cursor_mode:
              pyautogui.click()

        print(time.time())
        if self.distance(thumb, pinky) < 0.05 and handedness[0].category_name == 'Right' and time.time() - self.last_color_change > 1:
          self.selected_color += 1
          self.color = COLORS[self.selected_color % 3]
          self.last_color_change = time.time()
            
    return annotated_image

  def draw(self, annotated_image):
      for i, draw_origin in enumerate(self.overlay):
        x_draw, y_draw, color = draw_origin
        if i != 0:
          x2_draw, y2_draw, _ = self.overlay[i - 1]
          cv2.line(annotated_image, (x_draw, y_draw), (x2_draw, y2_draw), color, 20)

  def distance(self, finger1, finger2):
      return math.sqrt((finger1.x - finger2.x)**2 + (finger1.y - finger2.y)**2)