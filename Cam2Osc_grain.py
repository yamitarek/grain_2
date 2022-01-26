
# IMPORTS
from turtle import color
#from types import NoneType
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import matplotlib.pyplot as plt
from scipy import fftpack
from oscpy.client import OSCClient

# setup osc connection

OSC_HOST ="127.0.0.1" #127.0.0.1 is for same computer
OSC_PORT = 8000
OSC_CLIENT = OSCClient(OSC_HOST, OSC_PORT)

# media pipe objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

color_set = (0, 255, 0)


# get labels for hands

def get_label(index, hand, results):
    output = None

    if index == 0:
        label = results.multi_handedness[0].classification[0].label
        score = results.multi_handedness[0].classification[0].score
        text = '{} {}'.format(label, round(score, 2))
        coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                        [cam_width,cam_height]).astype(int))

        output = text, coords
        return output
    
    if index == 1:
        label = results.multi_handedness[1].classification[0].label
        score = results.multi_handedness[1].classification[0].score
        text = '{} {}'.format(label, round(score, 2))
        coords = tuple(np.multiply(
                        np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                        [cam_width,cam_height]).astype(int))

        output = text, coords
        return output
        
# draw finger position
def draw_finger_position(image, results, joint_list):

    #BUFFER Variable
    buff = np.array([0,0])
    
    thumb_pos_0 = None
    thumb_pos_1 = None
    index_pos_0 = None
    index_pos_1 = None
    pinky_pos_0 = None
    pinky_pos_1 = None

    i = 0
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        
        
        thumb_pos = None
        index_pos = None
        pinky_pos = None


        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            a_depth = (hand.landmark[joint[0]].z)*(-1)*100
            if a_depth < 0 : a_depth = 0
            if a_depth > 255: a_depth = 255

            txt_a = str(round(a[0], 2)) + ", " + str(round(a[1], 2)) + ", " + str(round(a_depth, 2))
                
            cv2.circle(image, center = tuple(np.multiply(a, [cam_width, cam_height]).astype(int)), 
                        radius = int(a_depth * 2), color = color_set, thickness = 1*int(a_depth/3))
            
            
            hand_ind = results.multi_hand_landmarks.index(hand)
            joint_ind = joint_list.index(joint)

            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/x'
            ruta = string_path.encode()
            if (buff[0] != a[0]):
                OSC_CLIENT.send_message(ruta, [float(a[0])])
            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/y'
            ruta = string_path.encode()
            if (buff[1] != a[1]):
                OSC_CLIENT.send_message(ruta, [float(a[1])])
            
            buff = a

            if joint == [4,3,2]:
                thumb_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))
            if joint == [8,7,6]:
                index_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))
            if joint == [20,19,18]:
                pinky_pos = tuple(np.multiply(a, [cam_width, cam_height]).astype(int))

            cv2.putText(image, txt_a, tuple(np.multiply(a, [cam_width, cam_height]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA)

        
        if i == 0:
            thumb_pos_0 = thumb_pos
            index_pos_0 = index_pos
            pinky_pos_0 = pinky_pos
            #print(thumb_pos_0)

        if i == 1:
            thumb_pos_1 = thumb_pos
            index_pos_1 = index_pos
            pinky_pos_1 = pinky_pos

        cv2.line(image, (thumb_pos), (index_pos), color_set, thickness = 2)
        cv2.line(image, (index_pos), (pinky_pos), color_set, thickness = 2)
        cv2.line(image, (pinky_pos), (thumb_pos), color_set, thickness = 2)

        i = i+1

    

    
    if (i == 2):
        #print("both")
        cv2.line(image, thumb_pos_0, thumb_pos_1, color_set, thickness = 2)
        cv2.line(image, index_pos_0, index_pos_1, color_set, thickness = 2)
        cv2.line(image, pinky_pos_0, pinky_pos_1, color_set, thickness = 2)

        thumb_0_x_y = np.array((thumb_pos_0[0], thumb_pos_0[1]))
        thumb_1_x_y = np.array((thumb_pos_1[0], thumb_pos_1[1]))
        index_0_x_y = np.array((index_pos_0[0], index_pos_0[1]))
        index_1_x_y = np.array((index_pos_1[0], index_pos_1[1]))
        pinky_0_x_y = np.array((pinky_pos_0[0], pinky_pos_0[1]))
        pinky_1_x_y = np.array((pinky_pos_1[0], pinky_pos_1[1]))

        dist_thumb = np.linalg.norm(thumb_0_x_y - thumb_1_x_y)
        dist_index = np.linalg.norm(index_0_x_y - index_1_x_y)
        dist_pinky = np.linalg.norm(pinky_0_x_y - pinky_1_x_y)

        middle_thumb = tuple(np.multiply(((thumb_0_x_y + thumb_1_x_y)/2), [1, 1]).astype(int))
        middle_index = tuple(np.multiply(((index_0_x_y + index_1_x_y)/2), [1, 1]).astype(int))
        middle_pinky = tuple(np.multiply(((pinky_0_x_y + pinky_1_x_y)/2), [1, 1]).astype(int))
        
        cv2.putText(image, str(dist_thumb), (middle_thumb), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "distance thumb: " + str(dist_thumb), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, str( dist_index), (middle_index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "distance index: " + str(dist_index), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, str( dist_pinky), (middle_pinky), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )
        cv2.putText(image, "distance pinkie: " + str(dist_pinky), (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_set, 2, cv2.LINE_AA )

    
               
    return image


# joint List 
joint_list = [[4,3,2], [8,7,6], [20,19,18]]

# Video Capture 
## this is where you choose your webcam. try 0, 1, etc. 
cap = cv2.VideoCapture(1)
# camera parameters
cam_width  = cap.get(3)  # float `width`
cam_height = cap.get(4)  # float `height`

#camera hardcoded for camo / iphone parameters
#cam_width = 960
#cam_height = 540

# camera parameters
print(cam_width, " ", cam_height)

with mp_hands.Hands(max_num_hands = 2, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        #print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                mp_drawing.draw_landmarks(image, hand, 
                                        mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color_set, thickness=2, circle_radius=0),
                                        )
                
                

                # Render left or right detection
                if get_label(num, hand, results):
                    
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, color_set, 2, cv2.LINE_AA)
                    
            
            # Draw angles to image from joint list
            #draw_finger_angles(image, results, joint_list)

            # Draw position to image from joint list
            draw_finger_position(image, results, joint_list)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        # Quit application by pressing 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)