import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import matplotlib.pyplot as plt
from scipy import fftpack
from oscpy.client import OSCClient

OSC_HOST ="127.0.0.1" #127.0.0.1 is for same computer
OSC_PORT_PD = 8000
OSC_PORT_WEK = 6448
OSC_CLIENT_PD = OSCClient(OSC_HOST, OSC_PORT_PD)
OSC_CLIENT_WEK = OSCClient(OSC_HOST, OSC_PORT_WEK)

try:
    while True:


        a = np.random.rand(1)
        string_path_pd = '/pd/x'
        ruta_pd = string_path_pd.encode()
        OSC_CLIENT_PD.send_message(ruta_pd, [float(a)])

        string_path_wek = '/wek/x'
        ruta_wek = string_path_wek.encode()
        OSC_CLIENT_WEK.send_message(ruta_wek, [float(-a)])

        time.sleep(1)

except KeyboardInterrupt:
    print("Press Ctrl-C to terminate while statement")
    pass
