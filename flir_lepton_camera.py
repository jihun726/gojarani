import queue
import cv2
import threading
import time
from flirpy.camera.lepton import Lepton
import numpy as np
import copy

class FlirLeptonCameraController(object):
    def __init__(self, device_name):

        self.thread_lock = threading.Lock()
        self.t_image_data = None

        self.is_thread_running = False
        self.is_camera_ready = False
        self.capture_thread = None
        self.count = 0
        self.device_name = device_name

        self.camera = None

    def get_device_name(self):
        return self.device_name

    def get_recent_data(self):
        with self.thread_lock:
            return self.t_image_data

    def write_current_data(self, data):
        with self.thread_lock:
            self.t_image_data = copy.deepcopy(data)

    def image_capture(self):

        self.is_camera_ready = True

        while self.is_thread_running and self.is_camera_ready:
            if self.camera is not None:

                image = self.camera.grab().astype(np.float32)
                #image = 255*(image - image.min())/(image.max()-image.min())
                #frame = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_INFERNO)

                if self.camera.status != 2096:  # 2072는 뭘까?? 스위치소리
                    continue

                if image is not None:
                    self.write_current_data(image)

                time.sleep(0.03)
                self.count += 1
            else:
                break


    def start(self):
        self.camera = Lepton()
        if self.camera is not None:
            self.capture_thread = threading.Thread(target=self.image_capture)
            self.capture_thread.daemon = False
            self.is_thread_running = True
            self.capture_thread.start()
        else:
            print('boson camera is not connected')

    def stop(self):
        if self.capture_thread is not None:
            self.is_thread_running = False
            self.capture_thread.join()
            self.capture_thread = None

        if self.camera is not None:
            self.camera.close()
            self.camera = None



if __name__=='__main__':
    controller_handle = FlirLeptonCameraController('lepton_thermal_image')
    controller_handle.start()

    while True:
        frame = controller_handle.get_recent_data()
        if frame is not None:
            image_tmp = copy.deepcopy(frame.astype(np.float32))
            img_celsius = image_tmp / 100 - 273.15

            print(img_celsius[:5,:5])
            image = img_celsius.copy()
            image = 255*(image - image.min())/(image.max()-image.min())
            scaled = cv2.applyColorMap(image.astype(np.uint8), cv2.COLORMAP_INFERNO)

            #cv2.imshow('capture image', image.astype(np.int8))
            cv2.imshow('capture image', scaled)
            cv2.waitKey(40)
