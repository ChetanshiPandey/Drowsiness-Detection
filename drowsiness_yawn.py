#python drowsiness_yawn.py --webcam webcam_index


# Import necessary libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os
from tkinter import Tk, Label, Button, BooleanVar, Checkbutton
from tkinter.messagebox import showinfo

# Global variables for controlling detection
detection_running = False  # To track whether detection is running
detection_thread = None    # Thread for running detection in the background
alarm_status = False       # Alarm status for drowsiness detection
alarm_status2 = False      # Alarm status for yawning detection
saying = False             # To prevent overlapping alarm sounds
COUNTER = 0                # Counter for consecutive frames meeting drowsiness criteria
EYE_AR_THRESH = 0.3        # Eye aspect ratio threshold for drowsiness detection
EYE_AR_CONSEC_FRAMES = 30  # Number of consecutive frames for triggering drowsiness alert
YAWN_THRESH = 20           # Threshold for detecting a yawn

# Function to play the alarm sound
def sound_alarm(path):
    global alarm_status, alarm_status2, saying
    while alarm_status:
        print('Playing drowsiness alarm...')
        playsound.playsound(path)

    if alarm_status2:
        print('Playing yawn alarm...')
        saying = True
        playsound.playsound(path)
        saying = False

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical eye landmarks
    B = dist.euclidean(eye[2], eye[4])  # Vertical eye landmarks
    C = dist.euclidean(eye[0], eye[3])  # Horizontal eye landmark
    ear = (A + B) / (2.0 * C)           # Compute EAR
    return ear

# Function to compute the average EAR for both eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate the distance between the lips for yawning detection
def lip_distance(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Function to start the detection process
def start_detection():
    global detection_running, detection_thread
    if not detection_running:
        detection_running = True
        detection_thread = Thread(target=run_detection)
        detection_thread.daemon = True
        detection_thread.start()
        start_button.config(state="disabled")
        stop_button.config(state="normal")
        showinfo("Detection", "Drowsiness detection started!")

# Function to stop the detection process
def stop_detection():
    global detection_running
    detection_running = False
    start_button.config(state="normal")
    stop_button.config(state="disabled")
    showinfo("Detection", "Drowsiness detection stopped!")

# Function to enable or disable the alarm
def toggle_alarm():
    global alarm_enabled
    alarm_enabled = alarm_var.get()
    status = "enabled" if alarm_enabled else "disabled"
    showinfo("Alarm", f"Alarm {status}!")

# Function to perform detection logic
def run_detection():
    global COUNTER, alarm_status, alarm_status2, saying, detection_running
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)  # Allow the camera to warm up

    while detection_running:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            # Visualize eyes and lips
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # Check drowsiness conditions
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                    alarm_status = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                alarm_status = False

            # Check yawning conditions
            if distance > YAWN_THRESH:
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()
                cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                alarm_status2 = False

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()

# Argument parser for command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default=r"C:\Users\chetanshi\OneDrive\Desktop\Drowsiness\Alert.wav", help="Path to alarm .WAV file")
args = vars(ap.parse_args())

if not os.path.isfile(args["alarm"]):
    print(f"Alert audio file not found at: {args['alarm']}")
    exit(1)

# Load face detector and landmark predictor
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize Tkinter GUI
root = Tk()
root.title("Drowsiness Detection")
root.geometry("400x200")

start_button = Button(root, text="Start Detection", command=start_detection, width=20)
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Detection", command=stop_detection, width=20, state="disabled")
stop_button.pack(pady=10)

alarm_var = BooleanVar()
alarm_var.set(True)  # Default to enabled
alarm_checkbox = Checkbutton(root, text="Enable Alarm", variable=alarm_var, command=toggle_alarm)
alarm_checkbox.pack(pady=10)

root.mainloop()





