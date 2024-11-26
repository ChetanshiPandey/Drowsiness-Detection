
# Drowsiness Detection System

Many road accidents which lead to death are because of drowsiness while driving. Drivers who drive long hours like truck drivers, bus drivers are likely to experience this problem. It is highly risky to drive with lack of sleep and driving for long hours will be more tiresome.Due to the drowsiness of the driver causes very dangerous consequences, it is estimated that 70,000 to 80,000 injures & crashes happen worldwide in a year. Even deaths have reached 1000-2000 every year. There are many unofficial deaths which are not confirmed by drivers that it was due to their drowsiness. This takes lives of many innocent people. It is a nightmare for a lot of people who travel across world. It is very important to identify the driver drowsiness and alert the driver to prevent crash.
The goal of this research is the detection of the indication of this fatigue of the driver. The acquisition system, processing system and warning system are the three blocks that are present in the detection system. The video of the driver’s front face is captured by the acquisition system, and it is transferred to the next stage i.e., processing block. The detection is processed online and if drowsiness of driver is detected, then the warning system gives a warning or alarm. The methods to detect the drowsiness of the drive may be done using intrusive or nonintrusive method i.e., with and without the use of sensors connected to the driver. The cost of the system depends on the sensors used in the system. Addition of more parameters can increase the accuracy of the system to some extent. The motivation for the development of cost effective, and real-time driver drowsiness system with acceptable accuracy are the motivational factors of this work. Hence, the proposed system detects the fatigue of the driver from the facial images,and image processing technology and machine learning method are used to achieve a cost effective and portable system.


## Project Description 
1. Arrays/Lists (NumPy Arrays)
Eye Landmarks & Lip Landmarks: Arrays from the facial landmarks predictor are used to represent specific coordinates of the face (e.g., eyes, lips).

2. NumPy is used to manipulate the coordinates for calculating the Eye Aspect Ratio (EAR) and the Mouth Open Ratio (MOR).

3. Tuples are used to store (x, y) coordinates of facial landmarks, such as the eyes and lips, retrieved from dlib’s 68-point facial landmark predictor.

4. Video Stream Data (Frames)
The frames captured from the webcam act as the primary data source, processed in real-time using OpenCV to detect facial features.
The frame-by-frame processing allows the detection of eyes and lips for further analysis.

5. Arguments (ArgumentParser)
Argparse is used to manage input arguments, allowing for flexibility in selecting the webcam index and providing the path to the alert sound file.

## Project Features
1. The features of this project are:
Webcam Integration: The system integrates with the webcam to capture video frames and analyse them for signs of drowsiness. It allows selecting different webcams via command-line arguments.

2. Real-time Drowsiness Monitoring: The system continuously monitors the user's face in real-time via a webcam, analysing facial landmarks to detect signs of drowsiness[9], such as drooping eyes and yawning.

3. Use of Haar Cascade Classifier for Face Detection: For face detection, the system uses a Haar Cascade Classifier from OpenCV, which provides a balance between speed and accuracy in detecting faces before applying further landmark detection.

4. Facial Landmark Detection: The system uses dlib's 68 facial landmarks to map key points on the face, such as the eyes and mouth, enabling precise tracking of eye closure and yawning.

5. Eye Aspect Ratio (EAR) Calculation: It calculates the Eye Aspect Ratio (EAR) based on specific facial landmarks around the eyes. If the EAR falls below a certain threshold for consecutive frames, it signals eye closure, indicating drowsiness.

6. Mouth Open Ratio (MOR[5]: The system measures lip distance using facial landmarks to detect yawning, which is another indicator of drowsiness. When the mouth open ratio (MOR) exceeds a predefined threshold, the system registers a yawn.

7. Alert Mechanism: If drowsiness is detected (either via eye closure or yawning), the system triggers an alarm sound to alert the user. This uses the play sound module to play an audio file specified in the code.

8. Visual Feedback: The system provides visual feedback by drawing contours around the eyes and mouth on the video feed to indicate the areas being monitored.

## Implementation

1. Hardware Constraints:  Webcam Quality.

2. Software Constraints: Libraries and Dependencies, Limited to 
3. Predefined Facial Features, Threshold Sensitivity Environmental Constraints: Lighting Conditions, Camera Positioning.
4. Real-time Processing Constraints: Frame Processing Speed, Thread Management.
5. Implementation Constraints: Alarm Path Dependency, File Size and Execution Time.

## Design Diagram

The Figure
outlines the flow of operations in a Drowsiness Detection System using computer vision and machine learning techniques The process begins by capturing live video from a camera, where the resolution of the video is a key input. OpenCV, a widely used computer vision library, is used to extract frames from the continuous video stream. Each frame is then passed through a face detection algorithm using Haar Cascades, which is effective in detecting human faces. This is a classical object detection technique based on machine learning. Once a face is detected, the system uses dlib's machine learning algorithms to mark 68 key facial landmarks on the face, including the eyes, mouth, and head position. Then EAR, MOR, NLR Calculation is done. These values (EAR, MOR, and NLR) are compared to predefined thresholds to determine whether the driver is showing signs of fatigue. Thresholds are used to decide if a particular value (e.g., eye closure or yawning) crosses a limit indicative of drowsiness. In parallel with checking thresholds, the system monitors for yawning to further reinforce the signs of drowsiness. If the Eye Aspect Ratio (EAR) falls below the threshold and yawning or head bending is detected, the system concludes the driver is drowsy and, If drowsiness is detected, the system triggers an alert, typically an audio warning, to wake the driver and prevent a possible accident.

![alt text](http://url/to/img.png)



## System Requirement

User Interface

1.Live Video Feed: 
The primary UI component is a live video feed that captures the driver's face in real-time using a connected webcam. This stream shows the user's face and any detected facial landmarks.

2.Drowsiness Alerts: Visual cues on the interface, such as a flashing message (e.g., "Drowsiness Alert" or "Yawn Detected"), indicate when the system detects signs of drowsiness.

3.Metrics Display: The Eye Aspect Ratio (EAR) and Yawn Detection values are shown on the interface, allowing the user to see real-time data related to eye closure and yawning frequency.

4.Audio Alert Trigger: When drowsiness is detected, an audio alert is activated (e.g., an alarm sound), providing auditory feedback through the system’s speakers.

## ML Model Used

The ML algorithms used in this project are:

1. OpenCV:- It’s a video capture protocol through which the system access the webcam and continuously process frames of the live video feed.

2. Haar cascade ML Algorithm:- An Object Detection Algorithm used to identify faces in an image or a real time video. The algorithm uses edge or line detection features proposed by Paul Viola and Michael Jones in 2001, and it's popular for its real-time performance in tasks like face detection.

3. Dlib machine learning library:- dlib’s pre-trained model (shape predictor) uses a trained dataset to detect and track facial landmarks for eyes and mouth in each frame. The 68-point facial landmark model used for detecting eyes and lips is based on the work of Kazemi and Sullivan (2014) in their paper "One Millisecond Face Alignment with an Ensemble of Regression Trees". This model is integrated into the dlib library for robust real-time face landmark detection.
4. Eye Aspect Ratio (EAR):- The eye aspect ratio was introduced by Tereza Soukup ova and Jan Cech in their paper titled "Real-Time Eye Blink Detection using Facial Landmarks" (2016). In this work, they proposed using the EAR as a reliable measure for blink detection, which can be extended to detect drowsiness. The EAR formula used in my code comes directly from their method, where they set the EAR threshold at 0.3 based on experimental data.
5. Mouth Open Ratio (MOR):- MOR is widely used in research for yawning detection. One such paper is "Yawning Detection for Monitoring Driver’s Drowsiness Based on Two Visual Features" (2015), which analyses lip distances and yawning behaviours.
