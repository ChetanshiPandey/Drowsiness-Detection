import os

audio_path = r"C:\Users\chetanshi\OneDrive\Desktop\Drowsiness\Alert.wav"

if os.path.exists(audio_path):
    print("Audio file found!")
else:
    print("Audio file not found.")
