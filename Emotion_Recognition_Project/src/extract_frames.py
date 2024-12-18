import cv2
import os

def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Couldn't open video file {video_path}")
        return

    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Extracted {count} frames from {video_path}")

input_video_path = 'data/videos/D02_20240220092513.mp4'
output_folder = 'data/frames/D02_20240220092513_frames'

video_to_frames(input_video_path, output_folder)