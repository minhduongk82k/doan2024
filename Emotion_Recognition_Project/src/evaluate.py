import os
import cv2

def detect_and_save_faces(frames_path, output_folder, padding=20):
 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(frames_path) if f.endswith(".jpg") or f.endswith(".png")]
    if not image_files:
        print(f"No images found in {frames_path}. Please add .jpg or .png files and try again.")
        return

    print(f"Processing images in {frames_path}...")
    for file_name in image_files:
        file_path = os.path.join(frames_path, file_name)

        try:
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            print(f"Detected {len(faces)} face(s) in {file_name}.")

            for i, (x, y, w, h) in enumerate(faces):
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, image.shape[1])
                y2 = min(y + h + padding, image.shape[0])

                face = image[y1:y2, x1:x2]
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_path, face)
                print(f"Saved face to {output_path}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print("Face detection and saving completed.")

if __name__ == "__main__":
    frames_path = "../data/frames/D02_20240220092513_frames"
    output_folder = "data/detected_faces3" 
    padding = 50 

    if not os.path.exists(frames_path):
        print(f"The frames path {frames_path} does not exist. Please create the folder and add images.")
        exit(1)

    detect_and_save_faces(frames_path, output_folder, padding)
