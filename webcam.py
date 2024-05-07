import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def detect_face_in_image(image, debug=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect human in the image
    humans = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

    if debug:
        print(f"Found {len(humans)} human(s) in the image")
        # draw bounding boxes
        for (x, y, w, h) in humans:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # write image to file
        cv2.imwrite("humans.jpg", image)

    return len(humans)


def capture_image_from_webcam(webcam_index=0, num_frames=1, debug=False):
    # Open the webcam
    cap = cv2.VideoCapture(webcam_index) # cv2.CAP_V4L2
    #cap.set(cv2.CAP_PROP_CONVERT_RGB, 1) # unneeded?

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        return

    frames = []

    for _ in range(num_frames):
        success, frame = cap.read()
        if success:
            frames.append(frame)

    if len(frames) == 0:
        print("Error: Unable to capture frame from webcam")
        return

    if num_frames > 1:
        # Average the captured frames
        averaged_frame = np.mean(frames, axis=0).astype(np.uint8)
    else:
        averaged_frame = frames[0]

    # Release the webcam
    cap.release()

    if debug:
        cv2.imwrite("webcam.jpg", averaged_frame)

    return averaged_frame

if __name__ == "__main__":
    # get webcam index from command line
    import sys
    webcam_index = 0
    if len(sys.argv) > 1:
        webcam_index = int(sys.argv[1])

    image = capture_image_from_webcam(webcam_index=webcam_index, num_frames=5, debug=True)
    if image is not None:
        print(detect_face_in_image(image, debug=True))