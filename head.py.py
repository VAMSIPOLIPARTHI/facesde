import cv2
import os

# Load the face detection classifier from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam capture (use 0 for the default webcam, or adjust if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Variables to count faces
total_faces_entered = 0
total_faces_left = 0
previous_faces = []

# Create a folder to store face images
if not os.path.exists('detected_faces'):
    os.makedirs('detected_faces')

frame_count = 0  # Frame counter

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Count the number of faces
    current_faces = len(faces)

    # Compare current faces with previous faces to check for entering or leaving
    if current_faces > len(previous_faces):
        # Faces have entered
        total_faces_entered += current_faces - len(previous_faces)
    elif current_faces < len(previous_faces):
        # Faces have left
        total_faces_left += len(previous_faces) - current_faces

    # Draw a rectangle around each face and save the detected faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the detected face to the 'detected_faces' folder
        face_img = frame[y:y+h, x:x+w]
        cv2.imwrite(f'detected_faces/face_{frame_count}_{i}.jpg', face_img)

    # Update the previous faces list for the next frame
    previous_faces = faces

    # Display the total faces detected, entered, and left
    cv2.putText(frame, f'Faces Detected: {current_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Faces Entered: {total_faces_entered}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Faces Left: {total_faces_left}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increase frame counter
    frame_count += 1

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Display total faces that entered and left
print(f"Total Faces Entered: {total_faces_entered}")
print(f"Total Faces Left: {total_faces_left}")