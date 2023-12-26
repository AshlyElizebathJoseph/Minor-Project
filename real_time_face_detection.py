import cv2
import pyttsx3

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Set the speech rate (change this value as needed)
engine.setProperty('rate', 150)  # Adjust the rate, default is 200

# Function to speak when a face is detected
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam (change the number as needed, usually 0 for built-in webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, play a sound and print a message
    if len(faces) > 0:
        print("Human has been spotted")
        speak("Human is Detected") # change speak-out text here

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()


