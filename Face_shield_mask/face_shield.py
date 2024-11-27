import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection module
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture("/dev/video1")  # Use 0 for default webcam, or try 1 if using another device

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the face detector
with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Convert the BGR frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_frame)
        
        # If faces are detected, draw bounding boxes
        if results.detections:
            for detection in results.detections:
                # Draw bounding box around detected face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
                
                # Optionally, draw landmarks on the face
                mp_drawing.draw_detection(frame, detection)
        
        # Display the resulting frame
        cv2.imshow('Face Detection (MediaPipe)', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()