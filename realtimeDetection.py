import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json

json_file = open("expressiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("expressiondetector.h5")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  
    return feature / 255.0  

task_assignment = {
    'angry': "Perform deep breathing exercises or listen to calming music.",
    'disgust': "Engage in a clear discussion, avoid negativity, and focus on achievements.",
    'fear': "Seek reassurance, talk to a mentor, or take a small break.",
    'happy': "Encourage collaboration and brainstorming with colleagues.",
    'neutral': "Continue with assigned tasks normally.",
    'sad': "Take a break, engage in social interaction, or work on a lighter task.",
    'surprise': "Channel creativity into innovative tasks or activities."
}

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        continue
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    detected_faces = []
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            gray_face = cv2.cvtColor(frame[y:y+height, x:x+width], cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (48, 48))
            img = extract_features(gray_face)
            
            pred = model.predict(img)
            detected_emotion = labels[pred.argmax()]
            assigned_task = task_assignment[detected_emotion]
            
            detected_faces.append({'emotion': detected_emotion, 'task': assigned_task})
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
