import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import av

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon="🚗")
st.title("🚗 Live Driver Drowsiness Detection")
st.write("Click 'Start' and allow camera access. The system will monitor your Eye Aspect Ratio (EAR) and alert you if you close your eyes for too long.")

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.25 
CONSEC_FRAMES_THRESHOLD = 20 
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- HELPER FUNCTION ---
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return detector, predictor

try:
    detector, predictor = load_models()
except Exception as e:
    st.error(f"Error loading models. Make sure '{PREDICTOR_PATH}' is in the repository. Details: {e}")
    st.stop()

# --- WEBRTC VIDEO PROCESSOR ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.COUNTER = 0
        self.ALARM_ON = False
        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42

    def recv(self, frame):
        # Convert WebRTC frame to numpy array (BGR format for OpenCV)
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Resize and convert to grayscale
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Detect faces
        faces = detector(gray, 0)

        for face in faces:
            # Determine facial landmarks
            shape = predictor(gray, face)
            
            # Convert to numpy array
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)

            # 3. Extract eye coordinates
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]

            # 4. Calculate EAR
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # 5. Draw contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # 6. Check EAR against threshold
            if avgEAR < EAR_THRESHOLD:
                self.COUNTER += 1

                if self.COUNTER >= CONSEC_FRAMES_THRESHOLD:
                    self.ALARM_ON = True
                    
                    # Visual Alarm
                    cv2.putText(img, "DROWSINESS DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(img, (0,0), (640,480), (0,0,255), 5)
            else:
                self.COUNTER = 0
                self.ALARM_ON = False
                
                # Show Active status
                cv2.putText(img, "Status: Active", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display EAR value
            cv2.putText(img, f"EAR: {avgEAR:.2f}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Return the processed frame to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- START WEBRTC STREAM ---
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
)