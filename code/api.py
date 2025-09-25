import cv2
import dlib
import math
import random
import time
import numpy as np
import base64
import uuid
import speech_recognition as sr
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import mediapipe as mp
import tempfile
import os
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

active_sessions = {}

# Speech verification challenge words
CHALLENGE_WORDS = [
    "apple is an awesome phone", 
    "tiger is famous in india", 
    "cloud computing is my passion", 
    "river flows downstream", 
    "sun rises in the east", 
    "mountains are very tall", 
    "oceans are where whales live", 
    "forest provides us oxygen"
]

def check_audio_dependencies():
    """Check if audio processing dependencies are available"""
    try:
        # Check if ffmpeg is available
        from pydub.utils import which
        if which("ffmpeg") is None:
            print("Warning: ffmpeg not found")
            return False
        return True
    except Exception as e:
        print(f"Audio dependency check failed: {e}")
        return False

def initialize_speech_recognition():
    """Initialize speech recognition with proper settings"""
    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5
        return recognizer
    except Exception as e:
        print(f"Speech recognition initialization failed: {e}")
        return None

class AdvancedLivenessDetector:
    def __init__(self):
        self.MIN_MOVE = 20
        self.challenge_phrase = random.choice(CHALLENGE_WORDS)  # 👈 keep this

        
        # Initialize multiple face detection methods for maximum reliability
        self.detector = None
        self.use_opencv_detector = False
        self.use_mediapipe_detector = False
        
        # Method 1: Try dlib
        try:
            self.detector = dlib.get_frontal_face_detector()
            print("dlib face detector initialized successfully")
        except Exception as e:
            print(f"dlib detector failed: {e}")
            self.detector = None
        
        # Method 2: OpenCV Haar Cascades
        try:
            self.cv_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.cv_detector.empty():
                raise Exception("Failed to load OpenCV face cascade")
            print("OpenCV face detector initialized successfully")
        except Exception as e:
            print(f"OpenCV detector failed: {e}")
            self.cv_detector = None
        
        # Method 3: MediaPipe Face Detection
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            print("MediaPipe face detector initialized successfully")
        except Exception as e:
            print(f"MediaPipe face detector failed: {e}")
            self.face_detection = None
        
        # Initialize MediaPipe for hand detection
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"MediaPipe hands failed: {e}")
            self.hands = None
        
        # Check audio dependencies
        if not check_audio_dependencies():  
            print("Warning: Some audio dependencies may be missing")

        self.recognizer = initialize_speech_recognition()
        if self.recognizer is None:
            print("Failed to initialize speech recognition")
        
        # Define the 3-step verification process
        self.steps = self._generate_verification_steps()
        self.current_step_index = 0
        self.last_face_center = None
        self.is_verified = False
        self.step_start_time = time.time()
        self.step_timeout = 30  # 30 seconds per step

    def _generate_verification_steps(self):
        # Step 1: Face movement
        movement_directions = ["LEFT", "RIGHT", "UP", "DOWN"]
        selected_direction = random.choice(movement_directions)
        
        # Step 2: Voice verification with random challenge phrase
        self.challenge_phrase = random.choice(CHALLENGE_WORDS)
        
        # Step 3: Hand gesture
        hand_gestures = ["thumbs_up", "peace_sign", "open_palm"]
        selected_gesture = random.choice(hand_gestures)
        
        return [
            {
                "step": 1,
                "type": "movement",
                "direction": selected_direction,
                "instruction": f"Move your face {selected_direction.lower()}",
                "completed": False
            },
            {
                "step": 2,
                "type": "voice",
                "challenge": self.challenge_phrase,
                "instruction": f"Please say: '{self.challenge_phrase}'",
                "completed": False
            },
            {
                "step": 3,
                "type": "gesture",
                "gesture": selected_gesture,
                "instruction": self._get_gesture_instruction(selected_gesture),
                "completed": False
            }
        ]

    def _get_gesture_instruction(self, gesture):
        instructions = {
            "thumbs_up": "Show thumbs up to the camera",
            "peace_sign": "Show peace sign (V sign) with your fingers",
            "open_palm": "Show your open palm to the camera"
        }
        return instructions.get(gesture, "Show your hand gesture")

    def _detect_faces_dlib(self, gray_image):
        """Detect faces using dlib with robust image handling"""
        if self.detector is None:
            return []
        
        try:
            # Ensure the image is the right format for dlib
            if gray_image.dtype != np.uint8:
                gray_image = gray_image.astype(np.uint8)
            
            # Ensure contiguous memory layout
            if not gray_image.flags['C_CONTIGUOUS']:
                gray_image = np.ascontiguousarray(gray_image, dtype=np.uint8)
            
            # Make sure it's single channel
            if len(gray_image.shape) != 2:
                if len(gray_image.shape) == 3 and gray_image.shape[2] == 1:
                    gray_image = gray_image.squeeze(axis=2)
                else:
                    raise ValueError("Invalid gray image shape")
            
            # Detect faces
            faces = self.detector(gray_image)
            print(f"dlib detected {len(faces)} faces")
            return faces
            
        except Exception as e:
            print(f"dlib detection error: {e}")
            return []

    def _detect_faces_opencv(self, gray_image):
        """Detect faces using OpenCV"""
        if self.cv_detector is None:
            return []
        
        try:
            faces = self.cv_detector.detectMultiScale(
                gray_image, 
                scaleFactor=1.1, 
                minNeighbors=4,
                minSize=(30, 30)
            )
            print(f"OpenCV detected {len(faces)} faces")
            
            # Convert to dlib-like rectangle objects for consistency
            dlib_rects = []
            for (x, y, w, h) in faces:
                class Rectangle:
                    def __init__(self, x, y, w, h):
                        self._x = x
                        self._y = y
                        self._w = w
                        self._h = h
                    
                    def left(self):
                        return self._x
                    
                    def right(self):
                        return self._x + self._w
                    
                    def top(self):
                        return self._y
                    
                    def bottom(self):
                        return self._y + self._h
                
                rect = Rectangle(x, y, w, h)
                dlib_rects.append(rect)
            
            return dlib_rects
            
        except Exception as e:
            print(f"OpenCV detection error: {e}")
            return []

    def _detect_faces_mediapipe(self, rgb_image):
        """Detect faces using MediaPipe"""
        if self.face_detection is None:
            return []
        
        try:
            results = self.face_detection.process(rgb_image)
            
            if not results.detections:
                print("MediaPipe detected 0 faces")
                return []
            
            print(f"MediaPipe detected {len(results.detections)} faces")
            
            # Convert to dlib-like rectangle objects
            dlib_rects = []
            h, w = rgb_image.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                class Rectangle:
                    def __init__(self, x, y, w, h):
                        self._x = x
                        self._y = y
                        self._w = w
                        self._h = h
                    
                    def left(self):
                        return self._x
                    
                    def right(self):
                        return self._x + self._w
                    
                    def top(self):
                        return self._y
                    
                    def bottom(self):
                        return self._y + self._h
                
                rect = Rectangle(x, y, width, height)
                dlib_rects.append(rect)
            
            return dlib_rects
            
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return []

    def _detect_faces(self, frame):
        """Detect faces using multiple methods as fallbacks"""
        faces = []
        
        # Convert to grayscale for dlib and OpenCV
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            gray = frame
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Try dlib first
        if self.detector is not None:
            faces = self._detect_faces_dlib(gray)
            if faces:
                return faces
        
        # Try OpenCV as fallback
        if self.cv_detector is not None:
            faces = self._detect_faces_opencv(gray)
            if faces:
                return faces
        
        # Try MediaPipe as final fallback
        if self.face_detection is not None:
            faces = self._detect_faces_mediapipe(rgb)
            if faces:
                return faces
        
        return []

    def _get_face_center(self, rect):
        """Get the center point of a face rectangle"""
        center_x = (rect.left() + rect.right()) / 2
        center_y = (rect.top() + rect.bottom()) / 2
        return (center_x, center_y)

    def _get_face_movement_direction(self, current_center, last_center, min_move):
        """Determine movement direction based on face center positions"""
        delta_x = current_center[0] - last_center[0]
        delta_y = current_center[1] - last_center[1]
        
        # Note: In image coordinates, Y increases downward
        if abs(delta_x) > abs(delta_y):
            if delta_x > min_move:
                return 'RIGHT'  # Face moved right in image
            elif delta_x < -min_move:
                return 'LEFT'   # Face moved left in image
        else:
            if delta_y > min_move:
                return 'DOWN'   # Face moved down in image
            elif delta_y < -min_move:
                return 'UP'     # Face moved up in image
        
        return None

    def _detect_hand_gesture(self, rgb_frame):
        """Detect hand gestures using MediaPipe"""
        if self.hands is None:
            return None
            
        try:
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
                
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                # Simple gesture recognition based on landmark positions
                return self._classify_gesture(landmarks)
        except Exception as e:
            print(f"Hand gesture detection error: {e}")
        
        return None

    def _classify_gesture(self, landmarks):
        """Simple gesture classification"""
        if len(landmarks) < 21:
            return None
            
        try:
            # Thumb tip and thumb IP
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            
            # Index finger tip and PIP
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            
            # Middle finger tip and PIP
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            
            # Check for thumbs up (thumb up, other fingers down)
            if thumb_tip[1] < thumb_ip[1] and index_tip[1] > index_pip[1]:
                return "thumbs_up"
            
            # Check for peace sign (index and middle up, others down)
            if (index_tip[1] < index_pip[1] and middle_tip[1] < middle_pip[1] and 
                landmarks[16][1] > landmarks[14][1]):  # Ring finger down
                return "peace_sign"
            
            # Check for open palm (all fingers extended)
            fingers_up = 0
            if thumb_tip[1] < thumb_ip[1]: fingers_up += 1
            if index_tip[1] < index_pip[1]: fingers_up += 1
            if middle_tip[1] < middle_pip[1]: fingers_up += 1
            if landmarks[16][1] < landmarks[14][1]: fingers_up += 1  # Ring
            if landmarks[20][1] < landmarks[18][1]: fingers_up += 1  # Pinky
            
            if fingers_up >= 4:
                return "open_palm"
        except Exception as e:
            print(f"Gesture classification error: {e}")
            
        return None

    def _convert_audio_to_wav(self, audio_bytes):
        """Convert audio bytes to proper WAV format using pydub"""
        try:
            print(f"Starting audio conversion, input size: {len(audio_bytes)} bytes")
            
            # Create temporary file for input
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name
            
            print(f"Created temp input file: {temp_input_path}")
            
            # Try multiple formats in order of likelihood
            formats_to_try = ['webm', 'ogg', 'wav', 'mp3', 'mp4', 'm4a']
            audio = None
            
            for fmt in formats_to_try:
                try:
                    print(f"Trying to load audio as {fmt}")
                    audio = AudioSegment.from_file(temp_input_path, format=fmt)
                    print(f"Successfully loaded audio as {fmt}")
                    break
                except Exception as e:
                    print(f"Failed to load as {fmt}: {e}")
                    continue
            
            if audio is None:
                print("Failed to load audio in any supported format")
                os.unlink(temp_input_path)
                return None
            
            print(f"Original audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")
            
            # Convert to optimal format for speech recognition
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Set to 16kHz
            audio = audio.set_sample_width(2)  # 16-bit samples
            audio = audio.normalize()  # Normalize audio levels
            
            print(f"Converted audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")
            
            # Export as WAV in memory
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_bytes = wav_io.getvalue()
            
            print(f"Final WAV size: {len(wav_bytes)} bytes")
            
            # Clean up temporary file
            os.unlink(temp_input_path)
            
            return wav_bytes
            
        except Exception as e:
            print(f"Audio conversion error: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up files on error
            try:
                if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                    os.unlink(temp_input_path)
            except:
                pass
            return None

    def process_frame(self, frame_bytes):
        if self.is_verified or self.current_step_index >= len(self.steps):
            return {"status": "verified", "instruction": "All verification steps completed! You are verified as a real person."}

        # Check for timeout
        if time.time() - self.step_start_time > self.step_timeout:
            return {"status": "timeout", "instruction": "Step timeout. Please try again."}

        try:
            # Decode base64 to BGR image
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Failed to decode image")

            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

            # Detect faces using multiple methods
            faces = self._detect_faces(frame)

        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                "status": "error",
                "instruction": f"Error processing image: {str(e)}",
                "current_step": self.current_step_index + 1,
                "total_steps": len(self.steps),
                "step_type": self.steps[self.current_step_index]["type"]
            }
        
        if not faces:
            self.last_face_center = None
            return {
                "status": "no_face", 
                "instruction": "No face detected! Please center your face in the camera.",
                "current_step": self.current_step_index + 1,
                "total_steps": len(self.steps),
                "step_type": self.steps[self.current_step_index]["type"]
            }

        current_step = self.steps[self.current_step_index]
        is_step_completed = False

        if current_step["type"] == "movement":
            current_center = self._get_face_center(faces[0])
            
            if self.last_face_center:
                direction = self._get_face_movement_direction(current_center, self.last_face_center, self.MIN_MOVE)
                print(f"Detected movement: {direction}, Expected: {current_step['direction']}")
                
                if direction and direction == current_step["direction"]:
                    is_step_completed = True
                    
            self.last_face_center = current_center

        elif current_step["type"] == "gesture":
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_gesture = self._detect_hand_gesture(rgb_frame)
            print(f"Detected gesture: {detected_gesture}, Expected: {current_step['gesture']}")
            
            if detected_gesture == current_step["gesture"]:
                is_step_completed = True

        if is_step_completed:
            self.steps[self.current_step_index]["completed"] = True
            self.current_step_index += 1
            self.step_start_time = time.time()  # Reset timer for next step
            
            if self.current_step_index >= len(self.steps):
                self.is_verified = True
                return {
                    "status": "verified", 
                    "instruction": "All verification steps completed! You are verified as a real person.",
                    "current_step": len(self.steps),
                    "total_steps": len(self.steps),
                    "step_type": "completed"
                }
            else:
                next_step = self.steps[self.current_step_index]
                return {
                    "status": "step_completed", 
                    "instruction": f"Step {self.current_step_index} completed! Next: {next_step['instruction']}",
                    "current_step": self.current_step_index + 1,
                    "total_steps": len(self.steps),
                    "step_type": next_step["type"],
                    "step_data": next_step
                }

        return {
            "status": "in_progress", 
            "instruction": current_step["instruction"],
            "current_step": self.current_step_index + 1,
            "total_steps": len(self.steps),
            "step_type": current_step["type"],
            "step_data": current_step
        }

    def verify_speech(self, audio_data):
        """New speech verification method using the challenge-response approach"""
        if self.current_step_index != 1 or self.steps[1]["type"] != "voice":
            return {"success": False, "message": "Voice input not expected at this step"}
        
        try:
            print("Starting speech verification...")
            print(f"Expected challenge: '{self.challenge_phrase}'")
            
            if not audio_data:
                return {"success": False, "message": "No audio data received"}
            
            # Convert base64 audio to bytes
            try:
                if audio_data.startswith('data:'):
                    audio_data = audio_data.split(',')[1]
                
                audio_bytes = base64.b64decode(audio_data)
                print(f"Decoded audio bytes length: {len(audio_bytes)}")
                
                if len(audio_bytes) < 100:
                    return {"success": False, "message": "Audio data too small, please record again"}
                    
            except Exception as e:
                print(f"Base64 decode error: {e}")
                return {"success": False, "message": "Invalid audio data format"}
            
            # Convert audio to WAV format
            wav_bytes = self._convert_audio_to_wav(audio_bytes)
            if wav_bytes is None:
                return {"success": False, "message": "Could not convert audio format"}
            
            # Use speech recognition
            wav_io = io.BytesIO(wav_bytes)
            
            with sr.AudioFile(wav_io) as source:
                audio_content = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
                
            if audio_content is None:
            # Handle the error gracefully, maybe log it and return a message to the frontend.
                return {"success": False, "message": "Could not transcribe audio. Please try again."}



            spoken_text = self.recognizer.recognize_google(audio_content).lower().strip()
            expected_text = self.challenge_phrase.lower().strip()
            
            print(f"Spoken: '{spoken_text}', Expected: '{expected_text}'")
            
            if spoken_text == expected_text:
                # Mark step as completed
                self.steps[1]["completed"] = True
                self.current_step_index += 1
                self.step_start_time = time.time()
                
                if self.current_step_index >= len(self.steps):
                    self.is_verified = True
                    return {"success": True, "message": "Speech verified! All verification completed!"}
                else:
                    next_step = self.steps[self.current_step_index]
                    return {
                        "success": True, 
                        "message": f"Speech verified! Next: {next_step['instruction']}",
                        "next_step": next_step
                    }
            else:
                return {
                    "success": False, 
                    "message": f"Expected '{self.challenge_phrase}', got '{spoken_text}'"
                }
                
        except sr.UnknownValueError:
            return {"success": False, "message": "Could not understand the audio"}
        except sr.RequestError as e:
            return {"success": False, "message": f"Speech recognition service error: {e}"}
        except Exception as e:
            print(f"Speech verification error: {e}")
            return {"success": False, "message": f"Speech verification failed: {str(e)}"}


# --- API Endpoints ---
@app.route("/start_liveness", methods=["POST"])
def start_liveness():
    session_id = str(uuid.uuid4())
    liveness_checker = AdvancedLivenessDetector()
    active_sessions[session_id] = liveness_checker
    
    first_step = liveness_checker.steps[0]
    
    response = {
        "session_id": session_id,
        "instruction": first_step["instruction"],
        "total_steps": len(liveness_checker.steps),
        "current_step": 1,
        "step_type": first_step["type"],
        "step_data": first_step,
        "challenge": liveness_checker.challenge_phrase,  # Include challenge phrase
        "all_steps": [
            {"step": step["step"], "type": step["type"], "instruction": step["instruction"]}
            for step in liveness_checker.steps
        ]
    }
    return jsonify(response)

@app.route("/process_frame", methods=["POST"])
def process_frame_api():
    data = request.json
    session_id = data.get("sessionId")
    image_data_base64 = data.get("imageData")

    if not session_id or not image_data_base64:
        return jsonify({"error": "Missing sessionId or imageData"}), 400

    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
        
    liveness_checker = active_sessions[session_id]

    try:
        # Remove data URL prefix if present
        if image_data_base64.startswith('data:'):
            image_data_base64 = image_data_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_data_base64)
    except Exception as e:
        print(f"Base64 decode error: {e}")
        return jsonify({"error": "Invalid base64 image data"}), 400
        
    result = liveness_checker.process_frame(image_bytes)

    if result["status"] == "verified":
        if session_id in active_sessions:
            del active_sessions[session_id]

    return jsonify(result)

@app.route("/verify_speech", methods=["POST"])
def verify_speech_api():
    """New endpoint for speech verification"""
    data = request.json
    session_id = data.get("sessionId")
    audio_data = data.get("audioData")

    if not session_id or not audio_data:
        return jsonify({"error": "Missing sessionId or audioData"}), 400

    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
        
    liveness_checker = active_sessions[session_id]
    result = liveness_checker.verify_speech(audio_data)

    if result.get("success") and liveness_checker.is_verified:
        if session_id in active_sessions:
            del active_sessions[session_id]

    return jsonify(result)

@app.route("/get_challenge", methods=["POST"])
def get_challenge():
    """Get the current challenge phrase for a session"""
    data = request.json
    session_id = data.get("sessionId")

    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
        
    liveness_checker = active_sessions[session_id]
    return jsonify({"challenge": liveness_checker.challenge_phrase})

@app.route("/get_step_info", methods=["POST"])
def get_step_info():
    data = request.json
    session_id = data.get("sessionId")

    if not session_id:
        return jsonify({"error": "Missing sessionId"}), 400

    if session_id not in active_sessions:
        return jsonify({"error": "Invalid or expired session ID"}), 404
        
    liveness_checker = active_sessions[session_id]
    current_step = liveness_checker.steps[liveness_checker.current_step_index] if liveness_checker.current_step_index < len(liveness_checker.steps) else None
    
    return jsonify({
        "current_step": liveness_checker.current_step_index + 1,
        "total_steps": len(liveness_checker.steps),
        "step_data": current_step,
        "is_verified": liveness_checker.is_verified,
        "challenge": liveness_checker.challenge_phrase
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Advanced liveness detection service is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
