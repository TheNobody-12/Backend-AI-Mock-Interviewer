from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from pypdf import PdfReader
import re
import chromadb
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import tempfile
import io
import base64
from google.cloud import speech, vision
from pydub import AudioSegment
from kafka import KafkaProducer, KafkaConsumer
import threading
import time
import traceback
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import cv2
from PIL import Image
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from scipy.spatial import distance
import queue
import logging 
import socket


load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///site.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["X-Custom-Header"],
        "supports_credentials": True,  # If using cookies/auth
        "max_age": 86400  # Cache preflight for 24 hours
    },
    r"/generate_questions": {
        "origins": ["http://localhost:3000"],  # Your React app's origin
        "methods": ["POST", "OPTIONS"],  # Allowed methods
        "allow_headers": ["Content-Type"]
    },
    r"/*": {  # Apply to all routes
        "origins": ["http://localhost:3000"],  # Your React app
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})


socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")  # Changed to threading mode

# Initialize Face Analysis components
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)


# Emotion weights for engagement score calculation
emotion_weights = {
    "happy": 1.0, "surprise": 0.8, "neutral": 0.5,
    "angry": 0.2, "sad": 0.1, "fear": 0.3, "disgust": 0.2
}

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 199
LEFT_EAR = 234
RIGHT_EAR = 454

# Frame processing queue
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

class FaceAnalyzer:
    def __init__(self):
        self.prev_face_center = None
        self.blink_counter = 0
        self.start_time = time.time()
        self.engagement_history = []
        self.last_emotion = "neutral"
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame

    def eye_aspect_ratio(self, landmarks, eye_points):
        A = distance.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
        B = distance.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
        C = distance.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
        return (A + B) / (2.0 * C)

    def calculate_head_tilt(self, landmarks):
        nose = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        left_ear = landmarks[LEFT_EAR]
        right_ear = landmarks[RIGHT_EAR]
        vertical_angle = abs(nose[1] - chin[1]) / abs(left_ear[0] - right_ear[0] + 1e-6)
        return -0.5 if vertical_angle < 0.8 else 0.0

    def calculate_engagement(self, emotion, blinks_per_sec, head_movement, head_tilt_score):
        emotion_score = emotion_weights.get(emotion, 0)
        blink_score = min(blinks_per_sec / 5, 1.0)
        movement_score = max(1.0 - min(head_movement / 50, 1.0), 0.1)
        final_score = (0.5 * emotion_score) + (0.2 * blink_score) + (0.3 * movement_score) + (0.2 * head_tilt_score)
        return max(0.0, min(final_score, 1.0))

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            dominant_emotion = self.last_emotion
            head_movement = 0
            head_tilt_score = 0
            ear = 0.3  # Default
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = {i: (lm.x * frame.shape[1], lm.y * frame.shape[0]) 
                               for i, lm in enumerate(face_landmarks.landmark)}
                    
                    leftEAR = self.eye_aspect_ratio(landmarks, LEFT_EYE)
                    rightEAR = self.eye_aspect_ratio(landmarks, RIGHT_EYE)
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    if ear < 0.25:
                        self.blink_counter += 1
                    
                    face_center = np.mean([landmarks[i] for i in range(468)], axis=0)
                    if self.prev_face_center is not None:
                        head_movement = np.linalg.norm(np.array(face_center) - np.array(self.prev_face_center))
                    self.prev_face_center = face_center
                    head_tilt_score = self.calculate_head_tilt(landmarks)
            
            elapsed_time = time.time() - self.start_time
            blinks_per_sec = self.blink_counter / elapsed_time if elapsed_time > 0 else 0
            
            if elapsed_time > 10:
                self.blink_counter = 0
                self.start_time = time.time()
            
            if self.frame_count % 15 == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    self.last_emotion = result[0]['dominant_emotion']
                except Exception as e:
                    logger.error(f"Emotion detection error: {str(e)}")
                    self.last_emotion = "neutral"
            
            engagement_score = self.calculate_engagement(
                self.last_emotion,
                blinks_per_sec,
                head_movement,
                head_tilt_score
            )
            
            self.engagement_history.append(engagement_score)
            if len(self.engagement_history) > 5:
                self.engagement_history.pop(0)
            
            return {
                "status": "success",
                "engagement_score": np.mean(self.engagement_history[-5:]) if self.engagement_history else 0,
                "emotion": self.last_emotion,
                "positivity_score": self.calculate_positivity(self.last_emotion, engagement_score)
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def calculate_positivity(self, emotion, engagement_score):
        emotion_weights = {
            "happy": 1.0, "surprise": 0.8, "neutral": 0.8,
            "sad": 0.3, "angry": 0.1, "fear": 0.2, "disgust": -0.1
        }
        emotion_score = emotion_weights.get(emotion, 0.5)
        return (0.6 * emotion_score) + (0.4 * engagement_score)

# Initialize analyzer
analyzer = FaceAnalyzer()

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

    def __repr__(self):
        return f'<User {self.email}>'


# Create tables
with app.app_context():
    db.create_all()

# Auth Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token.split()[1], app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def process_frames():
    """Worker thread for processing frames"""
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
                
            result = analyzer.process_frame(frame)
            if result:
                result_queue.put(result)
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Processing thread error: {str(e)}")

# Configure upload folder
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Google services
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("learnlm-1.5-pro-experimental")

# Initialize Kafka
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Global variables for live interview
latest_transcript = ""
candidate_answer = ""
latest_positivity_score = 0.5
current_question_index = 0
interview_questions = []

class GeminiEmbeddingFunction:
    def __call__(self, input):
        return genai.embed_content(
            model="models/embedding-001",
            content=input,
            task_type="retrieval_document",
            title="Custom query"
        )["embedding"]
    
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
def scrape_company_info(url, timeout=20, max_retries=2):
    """
    Scrape company information from a website with:
    - Better timeout handling
    - Retry mechanism
    - Improved content extraction
    - Comprehensive error handling
    """
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    # Validate and normalize URL
    if not is_valid_url(url):
        return "Invalid URL format"
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Session with retry strategy
    session = requests.Session()
    retry_adapter = requests.adapters.HTTPAdapter(
        max_retries=max_retries,
        pool_connections=1,
        pool_maxsize=1
    )
    session.mount('https://', retry_adapter)
    session.mount('http://', retry_adapter)

    try:
        # Set socket timeout
        socket.setdefaulttimeout(timeout)
        
        response = session.get(
            url,
            headers=headers,
            timeout=(timeout, timeout),  # Connect and read timeout
            verify=True,
            allow_redirects=True
        )
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return f"Non-HTML content detected: {content_type}"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        unwanted = ['script', 'style', 'nav', 'footer', 'iframe', 'noscript',
                  'svg', 'img', 'button', 'form', 'input', 'select',
                  'header', 'aside', 'meta', 'link']
        for tag in unwanted:
            for element in soup.find_all(tag):
                element.decompose()

        # Try to find main content first
        main_content = None
        content_selectors = [
            'main', 'article', 'div[role="main"]',
            '#content', '#main', '.main-content',
            'body'  # Fallback
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Clean text
        text = ' '.join(text.split())
        return text[:15000]  # Return first 15,000 characters

    except requests.exceptions.Timeout:
        print(f"Timeout occurred while accessing {url}")
        return "Request timeout - server took too long to respond"
    except requests.exceptions.SSLError:
        print(f"SSL error occurred with {url}")
        return "SSL verification failed"
    except requests.exceptions.TooManyRedirects:
        print(f"Too many redirects for {url}")
        return "Excessive redirects detected"
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {str(e)}")
        return f"Request error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error processing {url}: {str(e)}")
        return "Processing error occurred"
    finally:
        session.close()

def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return re.split('\n \n', text)

def create_chroma_db(documents, path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    try:
        chroma_client.delete_collection(name)
    except: pass
    
    db = chroma_client.create_collection(
        name=name,
        embedding_function=GeminiEmbeddingFunction()
    )
    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))
    return db

def generate_questions_with_rag(db, form_data):
    try:
        # Step 1: Retrieve relevant context
        query = f"{form_data['jobRole']} {form_data['questionType']} interview questions"
        relevant_text = db.query(query_texts=[query], n_results=5)['documents'][0]
        
        # Step 2: Generate questions with chain of thought
        prompt = f"""
You are an AI specialized in generating structured interview questions. Your task is to generate **exactly 10** questions for a **{form_data['experienceLevel']} {form_data['jobRole']}** position at **{form_data['company']}**.

### **Input Information:**
- **Job Role:** {form_data['jobRole']}
- **Company:** {form_data['company']}
- **Experience Level:** {form_data['experienceLevel']}
- **Question Type:** {form_data['questionType']} (e.g., Technical, Behavioral)
- **Job Description:** {form_data['jobDescription']}
- **Candidate Resume Highlights:** {" ".join(relevant_text)}

### **Instructions:**
1. **Analyze the Job Description & Resume**: Identify key skills, requirements, and responsibilities.
2. **Generate Questions**: Create exactly **10** questions that:
   - Match the **{form_data['questionType']}** category.
   - Are tailored to the job role and experience level.
   - Include a mix of conceptual, problem-solving, and scenario-based questions.

### **Output Format:**
Return only **valid JSON** in the following format, with **exactly** 10 questions:
```json
{{
  "questions": [
    {{"id": 1, "question": "First {form_data['questionType'].lower()} question"}},
    {{"id": 2, "question": "Second {form_data['questionType'].lower()} question"}},
    {{"id": 3, "question": "Third {form_data['questionType'].lower()} question"}},
    {{"id": 4, "question": "Fourth {form_data['questionType'].lower()} question"}},
    {{"id": 5, "question": "Fifth {form_data['questionType'].lower()} question"}},
    {{"id": 6, "question": "Sixth {form_data['questionType'].lower()} question"}},
    {{"id": 7, "question": "Seventh {form_data['questionType'].lower()} question"}},
    {{"id": 8, "question": "Eighth {form_data['questionType'].lower()} question"}},
    {{"id": 9, "question": "Ninth {form_data['questionType'].lower()} question"}},
    {{"id": 10, "question": "Tenth {form_data['questionType'].lower()} question"}}
  ]
}}
        """

        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"Error in question generation: {e}")
        raise


# Kafka Consumer Function
def kafka_consumer():
    consumer = KafkaConsumer(
        'interview_updates',
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
    for message in consumer:
        data = message.value
        if 'transcript' in data:
            socketio.emit('update', {'transcript': data['transcript']})
        elif 'positivity_score' in data:
            socketio.emit('update', {'positivity_score': data['positivity_score']})

# Start Kafka consumer in a separate thread
kafka_thread = threading.Thread(target=kafka_consumer)
kafka_thread.daemon = True
kafka_thread.start()

# Audio/Video Processing Functions
def convert_audio(audio_bytes):
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    if audio.channels > 1:
        audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Standard sample rate for speech recognition
    audio = audio.set_sample_width(2)  # 16-bit samples
    audio = audio.high_pass_filter(80)  # Remove low-frequency noise
    audio = audio.low_pass_filter(8000)  # Remove high-frequency noise
    buffer = io.BytesIO()
    audio.export(buffer, format="wav", parameters=[
        "-ac", "1",          # Mono channel
        "-ar", "16000",      # Sample rate
        "-acodec", "pcm_s16le"  # 16-bit PCM
    ])
    return buffer.getvalue()

def process_audio(audio_bytes):
    global latest_transcript, candidate_answer
    try:
        mono_audio = convert_audio(audio_bytes)
        
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=mono_audio)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="latest_long",
            use_enhanced=True,
            audio_channel_count=1,
            enable_word_confidence=True,
            enable_word_time_offsets=True
        )
        
        # Increase timeout for larger chunks
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=15)  # Increased from 10 to 15 seconds
        
        if response.results:
            transcript_piece = " ".join([result.alternatives[0].transcript 
                                      for result in response.results])
            candidate_answer += " " + transcript_piece
            producer.send('interview_updates', {
                'transcript': transcript_piece,
                'positivity_score': latest_positivity_score,
                'is_final': True
            })
            
    except Exception as e:
        producer.send('interview_errors', {
            'error': str(e),
            'type': 'audio_processing'
        })


def preprocess_audio(audio_bytes):
    """Improved audio preprocessing without equalize"""
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    
    # Normalize volume
    audio = audio.normalize()
    
    # Apply noise reduction
    audio = audio.low_pass_filter(8000).high_pass_filter(80)
    
    # Simple volume boost instead of equalization
    audio = audio + 3  # Boost volume by 3dB
    
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

#  Routes
@app.route('/generate_questions', methods=['POST'])
def generate_questions_endpoint():
    global interview_questions, current_question_index
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400
        
    file = request.files['resume']
    if not file or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid file type - only PDF accepted"}), 400

    try:
        # Save resume
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get form data
        form_data = {
            'jobRole': request.form.get('jobRole', '').strip(),
            'company': request.form.get('company', '').strip(),
            'jobDescription': request.form.get('jobDescription', '').strip(),
            'questionType': request.form.get('questionType', 'Technical').strip(),
            'experienceLevel': request.form.get('experienceLevel', 'Mid-level').strip(),
            'companyWebsite': request.form.get('companyWebsite', '').strip()
        }
        
        # Validate required fields
        if not all([form_data['jobRole'], form_data['company'], form_data['jobDescription']]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Process resume and company info
        resume_text = process_pdf(filepath)
        company_text = scrape_company_info(form_data['companyWebsite']) if is_valid_url(form_data['companyWebsite']) else ""
        all_documents = resume_text + [company_text] if company_text else resume_text
        
        # Create vector store
        db = create_chroma_db(all_documents, app.config['UPLOAD_FOLDER'], "interview_data")
        
        # Generate questions using RAG
        questions_data = generate_questions_with_rag(db, form_data)
        
        # Validate questions
        if not questions_data.get('questions'):
            raise ValueError("No questions generated")
            
        interview_questions = questions_data['questions']
        current_question_index = 0
        
        return jsonify({
            "success": True,
            "category": questions_data.get('category', form_data['questionType']),
            "questions": interview_questions,
            "currentQuestionIndex": 0,
            "message": "Questions generated successfully"
        })
        
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Failed to parse AI response",
            "details": str(e)
        }), 500
    except ValueError as e:
        return jsonify({
            "error": "Invalid question format",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Failed to generate questions",
            "details": str(e)
        }), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/send_audio', methods=['POST'])
def handle_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    try:
        audio_bytes = request.files['audio'].read()
        processed_audio = preprocess_audio(audio_bytes)
        
        # Process in background with increased timeout
        threading.Thread(
            target=process_audio,
            args=(processed_audio,),
            daemon=True
        ).start()
        
        return jsonify({
            "status": "processing",
            "chunk_size": len(processed_audio),
            "chunk_duration": 5.0  # Indicate the expected duration in seconds
        })
    except Exception as e:
        return jsonify({
            "error": "Audio processing failed",
            "details": str(e)
        }), 500
    


# Start processing thread
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()


@app.route('/send_video', methods=['POST'])
def handle_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    try:
        image_bytes = request.files['video'].read()
        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify({"error": "File too large"}), 413
            
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        try:
            frame_queue.put(frame, block=False)
        except queue.Full:
            logger.warning("Frame queue full - dropping frame")
            return jsonify({"status": "queued"})
        
        try:
            result = result_queue.get_nowait()
            socketio.emit('update', {
                'engagement_score': result['engagement_score'],
                'emotion': result['emotion'],
                'positivity_score': result['positivity_score']
            })
            return jsonify(result)
        except queue.Empty:
            return jsonify({"status": "processing"})
            
    except Exception as e:
        logger.error(f"Video endpoint error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

    
@app.route('/analyze_response', methods=['POST'])
def analyze_response():
    """Process the candidate's response and generate feedback."""
    try:
        # Ensure correct request format
        data = request.get_json()
        if not data or 'question' not in data or 'response' not in data:
            return jsonify({
                "status": "error",
                "error": "Invalid request format. Required fields: 'question', 'response'."
            }), 400

        question = data['question'][:1000]  # Limit length for safety
        response = data['response'][:5000]  # Limit length for safety

        # Gemini API prompt
        prompt = f"""

        You are an expert interview coach analyzing a mock interview. Follow these steps:

        1. FIRST, correct this transcribed response:
        - Fix grammar/syntax errors while preserving meaning
        - Complete fragmented sentences
        - Mark unclear parts with [?]
        - Keep technical terms intact

        RAW RESPONSE: {response}
        2. THEN, provide feedback on the candidate's performance based on the following criteria against the question {question}:
        - Technical knowledge and skills
        - Communication skills
        - Overall impression

        Provide feedback in this exact JSON format:
        {{
            "concise_feedback": "Brief summary of performance",
            "technical_score": 1-5,
            "communication_score": 1-5,
            "overall_score": 1-100,
            "strengths": ["list", "of", "strengths"],
            "improvements": ["list", "of", "improvements"],
            "suggested_answer": "Example of good answer"
        }}

        IMPORTANT: Only return valid JSON, no additional text or markdown.
        """

        # Call Gemini API with timeout
        try:
            genai_response = gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.3}
            )
            
            # Clean the response
            response_text = genai_response.text.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            feedback = json.loads(response_text)
            
            # Validate response structure
            required_fields = [
                "concise_feedback", "technical_score", 
                "communication_score", "overall_score",
                "strengths", "improvements", "suggested_answer"
            ]
            
            if not all(field in feedback for field in required_fields):
                raise ValueError("AI response missing required fields")
                
            return jsonify({
                "status": "success",
                "feedback": feedback
            })

        except json.JSONDecodeError as e:
            return jsonify({
                "status": "error",
                "error": "Invalid JSON response from AI",
                "details": str(e),
                "raw_response": genai_response.text if 'genai_response' in locals() else None
            }), 500
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": "Failed to process AI response",
                "details": str(e)
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": "Server error while processing feedback",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/current_question', methods=['GET'])
def get_current_question():
    return jsonify({
        "question": interview_questions[current_question_index] if interview_questions else "",
        "index": current_question_index,
        "total": len(interview_questions)
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
        
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=hashed_password
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'Registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not check_password_hash(user.password, data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.now() + datetime.timedelta(hours=24)
    }, app.config['SECRET_KEY'])
    
    return jsonify({
        'token': token,
        'user_id': user.id,
        'name': user.name
    })

@app.route('/api/check-auth', methods=['GET'])
@token_required
def check_auth(current_user):
    return jsonify({
        'authenticated': True,
        'user_id': current_user.id,
        'name': current_user.name
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the Interview AI API!"})



if __name__ == '__main__':
    try:
        socketio.run(app, port=5000, use_reloader=False)
    finally:
        frame_queue.put(None)  # Signal processing thread to stop
        processing_thread.join()
        face_mesh.close()