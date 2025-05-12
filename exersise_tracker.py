from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)

# Global variables for exercise tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = None
tracking_active = False
current_exercise = 'bicep'
exercise_count = 0
state = False
stabilization_counter = 0

exercises = {
    'bicep': {
        'points': [mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.LEFT_ELBOW,
                  mp_pose.PoseLandmark.LEFT_WRIST],
        'angles': {'min': 60, 'max': 150},
        'name': 'Bicep Curls'
    },
    'situp': {
        'points': [mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.LEFT_HIP,
                  mp_pose.PoseLandmark.LEFT_KNEE],
        'angles': {'min': 80, 'max': 160},
        'name': 'Sit-ups'
    }
}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle < 180 else 360 - angle

def generate_frames():
    global tracking_active, current_exercise, exercise_count, state, stabilization_counter
    while True:
        if cap and cap.isOpened() and tracking_active:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                exercise = exercises[current_exercise]
                
                try:
                    a = [landmarks[exercise['points'][0]].x, landmarks[exercise['points'][0]].y]
                    b = [landmarks[exercise['points'][1]].x, landmarks[exercise['points'][1]].y]
                    c = [landmarks[exercise['points'][2]].x, landmarks[exercise['points'][2]].y]
                    
                    angle = calculate_angle(a, b, c)
                    
                    if angle < exercise['angles']['min']:
                        if not state and stabilization_counter >= 5:
                            state = True
                            stabilization_counter = 0
                        stabilization_counter += 1
                    elif angle > exercise['angles']['max']:
                        if state and stabilization_counter >= 5:
                            exercise_count += 1
                            state = False
                            stabilization_counter = 0
                        stabilization_counter += 1

                    # Draw overlay
                    cv2.putText(frame, f"Count: {exercise_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Angle: {int(angle)}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, exercise['name'], (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error: {str(e)}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Exercise Tracker</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            .container { margin: 20px; }
            button { 
                padding: 15px 30px; 
                margin: 10px; 
                font-size: 16px; 
                cursor: pointer; 
                background: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 5px; 
            }
            button:hover { background: #45a049; }
            #videoFeed { 
                margin: 20px; 
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Exercise Tracker</h1>
            <img id="videoFeed" src="{{ url_for('video_feed') }}">
            <div>
                <button onclick="toggleTracking()" id="startBtn">Start Tracking</button>
                <button onclick="switchExercise()">Switch Exercise</button>
                <button onclick="resetCounter()">Reset Counter</button>
            </div>
        </div>
        <script>
            let isTracking = false;
            
            function toggleTracking() {
                isTracking = !isTracking;
                const btn = document.getElementById('startBtn');
                btn.textContent = isTracking ? 'Stop Tracking' : 'Start Tracking';
                fetch(`/toggle_tracking?active=${isTracking}`);
            }
            
            function switchExercise() {
                fetch('/switch_exercise');
            }
            
            function resetCounter() {
                fetch('/reset_counter');
            }
            
            // Auto-reconnect to video stream
            const videoFeed = document.getElementById('videoFeed');
            function reloadVideo() {
                videoFeed.src = "{{ url_for('video_feed') }}?t=" + new Date().getTime();
            }
            setInterval(reloadVideo, 5000);
        </script>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_tracking')
def toggle_tracking():
    global tracking_active, cap
    tracking_active = not tracking_active
    if tracking_active and not cap:
        cap = cv2.VideoCapture(0)
    elif not tracking_active and cap:
        cap.release()
        cap = None
    return '', 204

@app.route('/switch_exercise')
def switch_exercise():
    global current_exercise, exercise_count, state, stabilization_counter
    current_exercise = 'situp' if current_exercise == 'bicep' else 'bicep'
    exercise_count = 0
    state = False
    stabilization_counter = 0
    return '', 204

@app.route('/reset_counter')
def reset_counter():
    global exercise_count
    exercise_count = 0
    return '', 204

def run_app():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    run_app()