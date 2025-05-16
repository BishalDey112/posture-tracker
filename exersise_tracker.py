from flask import Flask, render_template_string, Response, request, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pytz
import cv2
import mediapipe as mp
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///exercise_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class ExerciseLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exercise_type = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

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
        'angles': {'min': 40, 'max': 120},
        'name': 'Sit-ups'
    }
}

calories_per_rep = {
    'Bicep Curls': 0.3,
    'Sit-ups': 0.5
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)

def save_to_database():
    global exercise_count, current_exercise
    if exercise_count > 0:
        with app.app_context():
            new_log = ExerciseLog(
                exercise_type=exercises[current_exercise]['name'],
                count=exercise_count
            )
            db.session.add(new_log)
            db.session.commit()
            print(f"Saved {exercise_count} {exercises[current_exercise]['name']} to database")

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
    logs = ExerciseLog.query.order_by(ExerciseLog.timestamp.desc()).limit(10).all()
    local_tz = pytz.timezone("Asia/Kolkata")
    for log in logs:
        log.local_time = log.timestamp.replace(tzinfo=pytz.utc).astimezone(local_tz)

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Exercise Tracker</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            button { padding: 10px 20px; margin: 10px; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #218838; }
            #videoFeed { margin: 20px; border: 2px solid #28a745; border-radius: 5px; width: 640px; height: 480px; }
            table { margin: auto; width: 80%; border-collapse: collapse; }
            th, td { border: 1px solid #ddd; padding: 10px; }
            th { background-color: #28a745; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>AI Exercise Tracker</h1>
        <img id="videoFeed" src="{{ url_for('video_feed') }}">
        <div>
            <button onclick="toggleTracking()" id="startBtn">Start Tracking</button>
            <button onclick="switchExercise()">Switch Exercise</button>
            <button onclick="resetCounter()">Reset Counter</button>
            <a href="/calories_chart"><button>View Calories Chart</button></a>
        </div>
        <h2>Exercise History</h2>
        <table>
            <tr>
                <th>Exercise</th>
                <th>Count</th>
                <th>Local Time</th>
            </tr>
            {% for log in logs %}
            <tr>
                <td>{{ log.exercise_type }}</td>
                <td>{{ log.count }}</td>
                <td>{{ log.local_time.strftime('%Y-%m-%d %I:%M %p') }}</td>
            </tr>
            {% endfor %}
        </table>
        <script>
            let isTracking = false;
            function toggleTracking() {
                isTracking = !isTracking;
                document.getElementById('startBtn').textContent = isTracking ? 'Stop Tracking' : 'Start Tracking';
                fetch(`/toggle_tracking?active=${isTracking}`);
            }
            function switchExercise() {
                fetch('/switch_exercise').then(() => location.reload());
            }
            function resetCounter() {
                fetch('/reset_counter');
            }
        </script>
    </body>
    </html>
    ''', logs=logs)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_tracking')
def toggle_tracking():
    global tracking_active, cap
    new_state = request.args.get('active') == 'true'
    if tracking_active and not new_state:
        save_to_database()
    tracking_active = new_state
    if tracking_active and cap is None:
        cap = cv2.VideoCapture(0)
    elif not tracking_active and cap:
        cap.release()
        cap = None
    return '', 204

@app.route('/switch_exercise')
def switch_exercise():
    global current_exercise, exercise_count, state, stabilization_counter
    save_to_database()
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

@app.route('/calories_chart')
def calories_chart():
    logs = ExerciseLog.query.order_by(ExerciseLog.timestamp).all()

    # Prepare calories per day data
    calories_by_day = defaultdict(float)
    # Prepare exercise counts per day per type
    counts_by_day_type = defaultdict(lambda: defaultdict(int))

    for log in logs:
        day = log.timestamp.date()
        cal = log.count * calories_per_rep.get(log.exercise_type, 0.4)
        calories_by_day[day] += cal
        counts_by_day_type[day][log.exercise_type] += log.count

    sorted_days = sorted(calories_by_day.keys())

    # Calories for line graph
    calories = [calories_by_day[day] for day in sorted_days]

    # Prepare stacked bar data
    exercise_types = sorted({ex for counts in counts_by_day_type.values() for ex in counts.keys()})
    counts_per_exercise = {ex: [counts_by_day_type[day].get(ex, 0) for day in sorted_days] for ex in exercise_types}

    # Convert dates for matplotlib
    mpl_dates = mdates.date2num(sorted_days)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Line plot - Calories burned per day
    ax1.plot_date(mpl_dates, calories, linestyle='solid', marker='o', color='blue')
    ax1.set_title("Total Calories Burned Per Day")
    ax1.set_ylabel("Calories")
    ax1.grid(True)

    # Stacked bar chart - Exercise counts per day
    bottom = np.zeros(len(sorted_days))
    colors = plt.cm.tab20.colors  # color palette
    for idx, ex in enumerate(exercise_types):
        ax2.bar(mpl_dates, counts_per_exercise[ex], bottom=bottom, label=ex, color=colors[idx % len(colors)], width=0.8)
        bottom += np.array(counts_per_exercise[ex])

    ax2.set_title("Exercise Counts Per Day by Type")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True)

    # Format x-axis dates nicely
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return send_file(img, mimetype='image/png')

def run_app():
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    run_app()
