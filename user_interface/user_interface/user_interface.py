#!/usr/bin/env python
import rclpy 
from rclpy.node import Node
import sys
from flask import Flask, render_template, request
sys.path.append('/home/harshini/capstone_venv/lib/python3.10/site-packages')
from flask_socketio import SocketIO
import base64
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import speech_recognition as sr
import threading
from cv_bridge import CvBridge

# Setup flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class UI(Node):
    def __init__(self):
        super().__init__("ui")
        self.publisher = self.create_publisher(
            String, 'user_prompt', 10)
        self.subscription = self.create_subscription(
            String, 'llm_response', self.display_chat_response, 10)
        self.subscription_img = self.create_subscription(
            Image, 'camera_feed', self.display_feed, 10)

        self.bridge = CvBridge()
        self.recognizer = sr.Recognizer()
        self.listening = True  # Toggle for speech recognition

    def display_chat_response(self, msg):
        self.get_logger().info(f"LLM response: {msg.data}")
        socketio.emit('chat_response', {'response': msg.data})

    def display_feed(self, msg):
        try:
            # self.get_logger().info(f"Camera feed: {msg}")
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if frame is None or frame.size == 0:
                self.get_logger().error("Received an empty frame!")
                return
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode()
            socketio.emit('live_feed', {'image': img_base64})
        except Exception as e:
            self.get_logger().error(f"Error processing camera feed: {e}")

    def recognize_speech(self):
        with sr.Microphone() as source:
            self.get_logger().info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            self.get_logger().info("Listening...")
            socketio.emit('speech_status', {'status': 'Listening...'})

            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    socketio.emit("speech_status", {"status": "Processing..."})

                    try:
                        live_text = self.recognizer.recognize_google(audio)
                        socketio.emit("speech_result", {"text": live_text})
                        self.publisher.publish(String(data=live_text))
                    except sr.UnknownValueError:
                        socketio.emit("speech_result", {"text": "Could not understand."})
                    except sr.RequestError:
                        socketio.emit("speech_result", {"text": "Speech recognition service unavailable."})

                except Exception as e:
                    socketio.emit("speech_result", {"text": f"Error: {str(e)}"})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_prompt')
def handle_user_input(data):
    ui_node.get_logger().info(f"Received user input: {data['message']}")
    msg = String()
    msg.data = data['message']
    ui_node.publisher.publish(msg)

@socketio.on('start_speech_recognition')
def handle_speech_request():
    if ui_node.listening:
        ui_node.listening = False
        socketio.emit('speech_status', {'status': 'Stopped Listening'})
        ui_node.get_logger().info("Speech recognition stopped.")
    else:
        ui_node.listening = True
        socketio.emit('speech_status', {'status': 'Listening...'})
        threading.Thread(target=ui_node.recognize_speech, daemon=True).start()

def run_flask():
    """Run Flask in a separate thread."""
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main(args=None):
    global ui_node
    rclpy.init(args=args)
    ui_node = UI()

    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Keep ROS spinning independently
    rclpy.spin(ui_node)

    ui_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
