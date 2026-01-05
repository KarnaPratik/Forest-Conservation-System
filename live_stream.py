from flask import Flask, Response, request, jsonify
import threading
import cv2
import time
import io
from inference_engine import run_vision_inference,run_audio_inference

cap = None
video_url = None
lock = threading.Lock()

app = Flask(__name__)

@app.route("/")
def home():
    return "Running!"

@app.route("/setting_camera", methods=["POST"])
def set_camera():
    global cap, video_url

    data = request.json
    ip = data.get("ip")

    if not ip:
        return jsonify({"error": "No IP address has been sent!"})
    
    real_ip = f"http://{ip}/video"

    with lock:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(real_ip)
        
        # Verify the connection
        if not cap.isOpened():
            cap = None
            return jsonify({"error": "Failed to connect to camera", "ip": real_ip}), 400

    return jsonify({"status": "Connected!", "ip": real_ip})


def generate_frame():
    """
    Generate video frames and run inference
    """
    global cap
    


    #run the loop infinitely
    while True:
        # Acquire lock, read frame, release lock immediately
        with lock:
            if cap is None or not cap.isOpened():
                frame = None
            else:
                success, frame = cap.read()
                if not success:
                    frame = None
        
        # If no frame available, sleep briefly and continue
        if frame is None:
            time.sleep(0.1)  # Prevent busy-waiting
            continue

        # Process frame outside the lock (this is important for performance)
        try:
            conf, label, annotated = run_vision_inference(FRAME_ARRAY=frame)
            cv2.putText(
                annotated,  # Draw on annotated frame, not original
                f"{label}, {conf:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()

            # Yield frame to browser
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue


@app.route("/video_feed")
def give_video():
    return Response(
        generate_frame(),
        mimetype="multipart/x-mixed-replace; boundary=frame" #here mimetype tells its gonna be multiple frames you gotta replace everytime and use frames as the boundary

    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=1234,threaded=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)