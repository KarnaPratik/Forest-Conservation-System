from flask import Flask, Response, request, jsonify
import threading
import cv2
import time
import io
from inference_engine import run_vision_inference,run_audio_inference
import requests


cap=None
video_url=None
lock=threading.Lock() #to stop any livestreams simultaneously
audio_prediction="0"
audio_label="Analysing sound..."
conf=0
label="Analysing Video"
combined_pred=0
combined_label="Analysing"

app=Flask(__name__)

@app.route("/")
def home():
    return "Running!"

@app.route("/setting_camera",methods=["POST"])
def set_camera(): #this function will be used to get the ip address after the user pushes it
    #we gonna use the global cap variable that will check if we are already reading videos from one endpoint
    global cap, video_url

    #this will receive json data which we will use to recieve the ip address
    data=request.json
    video_url=data.get("ip")


    if not video_url:
        return jsonify({"error":"No Ip address has been sent!"})
    
    real_ip=f"http://{video_url}/video"

    with lock:
        if not cap is None:
            cap.release()
        cap=cv2.VideoCapture(real_ip)

        threading.Thread(target=get_audio_inference, daemon=True).start()
    return jsonify({"status":"Connected!","ip":real_ip})

def get_audio_inference():
    global audio_prediction, audio_label,video_url


    audio_url=f"http://{video_url}/audio.wav"

    while True:
        try:
            with requests.get(audio_url,stream=True,timeout=5) as r:
                audio_buffer=io.BytesIO()
                first_time=time.time()

                for chunk in r.iter_content(chunk_size=1024):
                    audio_buffer.write(chunk)
                    if time.time()-first_time>3:
                        break

                audio_buffer.seek(0) #this makes sure when inference runs it points back to the starting
                audio_label,audio_prediction=run_audio_inference(FRAME_ARRAY=audio_buffer)

                with lock:
                    update_threat_logic()
        except:
            print("Audio_error")
            time.sleep(2)

def update_threat_logic():
    global audio_prediction, audio_label, label,conf,combined_pred,combined_label








# now we generate frames of the videos and run the backend logic of inference and everything
def generate_frame():
    """
    this will generate all the videos and run inference from the model 
    """
    global cap,conf,label,video_url
    


    #run the loop infinitely
    while True:
        with lock:
            #we skip everything if the camera is not ready and everything
            if cap is None or not cap.isOpened(): #this makes sure no streaming occurs when not needed
                continue
            success, frame=cap.read()

            if not success:
                continue #continues if it cannot extract a frame

            conf,label,annotated=run_vision_inference(FRAME_ARRAY=frame)
            cv2.putText(
            frame,
            f"{label},{conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
            

            #now after getting the image we convert the frames back into the videos or the MPEG format
            ret,buffer=cv2.imencode(".jpg",annotated)
            frame_bytes=buffer.tobytes()

            #now we yield or produce those frames in the browser till it is connected
            #yield function acts as a generator and keeps on sending images everytime
        yield(
                b"--frame\r\n" #this tells browser that previous streaming has been over where b is for specifiying its bytes r is telling to move cursor back and n is for new line
                b"Content-Type: image/jpeg\r\n\r\n"+frame_bytes+b"\r\n"


            )

#now we have a function that generates each frame and draws them but we need a function to access from frontend to run them
@app.route("/video_feed")
def give_video():
    #we connect browser here to run the function
    return Response(
        generate_frame(),
        mimetype="multipart/x-mixed-replace; boundary=frame" #here mimetype tells its gonna be multiple frames you gotta replace everytime and use frames as the boundary

    )

@app.route("/combined_pred",methods=["GET"])
def combined_pred():
    global combined_label,combined_pred
    return jsonify({
        "Combined_prediction":combined_pred,
        "Combined_label": combined_label
    })

if __name__=="__main__":
    app.run(host="0.0.0.0",port=1234,threaded=True)



