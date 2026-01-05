from flask import Flask, Response, request, jsonify
import threading
import cv2

cap=None
video_url=None
lock=threading.Lock() #to stop any livestreams simultaneously


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
    ip=data.get("ip")


    if not ip:
        return jsonify({"error":"No Ip address has been sent!"})
    
    real_ip=f"http://{ip}/video"

    with lock:
        if not cap is None:
            cap.release()
        cap=cv2.VideoCapture(real_ip)

    return jsonify({"status":"Connected!","ip":real_ip})


# now we generate frames of the videos and run the backend logic of inference and everything
def generate_frame():
    """
    this will generate all the videos and run inference from the model 
    """
    global cap
    from landingPage import run_vision_inference


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
            annotated,
            f"{label} {conf:.2f}",
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
        mimetype="multipart/x-mixed-replace;boundary=frame" #here mimetype tells its gonna be multiple frames you gotta replace everytime and use frames as the boundary

    )

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,threaded=True)



