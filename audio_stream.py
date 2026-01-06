from flask import Flask, request, jsonify, Response
from inference_engine import run_audio_inference
import threading
import time
import io
import requests
import traceback

lock=threading.Lock()
audio_url=None
audio_label="Analysing..."
audio_pred=0

app=Flask(__name__)


@app.route("/")
def home():
    return "Running! go at /audio for the audio live streaming!"

@app.route("/setting_audio", methods=["POST"])
def set_audio():
    global audio_url
    data = request.json or {}
    ip = data.get("ip")
    if not ip:
        return jsonify({"Error": "No Ip address has been found!"}), 400

    # Build candidate URL (you already send host:port)
    candidate = f"http://{ip}/audio.wav"
    print("[SET_AUDIO] candidate:", candidate)

    # quick validation with a short timeout and a browser UA
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IP-Webcam-Tester/1.0)"}
    try:
        r = requests.get(candidate, stream=True, timeout=3, headers=headers)
        r.close()
        print("[SET_AUDIO] validation status_code:", r.status_code, "content-type:", r.headers.get("Content-Type"))
        if r.status_code != 200:
            return jsonify({"Error": "Target returned non-200", "code": r.status_code}), 400
    except Exception as e:
        print("[SET_AUDIO] validation exception:", e)
        traceback.print_exc()
        return jsonify({"Error": f"Unable to reach target audio URL: {e}"}), 400

    audio_url = candidate
    print("[SET_AUDIO] audio_url set to:", audio_url)
    return jsonify({"Status": "Audio Link Established", "audio_url": audio_url}), 200

def get_audio_inference():
    global audio_label, audio_pred, audio_url

    headers = {"User-Agent": "Mozilla/5.0 (compatible; IP-Webcam-Inference/1.0)"}

    while True:
        if audio_url is None:
            # not configured yet — wait
            time.sleep(0.5)
            continue

        try:
            print("[INFER] attempting GET", audio_url)
            with requests.get(audio_url, stream=True, timeout=6, headers=headers) as r:
                print("[INFER] GET returned", getattr(r, "status_code", None))
                if r.status_code != 200:
                    print("[INFER] non-200 from audio source:", r.status_code)
                    time.sleep(1)
                    continue

                audio_buffer = io.BytesIO()
                start = time.time()
                for chunk in r.iter_content(chunk_size=4096):
                    if not chunk:
                        break
                    audio_buffer.write(chunk)
                    if time.time() - start > 3:  # ~3 seconds of data
                        break

                size = audio_buffer.getbuffer().nbytes
                print(f"[INFER] received {size} bytes; running inference now...")

                # small safety: require a minimum number of bytes
                if size < 2000:
                    print("[INFER] buffer too small, skipping inference")
                    time.sleep(0.5)
                    continue

                audio_buffer.seek(0)
                try:
                    # be explicit with keyword so signature mismatches don't bite us
                    conf, label = run_audio_inference(file_buffer=audio_buffer)
                except Exception as ie:
                    print("[INFER] inference raised exception:", ie)
                    traceback.print_exc()
                    time.sleep(1)
                    continue

                with lock:
                    audio_pred = float(conf)
                    audio_label = str(label)
                    print(f"[INFER] updated -> label={audio_label}, pred={audio_pred:.4f}")

        except Exception as e:
            print("[INFER] Network/inference loop exception:", e)
            traceback.print_exc()
            time.sleep(2)

def stream_audio():
    """
    this should fetch all the audios and yields it.
    ps: this is not vibecoded even when there is comment like this because i just dont want to forget what i wrote
        -Moss Pants ;)
    """

    global audio_url

    while True:
        if audio_url is None:
            time.sleep(1)
            continue
        try:
            with requests.get(audio_url,stream=True) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    yield(chunk)

        except Exception as e:
            print(f"Error streaming audio! code={e}")
            time.sleep(1)



@app.route("/audio_feed")
def audio_feed():
    return Response(stream_audio(),mimetype="audio/wav")

@app.route("/status")
def get_status():
    return jsonify({"Audio_Label":audio_label,"Audio_Prediction":audio_pred})


if __name__=="__main__":
    threading.Thread(target=get_audio_inference,daemon=True).start()
    app.run(host="0.0.0.0",port=5000,threaded=True)

    