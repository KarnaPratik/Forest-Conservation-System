import numpy as np
import librosa as lb
import librosa.display
from PIL import Image as PIL_Image

def format_shape(data, target_height=128, target_width=1000):
    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    diff = np.max(data) - np.min(data)
    if diff == 0:
        data = np.zeros_like(data)
    else:
        data = (255 * (data - np.min(data)) / diff)

    data = data.astype(np.uint8)

    # Resize logic (Rows/Height)
    rows, cols = data.shape
    if rows < target_height:
        reps = int(np.ceil(target_height / rows))
        data = np.tile(data, (reps, 1))[:target_height, :]
    elif rows > target_height:
        data = data[:target_height, :]

    # Resize logic (Cols/Width)
    rows, cols = data.shape
    if cols < target_width:
        pad_width = target_width - cols
        data = np.pad(data, ((0,0), (0, pad_width)), mode="constant")
    elif cols > target_width:
        data = data[:, :target_width]

    return data.astype(np.float32)

def audio_to_image(file=None, max_size=1000, y=None, sr=22050):
    if file is not None:
        if y is None:
            y, sr = lb.load(file, sr=22050)

    y = np.asarray(y, dtype=np.float32)

    mels = lb.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mels_db = lb.power_to_db(mels, ref=np.max)

    mels_delta = lb.feature.delta(mels_db)
    mels_delta2 = lb.feature.delta(mels_db, order=2)

    layer0 = format_shape(mels_db)
    layer1 = format_shape(mels_delta)
    layer2 = format_shape(mels_delta2)

    final_image = np.dstack([layer0, layer1, layer2]).astype(np.float32)
    return final_image, y, sr

def for_single_audio(file):
    img, y, sr = audio_to_image(file)

    # Add batch dimension: (1, 128, 1000, 3)
    img_ready = np.expand_dims(img, axis=0)

    # REMOVE OR COMMENT OUT THIS LINE:
    # img_ready = img_ready / 255.0  <-- This is killing your accuracy!

    return img_ready, y, sr