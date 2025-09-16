# server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import numpy as np
import os
import traceback
import logging

MODEL_PATH = "cow_model.h5"         # or path to saved_model dir
REMOTE_MODEL_URL = os.environ.get("REMOTE_MODEL_URL")  # optional: download at startup
ALLOW_LOCAL_H5 = True

app = FastAPI(title="GaushalaNet API")
logger = logging.getLogger("gaushala")
logger.setLevel(logging.INFO)

# model holder
_model = None
_model_load_error = None

def _try_load_model_local(path):
    """
    Try multiple ways of loading the model (tf.keras then keras).
    Returns model or raises exception.
    """
    # 1) Try tf.keras
    try:
        import tensorflow as tf
        logger.info("Trying tf.keras.models.load_model(...)")
        # compile=False avoids attempting to recreate optimizer state and may skip some issues
        m = tf.keras.models.load_model(path, compile=False)
        logger.info("Loaded model via tf.keras")
        return m
    except Exception as e_tf:
        tf_err = e_tf
        logger.warning(f"tf.keras load failed: {e_tf}")

    # 2) Try standalone keras if installed
    try:
        import importlib
        keras_spec = importlib.util.find_spec("keras")
        if keras_spec is not None:
            import keras
            logger.info("Trying keras.models.load_model(...) (standalone keras)")
            m = keras.models.load_model(path)
            logger.info("Loaded model via standalone keras")
            return m
    except Exception as e_keras:
        logger.warning(f"standalone keras load failed: {e_keras}")

    # failed both
    # raise the first error (tf error) but include both in message
    raise ValueError(f"Could not load model with tf.keras or keras. tf_error={tf_err}; check model format and TF/Keras versions.")

def _download_model_if_needed(url, dst="cow_model.h5"):
    import requests
    logger.info(f"Attempting to download model from {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(1024*1024):
            if chunk:
                f.write(chunk)
    logger.info("Model downloaded")

def try_load_model_on_startup():
    global _model, _model_load_error
    try:
        # optional remote download path
        if REMOTE_MODEL_URL and not os.path.exists(MODEL_PATH):
            _download_model_if_needed(REMOTE_MODEL_URL, MODEL_PATH)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        _model = _try_load_model_local(MODEL_PATH)
        _model_load_error = None
    except Exception as e:
        _model = None
        _model_load_error = "".join(traceback.format_exception_only(type(e), e)).strip()
        logger.error("Model load failed: %s", _model_load_error)
        logger.error(traceback.format_exc())

# Preload on startup
try_load_model_on_startup()

def preprocess_image_bytes(bytes_io, target_size=(224,224)):
    im = Image.open(io.BytesIO(bytes_io)).convert("RGB")
    im = im.resize(target_size)   # simple resize; you can switch to ImageOps.fit if you used center-crop during training
    arr = np.asarray(im).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": _model is not None, "model_error": _model_load_error}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        # return clear JSON error so UI can show it
        raise HTTPException(status_code=503, detail={"error":"model_not_loaded", "message": _model_load_error})

    try:
        data = await file.read()
        x = preprocess_image_bytes(data)
        # predict — some models use model.predict, others call model(x)
        try:
            preds = _model.predict(x)
        except Exception:
            preds = _model(x)  # try calling the model (TF eager)
        # handle common output shapes
        if hasattr(preds, "numpy"):
            preds = preds.numpy()
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[1] > 1:
            idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))
        else:
            # fallback: single-output regression or single-probability
            idx = int(np.argmax(preds)) if preds.size > 1 else int(np.argmax(preds.flatten()))
            conf = float(np.max(preds.flatten()))
        return JSONResponse({"class_index": idx, "confidence": conf})
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail={"error":"prediction_failed", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), log_level="info")
