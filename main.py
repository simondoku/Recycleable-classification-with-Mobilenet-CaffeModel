import cv2
import streamlit as st
from gtts import gTTS
from tempfile import NamedTemporaryFile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Page configuration
st.set_page_config(page_title="Midori Eco", layout="wide", page_icon="♻️")

# Recycling bin classification based on detected objects
BIN_GUIDE = {
    "book": "Blue recycling bin",
    "bottle": "Red recycling bin",
    "chair": "Black recycle bin",
    "pottedplant": "Pink recycling bin",
    "tvmonitor": "Purple recycle bin"
}

# Load MobileNet SSD model
@st.cache_resource
def load_model():
    prototxt_path = "deploy.prototxt"
    model_path = "mobilenet_iter_73000.caffemodel"
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

model = load_model()

# Class labels for the MobileNet SSD model
CLASS_LABELS = {
    5: "bottle",
    9: "chair",
    16: "pottedplant",
    20: "tvmonitor",
}

def play_audio(text):
    """Generate and play audio for a given text."""
    tts = gTTS(text=text, lang="en", slow=False)
    temp_audio = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    st.audio(temp_audio.name, format="audio/mp3")

def detect_objects(frame, model):
    """Detect objects using the MobileNet SSD model."""
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    detections = model.forward()
    return detections

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detect objects
        detections = detect_objects(img, self.model)

        h, w = img.shape[:2]
        detected_messages = []
        detected_labels = set()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Threshold for detection
                class_id = int(detections[0, 0, i, 1])
                label = CLASS_LABELS.get(class_id, "Unknown")
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # Draw green box for each detected object
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{label}: {confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if label in BIN_GUIDE and label not in detected_labels:
                    bin_instruction = BIN_GUIDE[label]
                    message = f"I detected {label}. Please place it in the {bin_instruction}."
                    detected_messages.append(message)
                    play_audio(message)  # Play audio for the detected object
                    detected_labels.add(label)  # Add the label to the set to avoid repeats

                elif label not in detected_labels:
                    message = f"I detected {label}. I'm not sure which bin it goes in. Please check the recycling guide."
                    detected_messages.append(message)
                    play_audio(message)  # Play audio for unknown objects
                    detected_labels.add(label)  # Add the label to the set to avoid repeats

        # Display detection messages
        st.write("\n".join(detected_messages) if detected_messages else "No objects detected.")

        return img

def main():
    st.title("♻️ Midori Eco: Your Recycling Assistant")
    st.write(
        "Hello, I am **Midori Eco**, your smart assistant for recycling. "
        "Hold an item in front of the webcam and press **'Detect Object'** to find out which bin it goes in."
    )
    st.sidebar.header("User Guide")
    st.sidebar.write(
        "- **Start the webcam** to enable live detection.\n"
        "- Click **'Detect Object'** to check an item's category.\n"
        "- Press **'Stop Webcam'** when you're done."
    )

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()