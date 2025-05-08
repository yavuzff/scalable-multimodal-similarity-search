import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image


class VideoEmbeddingGenerator:
    """
    Video embedding generator using HuggingFace CLIP model.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    def generate_video_embedding(self, video_path):
        """
        Extract frames from the video, compute CLIP image embeddings for each frame,
        and return a mean-pooled embedding.
        """
        frames = extract_frames(video_path)
        if not frames:
            print(f"No frames extracted from {video_path}. Skipping.")
            return None

        # preprocess the frames with CLIPProcessor
        inputs = self.processor(images=frames, return_tensors="pt")
        # move tensors to the desired device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # compute image features using CLIP
        with torch.no_grad():
            frame_embeddings = self.model.get_image_features(**inputs)

        # mean pooling over all frames to obtain a single video embedding
        video_embedding = frame_embeddings.mean(dim=0)
        return video_embedding


def extract_frames(video_path):
    """
    Extract frames from a video file using OpenCV and convert them to PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # convert from BGR to RGB and then to a PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames
