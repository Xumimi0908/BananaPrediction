import cv2
from PIL import Image
import torch
from collections import Counter

class VideoProcessor:
    def __init__(self, model, transform, class_names):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.model.eval()
        self.all_predictions = []
        self.progress = 0

    def process_frame(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = self.transform(pil_image).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            pred_class = self.class_names[predicted.item()]
            self.all_predictions.append(pred_class)
        return pred_class

    def process_video(self, video_path, update_callback=None):
        self.all_predictions = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pred = self.process_frame(frame)
            self.progress = (cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100

            if update_callback:
                update_callback(frame, pred, self.progress)

        cap.release()
        return self.get_statistics()

    def get_statistics(self):
        if not self.all_predictions:
            return None
        count = Counter(self.all_predictions)
        total = len(self.all_predictions)
        return {
            "total_frames": total,
            "stats": {k: f"{(v / total) * 100:.1f}%" for k, v in count.items()},
            "dominant_state": count.most_common(1)[0][0]
        }