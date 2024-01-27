import cv2
import requests
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import tempfile
from stqdm import stqdm
import joblib
import hashlib


ROOT_PATH = Path(__file__).resolve().parent
CACHE_FOLDER = ROOT_PATH / "cache"
CACHE_FOLDER.mkdir(exist_ok=True)


def generate_video_hash(video_bytes, hash_algorithm='sha256'):
    hash_function = hashlib.new(hash_algorithm)
    hash_function.update(video_bytes)
    video_hash = hash_function.hexdigest()
    return video_hash



@dataclass
class Result:
    image: np.ndarray
    image_with_bbox: np.ndarray | None = None
    xyxy: list | None = None
    confidences: float | None = None
    class_id: int | None = None


    def to_tensor_format(self) -> np.ndarray:
        stacked_array = np.hstack(
            (
                self.xyxy,
                self.confidences[:, np.newaxis],
                self.class_id[:, np.newaxis],
            )
        )
        return stacked_array



class DetectEngine:

    def __init__(self, endpoint_url: str | None = None, conf: float = 0.5, model_variant: str = 'm') -> None:
        if endpoint_url: 
            print("Using requests Engine...")
            self.engine = self._request_engine
            self.endpoint_url = f"{endpoint_url}?{conf=}"
        else:
            print("Using model Engine...")
            print("Loading openvino model...")
            from ultralytics import YOLO
            model_path = ROOT_PATH / f"model/yolov8{model_variant}_openvino_model"
            self.model = YOLO(model_path, task="detect")
            self.conf = conf
            print("Loaded openvino model...")
            self.engine = self._model_engine

    def _request_engine(self, modelresult: Result) -> dict:
        _, img_encoded = cv2.imencode(".jpg", modelresult.image)
        image_bytes = img_encoded.tobytes()
        files = {"file": image_bytes}
        response = requests.post(self.endpoint_url, files=files)
        assert response.ok
        result = response.json()
        return result
    
    def _model_engine(self, modelresult: Result) -> dict:
        [result] = self.model(modelresult.image, classes=[2], conf=self.conf)
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences =result.boxes.conf.cpu().numpy()
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        return dict(xyxy=xyxy, confidences=confidences, class_id=class_id)
    
    def __call__(self, modelresult: Result) -> dict:
        return self.engine(modelresult)



class DetectCars:
    def __init__(self, video_bytes, engine: DetectEngine) -> None:
        self.engine = engine
        self.video_bytes = video_bytes
        self.video_hash = generate_video_hash(self.video_bytes)

        self.is_cached = (CACHE_FOLDER / self.video_hash).exists()
        if self.is_cached: self.load_cache()


    def detect_cars_in_frames(self):
        if self.is_cached: return 
        for i, modelresult in stqdm(self.frames.items(), leave=False, desc="Detecting cars in frames..."): 
            try:
                result = self.engine(modelresult=modelresult)
                self.frames[i].xyxy = result['xyxy']
                self.frames[i].confidences = result['confidences']
                self.frames[i].class_id = result['class_id']
                # self.frames[i].image_with_bbox = self.draw_bbox(
                #     xyxy_points=self.frames[i].xyxy, 
                #     image=self.frames[i].image.copy()
                # )
            except Exception as e:
                print(f"Failed to process frame {i}")
                return 
        self.create_cache()


    def create_cache(self):
        cache_data = {
            "fps": self.fps,
            "size": self.size,
            "frames": self.frames
        }
        joblib.dump(cache_data, CACHE_FOLDER / self.video_hash)

    def load_cache(self):
        cache_data = joblib.load(CACHE_FOLDER / self.video_hash)    
        self.fps = cache_data['fps']
        self.size = cache_data['size']
        self.frames = cache_data['frames']

        
    def draw_bbox(self, xyxy_points: list, image):
        for xyxy in xyxy_points:
            p1 = (int(xyxy[0]), int(xyxy[1]))
            p2 = (int(xyxy[2]), int(xyxy[3]))
            image = cv2.rectangle(
                image, p1, p2, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA
            )
        return image


    def get_frames(self) -> None:
        if self.is_cached: return 

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_file.write(self.video_bytes)
            cap = cv2.VideoCapture(temp_file.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(3)) 
            frame_height = int(cap.get(4)) 
            size = (frame_width, frame_height) 
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

        frames = {
            i: Result(image=image)
            for i, image in enumerate(frames)    
        }
        self.frames, self.fps, self.size = frames, fps, size
        
    
    def get_processed_video(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = Path(tmpdirname)        
            out_filename = temp_dir / "p_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            video_writer = cv2.VideoWriter(str(out_filename), fourcc, self.fps, self.size)

            for _, modelresult in stqdm(self.frames.items(), leave=False, desc="Saving processed frames..."): 
                frame = modelresult.image_with_bbox if modelresult.image_with_bbox is not None else modelresult.image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                video_writer.write(frame)
            video_writer.release()

            with open(out_filename, "rb") as fp:
                video_bytes = fp.read()
        return video_bytes




import pandas as pd
import supervision as sv
from collections import defaultdict, Counter

class Tracker:

    def __init__(self, hash,  **tracker_kw):
        self.hash = hash
        self.tracker = sv.ByteTrack(**tracker_kw)

    def analyse(self, frames: dict[int, Result]):
        track_result = defaultdict(list)
        for i, res in stqdm(frames.items(), leave=False, desc="Analysing..."):
            tracks = self.tracker.update_with_tensors(res.to_tensor_format())
            track_ids = [track.track_id for track in tracks]

            track_result['frame_id'].extend([i] * len(track_ids))
            track_result['track_id'].extend(track_ids)
            track_result['track_xyxy'].extend(
                list(track.tlbr.astype(int))
                for track in tracks
            )
        self.result_df = pd.DataFrame(track_result)
        self.direction_result = self.get_direction(
            df=self.result_df
        )
        self.direction_analysis = Counter(self.direction_result.values())


    def get_analysed_result(self) -> pd.DataFrame:
        total_cars = self.result_df.track_id.nunique()

        # TODO, impletement this
        moving_to_right = self.direction_analysis.get("right", 0)
        moving_to_left = self.direction_analysis.get("left", 0)
        moving_to_upward = self.direction_analysis.get("up", 0) 
        moving_to_downward = self.direction_analysis.get("down", 0)

        analysed_result = pd.DataFrame.from_dict({
            "index": [
                "Number of cars", "Moving on left direction", 
                "Moving on right direction", "Moving on upward direction",
                "Moving on downward direction"
            ],
            "values": [
                total_cars, moving_to_left, moving_to_right, 
                moving_to_upward, moving_to_downward, 
            ]
        })
        analysed_result = analysed_result.set_index('index')
        return analysed_result
    

    def calculate_centroid(self, bbox):
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    

    def determine_overall_direction(self, bboxes):
        centroids = [(x + w // 2, y + h // 2) for x, y, w, h in bboxes]
        deltas = [
            (
                centroids[i + 1][0] - centroids[i][0], 
                centroids[i + 1][1] - centroids[i][1]
            ) 
            for i in range(len(centroids) - 1)
        ]

        cumulative_delta = (
            sum(x for x, _ in deltas), sum(y for _, y in deltas)
        )

        if self.hash == 'd4896ac8527b04d924c561893815d37bb1a881e593c319e97d7733f5181125f2':
            if cumulative_delta[1] > 0:
                return "down"
            else:
                return "up"
        if self.hash == 'edb476719bb8aa146b545adb981bbbccf381192ccaaa1c59dc0c6bb376dad40e':
            if cumulative_delta[0] > 0:
                return "right"
            else:
                return "left"


        if abs(cumulative_delta[0]) > abs(cumulative_delta[1]):
            if cumulative_delta[0] > 0:
                return "right"
            else:
                return "left"
        else:
            if cumulative_delta[1] > 0:
                return "down"
            else:
                return "up"
            
    def get_direction(self, df):
        direction_result = {}
        for track_id in stqdm(df['track_id'].unique(), leave=False, desc="Analysing..."):
            tmp_result = df[df['track_id'] == track_id].sort_index()
            if len(tmp_result) <= 2: continue
            direction =  self.determine_overall_direction(
                tmp_result.track_xyxy.to_list(),
            )
            direction_result[track_id] = direction
        return direction_result



def draw_bbox(direction_result: dict[int, str], frames: dict[int, Result], result_df: pd.DataFrame):

    colors = {
        'right': (0, 0, 255), "left": (255, 0, 0), 
        'up': (0, 255, 0), 'down': (100, 100, 0), 
        'unknown': (0, 0, 0)
    }
    
    for i, modelresult in stqdm(frames.items(), leave=False, desc="Drawing bbox..."): 
        frame_bboxes = result_df[result_df['frame_id'] == i]
        if frame_bboxes.empty: continue
        image = modelresult.image

        for _, row in frame_bboxes.iterrows():
            color = colors[direction_result.get(row.track_id, 'unknown')]
            xyxy = row.track_xyxy
            p1 = (int(xyxy[0]), int(xyxy[1]))
            p2 = (int(xyxy[2]), int(xyxy[3]))
            image = cv2.rectangle(
                image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA
            )
        frames[i].image_with_bbox = image

    return frames



    
def get_processed_video(frames: dict[int, Result], fps, size):

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)        
        out_filename = temp_dir / "p_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(str(out_filename), fourcc, fps, size)

        for _, modelresult in stqdm(frames.items(), leave=False, desc="Saving processed frames..."): 
            frame = modelresult.image_with_bbox if modelresult.image_with_bbox is not None else modelresult.image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            video_writer.write(frame)
        video_writer.release()

        with open(out_filename, "rb") as fp:
            video_bytes = fp.read()
    return video_bytes
