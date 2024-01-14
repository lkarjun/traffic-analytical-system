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
class ModelResults:
    image: np.ndarray
    image_with_bbox: np.ndarray | None = None
    xyxy_points: list | None = None

    

def detect_cars(frames: list[np.ndarray], endpoint_url: str) -> dict[int, ModelResults]:
    results = {}
    for i, image in stqdm(enumerate(frames), total=len(frames)):
        _, img_encoded = cv2.imencode(".jpg", image)
        image_bytes = img_encoded.tobytes()
        files = {"file": image_bytes}
        response = requests.post(endpoint_url, files=files)
        assert response.ok, "Failed to detect image..."
        results[i] = ModelResults(
            image=image, xyxy_points=response.json()
        )
    return results



class DetectCars:
    def __init__(self, video_file, endpoint_url="") -> None:
        self.video_bytes = video_file.read()
        self.endpoint_url = endpoint_url
        self.video_hash = generate_video_hash(self.video_bytes)

        self.is_cached = (ROOT_PATH / self.video_hash).exists()



    def detect_cars_in_frames(self):
        for i, modelresult in stqdm(self.frames.items(), leave=False, desc="Saving processed frames..."): 
            _, img_encoded = cv2.imencode(".jpg", modelresult.image)
            image_bytes = img_encoded.tobytes()
            files = {"file": image_bytes}
            response = requests.post(self.endpoint_url, files=files)
            assert response.ok, "Failed to detect image..."
            self.frames[i].xyxy_points = response.json()
        

    def get_frames(self) -> None:
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
            i: ModelResults(image=image)
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
                frame = modelresult.image_with_bbox if modelresult.image_with_bbox else modelresult.image
                video_writer.write(frame)
            video_writer.release()

            with open(out_filename, "rb") as fp:
                video_bytes = fp.read()
        return video_bytes


# async def get_model_result(
#         session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, frames: dict, idx: int, url: str
#     ) -> None:

#     image = frames[idx].image
#     _, img_encoded = cv2.imencode(".jpg", image)
#     image_bytes = img_encoded.tobytes()
#     async with semaphore, session.post(url, data={"file": image_bytes}) as response:
#         result = await response.json() if response.status == 200 else None
#         frames[idx].xyxy_points = result


# async def detect_cars(frames: list[np.ndarray], endpoint_url: str, max_concurrent_uploads=2):
#     frames = {
#         i: ModelResults(image=image) for i, image in enumerate(frames)
#     }

#     semaphore = asyncio.Semaphore(max_concurrent_uploads)

#     async with aiohttp.ClientSession() as session:
#         tasks = [
#             get_model_result(
#                 session, semaphore, frames=frames, idx=idx, url=endpoint_url
#             ) 
#             for idx in frames.keys()
#         ]
#         await asyncio.gather(*tasks)
#     return frames



# if __name__ == "__main__":
#     from tqdm import tqdm
#     endpoint_url = 'https://a6e8-34-67-97-119.ngrok-free.app?conf=0.1'

#     # image = cv2.imread("../traffic_signal.jpg")
#     # print(detect_cars(image=image, endpoint_url=endpoint_url))

#     cap = cv2.VideoCapture("RoadTraffic.mp4")
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(3)) 
#     frame_height = int(cap.get(4)) 
#     size = (frame_width, frame_height) 
    
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#     cap.release()

#     # frames = asyncio.run(detect_cars(frames=frames, endpoint_url=endpoint_url))

#     print(len(frames))
#     total = 0
#     for frame in tqdm(frames):
#         try:
#             detect_cars(frame, endpoint_url=endpoint_url)
#             total += 1
#         except Exception as e:
#             print(str(e))
#         # if frame.xyxy_points:

#         #     total += 1
    
#     print(f"{total=} {len(frames)=}")

