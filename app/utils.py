import cv2
import numpy as np
import tempfile

def draw_bbox(xyxy_points: list, image):
    for xyxy in xyxy_points:
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        image = cv2.rectangle(image, p1, p2, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return image


def compute_centroid(bbox):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy



def get_frames(video_file) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
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

    return frames, fps, size




# def get_movement_direction(box1, box2):
#     directions = []
    
#     for i in range(1, len(bounding_boxes)):
#         # Compute centroids for consecutive frames
#         centroid_prev = compute_centroid(bounding_boxes[i - 1])
#         centroid_curr = compute_centroid(bounding_boxes[i])
        
#         # Compute the displacement vector
#         displacement_vector = np.subtract(centroid_curr, centroid_prev)
        
#         # Determine the horizontal direction based on the sign of the horizontal component
#         if displacement_vector[0] > 0:
#             horizontal_direction = 'Right'
#         elif displacement_vector[0] < 0:
#             horizontal_direction = 'Left'
#         else:
#             horizontal_direction = 'No Horizontal Movement'
        
#         # Determine the vertical direction based on the sign of the vertical component
#         if displacement_vector[1] > 0:
#             vertical_direction = 'Down'
#         elif displacement_vector[1] < 0:
#             vertical_direction = 'Up'
#         else:
#             vertical_direction = 'No Vertical Movement'
        
#         # Combine horizontal and vertical directions
#         direction = f"{vertical_direction}, {horizontal_direction}"
#         directions.append(direction)
    
#     return directions

# # Example usage
# directions = get_movement_direction(bounding_boxes)
