import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
)
import cv2
import numpy as np

gaze_path = "general_eye_gaze.csv"
vrs_file = "Eye_track_test.vrs"
rgb_stream_id = StreamId("214-1")
depth_m = 1.0        # Default eye gaze depth for 3D points to 1 meter
output_mp4 = "Test_gaze.mp4"
fps = 10.0           # Have to change according to different profiles used

print("Loading gaze csv...")
gaze_cpf = mps.read_eyegaze(gaze_path)

print("Opening VRS...")
vrs_dp = data_provider.create_vrs_data_provider(vrs_file)
rgb_stream_label = vrs_dp.get_label_from_stream_id(rgb_stream_id)
device_calib = vrs_dp.get_device_calibration()
rgb_calib = device_calib.get_camera_calib(rgb_stream_label)

frame0 = vrs_dp.get_image_data_by_index(rgb_stream_id, 0)
img0 = frame0[0].to_numpy_array()        
h, w = img0.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
print(f"Video size = {w}x{h}, fps = {fps}")

idx = 0
while True:
    try:
        frame = vrs_dp.get_image_data_by_index(rgb_stream_id, idx)
    except RuntimeError:
        break

    image_data = frame[0]   
    meta       = frame[1]   
    frame_ts_ns = meta.capture_timestamp_ns
    eye_gaze_info = get_nearest_eye_gaze(gaze_cpf, frame_ts_ns)

    x, y = None, None
    if eye_gaze_info is not None:
        gaze_projection = get_gaze_vector_reprojection(
            eye_gaze_info,
            rgb_stream_label,
            device_calib,
            rgb_calib,
            depth_m,
        )
        x, y = gaze_projection 

    # Debugging
    # try:
    #     img = image_data.to_numpy_array()    
    # except RuntimeError as e:
    #     print(f"[WARN] to_numpy_array failed at frame {idx}: {e}")
    #     idx += 1
    #     continue

    img = image_data.to_numpy_array()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if x is not None and y is not None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img_bgr, (xi, yi), 6, (0, 0, 255), -1)

    writer.write(img_bgr)

    idx += 1
    if idx % 100 == 0:
        print(f"Processed {idx} frames...")

writer.release()
print(f"Done. Saved gaze overlay video -> {output_mp4}")
print(idx, "frames processed.", meta.frame_number if meta else None)