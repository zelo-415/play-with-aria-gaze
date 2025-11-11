# overlay_with_mps_utils.py
import cv2
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps import read_eyegaze, utils as mps_utils

VRS = r"Eye_track_test.vrs"
MP4_IN = r"Eye_track_test.mp4"
MP4_OUT = r"Eye_track_test_gaze.mp4"
GAZE_CSV = r"mps_Eye_track_test_vrs\eye_gaze\general_eye_gaze.csv"
RGB_STREAM = "214-1"
DEFAULT_DEPTH_M = 1.0                # 没有深度时的回退

provider = data_provider.create_vrs_data_provider(VRS)
rgb_id = StreamId(RGB_STREAM)
rgb_label = provider.get_label_from_stream_id(rgb_id)
device_calib = provider.get_device_calibration()
rgb_calib = device_calib.get_camera_calib(rgb_label)

gaze = read_eyegaze(GAZE_CSV)

cap = cv2.VideoCapture(MP4_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(MP4_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ts_ns = int(frame_idx / fps * 1e9)
    info = mps_utils.get_nearest_eye_gaze(gaze, ts_ns)
    if info:
        depth_m = getattr(info, "depth_m", float("nan"))
        if not (depth_m == depth_m):
            depth_m = DEFAULT_DEPTH_M
        uv = mps_utils.get_gaze_vector_reprojection(info, rgb_label, device_calib, rgb_calib, depth_m)
        if uv is not None:
            u, v = int(uv[0]), int(uv[1])
            if 0 <= u < w and 0 <= v < h:
                
                cv2.circle(frame, (u, v), 6, (0, 255, 0), -1)
                cv2.circle(frame, (u, v), 10, (0, 0, 0), 1)
    writer.write(frame)
    frame_idx += 1

cap.release(); writer.release()
print("Done:", MP4_OUT)
