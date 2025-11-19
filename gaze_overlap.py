import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
)
import cv2
import numpy as np

def estimate_rgb_fps(vrs_dp, rgb_stream_id, max_frames=30):
    timestamps = []
    idx = 0
    while idx < max_frames:
        try:
            frame = vrs_dp.get_image_data_by_index(rgb_stream_id, idx)
        except RuntimeError:
            break
        meta = frame[1]
        ts_ns = meta.capture_timestamp_ns
        timestamps.append(ts_ns)
        idx += 1

    if len(timestamps) < 2:
        print("Not enough frames to estimate FPS")
        return None

    diffs_ns = np.diff(timestamps)
    diffs_s  = diffs_ns / 1e9
    mean_dt  = float(np.mean(diffs_s))
    fps      = 1.0 / mean_dt
    print(f"Estimated RGB FPS: {fps:.3f} (from {len(timestamps)} frames)")
    return fps

def estimate_gaze_rate(gaze_cpf, max_samples=30):
    timestamps = []
    for i, rec in enumerate(gaze_cpf):
        if i >= max_samples:
            break
        ts_s = rec.tracking_timestamp.total_seconds()
        timestamps.append(ts_s)

    if len(timestamps) < 2:
        print("Not enough gaze samples to estimate rate")
        return None

    diffs = np.diff(timestamps)
    mean_dt = float(np.mean(diffs))
    fps = 1.0 / mean_dt
    print(f"Estimated gaze rate: {fps:.3f} Hz (from {len(timestamps)} samples)")
    return fps

# =========Configs===========
gaze_path = "mps_Eye_track_test_vrs\eye_gaze\general_eye_gaze.csv"
vrs_file = "Eye_track_test.vrs"
rgb_stream_id = StreamId("214-1")
depth_m = 1.0        # Default eye gaze depth for 3D points to 1 meter
output_mp4 = "Test_gaze.mp4"

print("Loading gaze csv...")
gaze_cpf = mps.read_eyegaze(gaze_path)

print("Opening VRS...")
vrs_dp = data_provider.create_vrs_data_provider(vrs_file)
rgb_stream_label = vrs_dp.get_label_from_stream_id(rgb_stream_id)
device_calib = vrs_dp.get_device_calibration()
rgb_calib = device_calib.get_camera_calib(rgb_stream_label)

rgb_fps = estimate_rgb_fps(vrs_dp, rgb_stream_id)
gaze_fps = estimate_gaze_rate(gaze_cpf)
if abs(rgb_fps - gaze_fps) > 0.5:
    print("[WARN]: The fps of rgb and gaze are not the same.")

frame0 = vrs_dp.get_image_data_by_index(rgb_stream_id, 0)
img0 = frame0[0].to_numpy_array()
w, h = img0.shape[:2]   # rotate

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_mp4, fourcc, rgb_fps, (w, h))
print(f"Video size = {w}x{h}, fps = {rgb_fps}")

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

    img = image_data.to_numpy_array()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)  # rotate

    if x is not None and y is not None:
        # rotate the gaze coords
        new_x = w - 1 - y
        new_y = x
        xi, yi = int(round(new_x)), int(round(new_y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(img_bgr, (xi, yi), 10, (0, 0, 255), -1)

    writer.write(img_bgr)

    idx += 1
    if idx % 100 == 0:
        print(f"Processed {idx} frames...")

writer.release()
print(f"Done. Saved gaze overlay video -> {output_mp4}")
print(idx, "frames processed.", meta.frame_number if meta else None)
