#!/usr/bin/env python3
"""OAK camera live demo: RGB + depth + host-side ONNX hole detection (MMYOLO export)."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

from singleinfer_onnx import create_onnx_session, letterbox, postprocess_output

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'onnx' / 'epoch_230.onnx'
DEFAULT_SAVE = ROOT / 'savedata'


def parse_args():
    parser = argparse.ArgumentParser(
        description='OAK 实时坑洞检测（主机 ONNX 推理 + 深度测距）')
    parser.add_argument(
        '--model',
        type=str,
        default=str(DEFAULT_MODEL),
        help='MMYOLO 导出的 ONNX 模型路径',
    )
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--save', action='store_true', help='保存 4K RGB 帧')
    parser.add_argument(
        '--save-dir',
        type=str,
        default=str(DEFAULT_SAVE),
        help='保存根目录',
    )
    return parser.parse_args()


def calculate_xyz(det_norm, depth_data, fx, fy, cx, cy):
    if depth_data is None:
        return None, None, None
    center_x = int(
        (det_norm['xmin'] + det_norm['xmax']) * depth_data.shape[1] / 2)
    center_y = int(
        (det_norm['ymin'] + det_norm['ymax']) * depth_data.shape[0] / 2)
    try:
        z = float(depth_data[center_y, center_x])
        if z == 0:
            return None, None, None
        x = (center_x - cx) * z / fx
        y = (center_y - cy) * z / fy
        return x, y, z
    except (IndexError, TypeError):
        return None, None, None


def main():
    args = parse_args()
    rgb_dir = None
    if args.save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rgb_dir = Path(args.save_dir) / f'rgb_{timestamp}'
        rgb_dir.mkdir(parents=True, exist_ok=True)
        print(f'保存目录: {rgb_dir}')

    session = create_onnx_session(args.model)

    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb_full = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    xout_depth.setStreamName('depth')
    xout_rgb_full.setStreamName('rgb_full')

    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setIspScale(1, 1)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(True)
    depth.setSubpixel(True)

    mono_left.out.link(depth.left)
    mono_right.out.link(depth.right)
    depth.depth.link(xout_depth.input)
    cam_rgb.preview.link(xout_rgb.input)
    cam_rgb.video.link(xout_rgb_full.input)

    with dai.Device(pipeline) as device:
        calib = device.readCalibration()
        m_rgb = np.array(
            calib.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                cam_rgb.getResolutionWidth(),
                cam_rgb.getResolutionHeight(),
            ))
        fx, fy = m_rgb[0][0], m_rgb[1][1]
        cx, cy = m_rgb[0][2], m_rgb[1][2]

        q_rgb = device.getOutputQueue('rgb', 4, False)
        q_depth = device.getOutputQueue('depth', 4, False)
        q_rgb_full = device.getOutputQueue('rgb_full', 4, False)

        frame_count = 0
        counter = 0
        start_time = time.monotonic()
        depth_data = None
        display_frame = None
        last_detections = []
        last_detection_time = 0.0
        detection_timeout = 0.5

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                preview = in_rgb.getCvFrame()
                fps = counter / max(time.monotonic() - start_time, 1e-6)
                cv2.putText(
                    preview, f'FPS: {fps:.2f}', (2, preview.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))

            in_depth = q_depth.tryGet()
            depth_vis = None
            if in_depth is not None:
                depth_data = in_depth.getFrame().copy()
                depth_vis = (
                    depth_data * (255 / depth.initialConfig.getMaxDisparity())
                ).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            in_full = q_rgb_full.tryGet()
            frame_full = None
            if in_full is not None:
                frame_full = in_full.getCvFrame()
                display_frame = cv2.resize(
                    frame_full,
                    (frame_full.shape[1] // 4, frame_full.shape[0] // 4))
                if args.save and rgb_dir is not None:
                    out_path = rgb_dir / f'frame_{frame_count:06d}.jpg'
                    cv2.imwrite(str(out_path), frame_full)
                    frame_count += 1

            if frame_full is not None and depth_data is not None:
                img = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
                img, ratio, pad = letterbox(img, (640, 640), auto=False)
                blob = img.astype(np.float32) / 255.0
                blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)

                outputs = session.run(None, {'images': blob})
                detections = postprocess_output(
                    outputs, ratio, pad, conf_thres=args.conf)

                now = time.time()
                if detections:
                    last_detections = detections
                    last_detection_time = now
                elif now - last_detection_time < detection_timeout:
                    detections = last_detections

                for det in detections:
                    x1, y1, x2, y2 = map(int, det['box'])
                    h, w = frame_full.shape[:2]
                    det_norm = {
                        'xmin': x1 / w,
                        'ymin': y1 / h,
                        'xmax': x2 / w,
                        'ymax': y2 / h,
                    }
                    x, y, z = calculate_xyz(
                        det_norm, depth_data, fx, fy, cx, cy)

                    x1d, y1d, x2d, y2d = x1 // 4, y1 // 4, x2 // 4, y2 // 4
                    if z is not None:
                        print(
                            f'距离 Z={z/1000:.2f}m X={x/1000:.2f}m Y={y/1000:.2f}m')
                        cv2.putText(
                            display_frame, f'Z:{z/1000:.2f}m',
                            (x1d + 10, y1d + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            (0, 255, 0))
                    cv2.rectangle(
                        display_frame, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)

            if display_frame is not None:
                cv2.imshow('RGB', display_frame)
            if depth_vis is not None:
                cv2.imshow('Depth', depth_vis)

            counter += 1
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
