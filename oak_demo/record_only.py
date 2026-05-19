#!/usr/bin/env python3
"""OAK camera: RGB + depth preview and optional 4K frame recording (no inference)."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE = ROOT / 'savedata'


def parse_args():
    parser = argparse.ArgumentParser(description='OAK 数据采集（无推理）')
    parser.add_argument('--save', action='store_true', help='连续保存 4K RGB 帧')
    parser.add_argument('--depth', action='store_true', help='显示深度预览')
    parser.add_argument('--rgb', action='store_true', help='显示 RGB 预览')
    parser.add_argument(
        '--save-dir',
        type=str,
        default=str(DEFAULT_SAVE),
        help='保存根目录',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.depth and not args.rgb:
        args.rgb = True

    rgb_dir = None
    if args.save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rgb_dir = Path(args.save_dir) / f'rgb_{timestamp}'
        rgb_dir.mkdir(parents=True, exist_ok=True)
        print(f'保存目录: {rgb_dir}')

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
        q_rgb = device.getOutputQueue('rgb', 4, False)
        q_depth = device.getOutputQueue('depth', 4, False)
        q_rgb_full = device.getOutputQueue('rgb_full', 4, False)

        frame = None
        depth_frame = None
        frame_count = 0
        counter = 0
        start_time = time.monotonic()

        if args.rgb:
            cv2.namedWindow('RGB Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('RGB Preview', 640, 480)
        if args.depth:
            cv2.namedWindow('Depth Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Depth Preview', 320, 240)

        print(f'RGB={args.rgb} Depth={args.depth} Save={args.save}')
        print("按 'q' 退出")

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                fps = counter / max(time.monotonic() - start_time, 1e-6)
                cv2.putText(
                    frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            in_depth = q_depth.tryGet()
            if in_depth is not None:
                depth_frame = in_depth.getFrame()
                depth_frame = (
                    depth_frame * (255 / depth.initialConfig.getMaxDisparity())
                ).astype(np.uint8)
                depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
                depth_frame = cv2.resize(depth_frame, (320, 240))

            in_full = q_rgb_full.tryGet()
            if in_full is not None and args.save and rgb_dir is not None:
                frame_full = in_full.getCvFrame()
                cv2.imwrite(
                    str(rgb_dir / f'frame_{frame_count:06d}.jpg'), frame_full)
                frame_count += 1

            if args.rgb and frame is not None:
                cv2.imshow('RGB Preview', frame)
            if args.depth and depth_frame is not None:
                cv2.imshow('Depth Preview', depth_frame)

            counter += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
