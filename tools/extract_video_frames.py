import cv2
import os

# 分类映射
class_names = {
    0: "recyclable_waste",
    1: "hazardous_waste",
    2: "kitchen_waste",
    3: "other_waste"
}


def create_output_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in class_names.values():
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)


def extract_frames(video_path, output_dir, frame_interval=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 无法打开视频文件:", video_path)
        return

    create_output_dirs(output_dir)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            video_path = os.path.splitext(os.path.basename(video_path))[0]
            filename = f"{video_path}_{saved_count:05d}.jpg"
            # 默认保存到 recyclable_waste，可之后手动移动或标注
            save_path = os.path.join(output_dir, class_names[0], filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ 完成，提取并保存了 {saved_count} 张图片（每 {frame_interval} 帧保存一次）")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从视频中提取图像帧（用于YOLO数据集）")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--interval", type=int, default=10, help="每隔多少帧保存一张图像")

    args = parser.parse_args()

    extract_frames(args.video, args.output, args.interval)

"python extract_video_frames.py --video video/shuiping-1.mp4 --interval 4"
