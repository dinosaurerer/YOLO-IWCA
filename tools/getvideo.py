import cv2
import os
import glob
import re
import sys
from pathlib import Path
import time
import numpy as np


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def resize_with_padding(image, target_width, target_height):

    h, w = image.shape[:2]
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # 取最小比例，确保图像完全适应目标尺寸

    # 计算调整后的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 调整图像大小，保持宽高比
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建黑色背景图像
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算居中位置
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # 将调整后的图像复制到居中位置
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return result


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {script_dir}")

    # 图片所在文件夹路径
    image_folder = os.path.join(script_dir, 'videodata')
    print(f"图像文件夹: {image_folder}")

    # 输出视频的路径
    output_video = os.path.join(script_dir, 'output_video.mp4')

    # 检查文件夹是否存在
    if not os.path.exists(image_folder):
        print(f"错误: 文件夹 '{image_folder}' 不存在")
        print(f"请确保在脚本同级目录下创建名为'videodata'的文件夹并放入图片")
        input("按任意键退出...")
        sys.exit(1)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff']

    # 获取所有图片文件并移除重复项
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(image_folder, f'*{ext.upper()}')))

    # 去除重复项
    image_files = list(set(image_files))

    # 检查是否找到图片
    if not image_files:
        print(f"错误: 在 '{image_folder}' 中未找到任何图片文件")
        print("支持的图片格式:", ', '.join(image_extensions))
        input("按任意键退出...")
        sys.exit(1)

    # 使用自然排序
    image_files.sort(key=natural_sort_key)

    print(f'找到 {len(image_files)} 个图片文件，开始合成视频...')

    # 列出前几个文件路径用于调试
    for i, path in enumerate(image_files[:5]):
        print(f"文件 {i + 1}: {path}")
        # 检查文件是否真实存在
        if not os.path.isfile(path):
            print(f"  警告: 文件不存在!")
        else:
            # 检查文件大小
            print(f"  文件大小: {os.path.getsize(path)} 字节")

    # 定义固定的输出视频尺寸
    target_width, target_height = 640, 640  # 可以根据需要调整这个值

    # 尝试不同的编码器
    try:
        # 首先尝试 H.264 编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        video_writer = cv2.VideoWriter(output_video, fourcc, 2, (target_width, target_height))

        # 检查视频写入器是否成功初始化
        if not video_writer.isOpened():
            # 如果失败，尝试 XVID 编码
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI格式
            output_video = os.path.join(script_dir, 'output_video.avi')
            video_writer = cv2.VideoWriter(output_video, fourcc, 2, (target_width, target_height))

            # 如果仍然失败
            if not video_writer.isOpened():
                print("错误: 无法创建视频写入器，请检查OpenCV安装和编解码器")
                input("按任意键退出...")
                sys.exit(1)
    except Exception as e:
        print(f"错误: 创建视频写入器时出错: {e}")
        input("按任意键退出...")
        sys.exit(1)

    # 进度显示变量
    total_images = len(image_files)
    processed_images = 0
    start_time = time.time()

    try:
        # 将图片添加到视频中
        for img_path in image_files:
            print(f"正在处理: {img_path}", end="")

            # 尝试使用不同的读取方法
            img = None

            # 方法1：使用opencv标准读取
            img = cv2.imread(img_path)

            # 方法2：如果标准方法失败，尝试使用imdecode
            if img is None:
                try:
                    with open(img_path, 'rb') as f:
                        img_array = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    print(" - 使用imdecode成功")
                except Exception as e:
                    print(f" - imdecode失败: {e}")

            # 方法3：如果上述方法都失败，尝试使用PIL库
            if img is None:
                try:
                    from PIL import Image
                    pil_img = Image.open(img_path)
                    img = np.array(pil_img.convert('RGB'))
                    # 转换BGR（OpenCV使用的颜色空间）
                    img = img[:, :, ::-1].copy()
                    print(" - 使用PIL成功")
                except Exception as e:
                    print(f" - PIL失败: {e}")

            if img is not None:
                print(" - 成功")

                # 调整图像大小，保持宽高比，并用黑色填充
                resized_img = resize_with_padding(img, target_width, target_height)

                # 写入调整后的图像
                video_writer.write(resized_img)
                processed_images += 1

                # 显示进度
                progress = (processed_images / total_images) * 100
                elapsed_time = time.time() - start_time
                images_per_sec = processed_images / elapsed_time if elapsed_time > 0 else 0
                eta = (total_images - processed_images) / images_per_sec if images_per_sec > 0 else 0

                print(f"\r处理进度: {progress:.1f}% ({processed_images}/{total_images}) "
                      f"速度: {images_per_sec:.1f}张/秒 剩余时间: {eta:.1f}秒")
            else:
                print(f"\n警告: 无法读取图片 '{img_path}'，已跳过")

    except Exception as e:
        print(f"\n错误: 处理图片时出错: {e}")
        # 打印详细的堆栈跟踪
        import traceback
        traceback.print_exc()
        # 确保释放资源
        video_writer.release()
        input("按任意键退出...")
        sys.exit(1)

    # 释放资源
    video_writer.release()

    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        print(f"视频生成完成: {output_video}")
        # 获取文件大小
        file_size_bytes = os.path.getsize(output_video)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"视频大小: {file_size_mb:.2f} MB")
        print(f"视频时长: {total_images / 2:.1f} 秒 (每0.5秒一帧)")

        # 打开输出视频所在的文件夹
        try:
            if os.name == 'nt':  # Windows
                os.startfile(os.path.dirname(output_video))
            elif os.name == 'posix':  # macOS 和 Linux
                if sys.platform == 'darwin':  # macOS
                    os.system(f'open "{os.path.dirname(output_video)}"')
                else:  # Linux
                    os.system(f'xdg-open "{os.path.dirname(output_video)}"')
        except:
            pass  # 如果打开文件夹失败，继续执行
    else:
        print(f"错误: 视频生成失败或文件大小为0")

    print("任务完成")
    input("按任意键退出...")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"发生未处理的错误: {e}")
        # 打印详细的堆栈跟踪
        import traceback

        traceback.print_exc()
        input("按任意键退出...")