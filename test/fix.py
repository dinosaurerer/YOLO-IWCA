import cv2
import pyttsx3
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

import threading
# 异步语音播报函数，避免阻塞和 run loop 冲突
tts_lock = threading.Lock()


def add_rounded_corners(image, corner_radius=20, border_color=(0, 0, 0)):
    """
    为图像添加透明圆角边框
    :param image: 原始BGR图像
    :param corner_radius: 圆角半径
    :param border_color: 边框颜色
    :return: 带圆角的BGRA格式图像
    """
    # 转换为BGRA格式
    h, w = image.shape[:2]
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 创建透明边框
    border_size = corner_radius
    new_h = h + border_size * 2
    new_w = w + border_size * 2
    bordered = cv2.copyMakeBorder(bgra, border_size, border_size, border_size, border_size,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # 创建圆角蒙版
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.rectangle(mask, (corner_radius, 0), (new_w - corner_radius, new_h), 255, -1)
    cv2.rectangle(mask, (0, corner_radius), (new_w, new_h - corner_radius), 255, -1)
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (new_w - corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (corner_radius, new_h - corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (new_w - corner_radius, new_h - corner_radius), corner_radius, 255, -1)

    # 应用蒙版
    bordered[:, :, 3] = cv2.bitwise_and(bordered[:, :, 3], mask)

    # 添加外边框
    cv2.rectangle(bordered, (0, 0), (new_w - 1, new_h - 1), (*border_color, 255), 1)
    return bordered



def speak_trash_tip(text):
    def _speak():
        with tts_lock:
            local_engine = pyttsx3.init()  # 用局部引擎防止全局状态污染
            local_engine.setProperty('rate', 160)
            local_engine.setProperty('volume', 1.0)
            try:
                local_engine.say(text)
                local_engine.runAndWait()
            except RuntimeError:
                pass
            finally:
                del local_engine

    threading.Thread(target=_speak, daemon=True).start()

class YOLOv8Detector:
    def __init__(self, model_path=None, img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.model = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.names = self.model.names
            return True, "模型加载成功!"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"

    def detect_image(self, image):
        if self.model is None:
            return None, "请先加载模型!"

        try:
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = image.copy()

            results = self.model.predict(
                source=img,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                device=self.device
            )

            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            detection_info = []
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                conf = float(det.conf[0])
                cls = int(det.cls[0])
                detection_info.append({
                    'class': self.names[cls],
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

            return result_img_rgb, detection_info
        except Exception as e:
            return None, f"图片检测失败: {str(e)}"

    def process_video(self, video_path):
        if self.model is None:
            return None, "请先加载模型!"

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = video_path.replace(".mp4", "_output.mp4")
            # st.write(f"输出视频路径: {output_path}")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(
                    source=frame,
                    imgsz=self.img_size,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    device=self.device
                )

                result_img = results[0].plot()
                out.write(result_img)

            cap.release()
            out.release()
            return output_path, None
        except Exception as e:
            return None, f"视频处理失败: {str(e)}"



# 初始化模型
model_path = "../weights/yolo11-365-noaug.pt"  # 替换为实际模型路径
detector = YOLOv8Detector(
            model_path=model_path,
            img_size=640,
            conf_thres=0.25,
            iou_thres=0.45
        )

# 垃圾分类提示信息 (示例)
TRASH_HINTS = {
    "recyclable waste": {"tip": "请投放至蓝色可回收垃圾桶。", "color": "蓝色 ♻️"},
    "hazardous waste": {"tip": "请投放至红色有害垃圾桶。", "color": "红色 ☣️"},
    "kitchen waste": {"tip": "请投放至绿色厨余垃圾桶。", "color": "绿色 🥦"},
    "other waste": {"tip": "请投放至灰色其他垃圾桶。", "color": "灰色 🗑️"}
}


# 本地摄像头检测函数
def run_camera_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    last_spoken_classes = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break

            # 模型推理

            results = detector.model.predict(
                source=frame,
                imgsz=detector.img_size,
                conf=detector.conf_thres,
                iou=detector.iou_thres,
                device=detector.device
            )

            # 显示结果
            result_img = results[0].plot()
            # 添加圆角效果
            decorated_img = add_rounded_corners(result_img,
                                                corner_radius=20,
                                                border_color=(100, 100, 100))

            # 转换回BGR格式（部分环境可能不支持透明通道显示）
            final_display = cv2.cvtColor(decorated_img, cv2.COLOR_BGRA2BGR)

            cv2.imshow('Detection', final_display)

            # 原始语音逻辑（完全保持您的实现）
            detected_classes = [detector.model.names[int(box.cls)] for box in results[0].boxes]
            if detected_classes:
                for cls in set(detected_classes):
                    if cls in TRASH_HINTS and cls not in last_spoken_classes:
                        speak_trash_tip(TRASH_HINTS[cls]["tip"])
                        last_spoken_classes.add(cls)
            else:
                last_spoken_classes.clear()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# 运行摄像头检测
if __name__ == "__main__":
    run_camera_detection()