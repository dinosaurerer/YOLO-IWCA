import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import time
import subprocess
import shutil
from PIL import Image
from ultralytics import YOLO

XX="基于YOLOv11垃圾分类检测系统"
YY="YOLOv11"

# 设置页面配置
st.set_page_config(
    page_title=XX,
    layout="wide",
    initial_sidebar_state="expanded"
)


# 初始化会话状态
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# 登录和注册界面
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        st.header("用户登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.button("登录"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("登录成功！")
            else:
                st.error("用户名或密码错误！")

    with tab2:
        st.header("用户注册")
        new_username = st.text_input("新用户名")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        if st.button("注册"):
            if new_username in st.session_state.users:
                st.error("该用户名已存在，请选择其他用户名！")
            elif new_password != confirm_password:
                st.error("两次输入的密码不一致，请重新输入！")
            else:
                st.session_state.users[new_username] = new_password
                st.success("注册成功，请登录！")
    st.stop()

# 主界面布局
st.title(XX)
st.markdown("上传图片、视频和"+YY+"模型权重文件进行目标检测")

# 初始化会话状态变量
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_weights' not in st.session_state:
    st.session_state.current_weights = None


def convert_video_to_compatible_format(input_path, output_path):
    """
    使用OpenCV重新编码视频以确保兼容性
    """
    try:
        # 打开输入视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False, "无法打开视频文件"

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:  # 如果fps异常低，设置一个合理的默认值
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 写入帧
            out.write(frame)

        # 释放资源
        cap.release()
        out.release()

        return True, "视频转换成功"
    except Exception as e:
        return False, f"视频转换失败: {str(e)}"


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
            # 首先确保输入视频是兼容格式
            compatible_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            success, message = convert_video_to_compatible_format(video_path, compatible_input)

            if not success:
                return None, message

            # 使用兼容格式的视频进行处理
            cap = cv2.VideoCapture(compatible_input)
            if not cap.isOpened():
                return None, "无法打开视频文件"

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps < 1:  # 如果fps异常低，设置一个合理的默认值
                fps = 25.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建临时输出文件
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = output_file.name
            output_file.close()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # 如果无法获取总帧数，则估算一个值
                total_frames = 1000
                st.warning("无法确定视频总帧数，使用估计值进行进度显示")

            progress_bar = st.progress(0)
            status_text = st.empty()

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

                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                frame_count += 1
                progress = min(frame_count / total_frames, 1.0)  # 确保不超过1.0
                progress_bar.progress(progress)
                status_text.text(f"处理进度: {frame_count}/{total_frames} 帧")

            cap.release()
            out.release()

            # 清理临时转换文件
            if os.path.exists(compatible_input):
                os.unlink(compatible_input)

            progress_bar.empty()
            status_text.text("视频处理完成！")

            # 确保视频被正确编码，可以考虑使用ffmpeg进行最终处理
            final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            try:
                # 如果系统安装了ffmpeg，使用ffmpeg重新编码确保兼容性
                subprocess.run([
                    'ffmpeg', '-y', '-i', output_path,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', final_output
                ], check=True, stderr=subprocess.PIPE)

                # 如果ffmpeg成功，使用重新编码的视频
                if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                    return final_output, None
            except (subprocess.SubprocessError, FileNotFoundError):
                # 如果ffmpeg失败或不存在，继续使用原始输出
                if os.path.exists(final_output):
                    os.unlink(final_output)

            return output_path, None

        except Exception as e:
            return None, f"视频处理失败: {str(e)}"


# 侧边栏配置
st.sidebar.header("模型设置")
uploaded_weights = st.sidebar.file_uploader("上传"+YY+"权重文件", type=["pt"])

# 模型参数
img_size = st.sidebar.slider("图像尺寸", 320, 1280, 640, 32)
conf_thres = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IoU阈值", 0.1, 1.0, 0.45, 0.05)

# 加载或更新模型
if uploaded_weights and (uploaded_weights != st.session_state.current_weights):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(uploaded_weights.getvalue())
        model_path = tmp_file.name

    with st.spinner("加载模型中..."):
        detector = YOLOv8Detector(
            model_path=model_path,
            img_size=img_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )
        success, message = detector.load_model(model_path)

    if success:
        st.session_state.model = detector
        st.session_state.current_weights = uploaded_weights
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
        st.session_state.model = None

# 主内容区域
tab1, tab2 = st.tabs(["图像检测", "视频检测"])

with tab1:
    st.header("图像检测")
    uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")

    if uploaded_image:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_image)

        with col1:
            st.image(image, caption="原始图像", use_column_width=True)

        if st.button("执行图像检测"):
            if st.session_state.model is None:
                st.error("请先加载模型！")
            else:
                start_time = time.time()
                result_img, detection_info = st.session_state.model.detect_image(image)
                elapsed_time = time.time() - start_time

                if result_img is not None:
                    with col2:
                        st.image(result_img, caption="检测结果", use_column_width=True)
                        st.success(f"检测完成！耗时: {elapsed_time:.2f}秒")

                    if detection_info:
                        result_df = {
                            "类别": [d['class'] for d in detection_info],
                            "置信度": [f"{d['confidence']:.2f}" for d in detection_info],
                            "位置": [f"{d['bbox']}" for d in detection_info]
                        }
                        st.dataframe(result_df, use_container_width=True)
                else:
                    st.error("图像检测失败")

with tab2:
    st.header("视频检测")
    uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if uploaded_video:
        video_bytes = uploaded_video.read()

        # 保存上传的视频到临时文件
        temp_video_path = tempfile.NamedTemporaryFile(delete=False,
                                                      suffix=f'.{uploaded_video.name.split(".")[-1]}').name
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        if st.button("执行视频检测"):
            if st.session_state.model is None:
                st.error("请先加载模型！")
            else:
                start_time = time.time()
                processed_path, error = st.session_state.model.process_video(temp_video_path)
                elapsed_time = time.time() - start_time

                if processed_path:
                    st.success(f"视频处理完成！耗时: {elapsed_time:.2f}秒")

                    # 尝试使用Streamlit的原生视频播放器
                    try:
                        with open(processed_path, "rb") as f:
                            video_bytes = f.read()
                        st.video(video_bytes)
                    except Exception as video_error:
                        st.error(f"视频播放失败: {str(video_error)}")
                        st.warning("处理后的视频可能格式不兼容Streamlit播放器，但处理成功。请在应用外查看视频文件。")

                        # 提供下载链接作为备选方案
                        with open(processed_path, "rb") as f:
                            video_bytes = f.read()
                        st.download_button(
                            label="下载处理后的视频",
                            data=video_bytes,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )

                    # 清理临时文件
                    try:
                        os.unlink(temp_video_path)
                        os.unlink(processed_path)
                    except:
                        pass  # 忽略清理错误
                elif error:
                    st.error(error)

# 设备信息
st.sidebar.markdown("---")
device_info = f"运行设备: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}"
if torch.cuda.is_available():
    device_info += f"\n{torch.cuda.get_device_name(0)}"
st.sidebar.code(device_info)

# 使用说明
st.sidebar.markdown("---")
st.sidebar.markdown("### 操作指南")
st.sidebar.markdown("""
1. 上传权重文件(.pt)
2. 调整检测参数
3. 选择检测模式（图片/视频）
4. 上传待检测文件
5. 点击执行检测按钮
""")