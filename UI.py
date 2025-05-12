import base64

import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import datetime
from datetime import date
import mysql.connector
from mysql.connector import Error
import pandas as pd
import time

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "dbwaste"
}

XX = "基于YOLOv11的垃圾分类检测系统"
YY = "YOLOv11"


# 初始化数据库表
def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(255) PRIMARY KEY,
                password VARCHAR(255) NOT NULL
            )
        ''')
        conn.commit()
    except Error as e:
        st.error(f"数据库错误: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# 用户注册到数据库
def register_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()

        # 创建用户专属的检测结果表
        table_name = f"results_{username}"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_name VARCHAR(255),
                class VARCHAR(255),
                confidence FLOAT,
                bbox VARCHAR(255),
                detection_time DATETIME
            )
        """
        cursor.execute(create_table_query)
        conn.commit()

        return True
    except Error as e:
        st.error(f"注册失败: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# 验证用户登录
def verify_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        return result is not None and result[0] == password
    except Error as e:
        st.error(f"登录验证失败: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


# 页面配置必须最先
st.set_page_config(
    page_title=XX,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 仅首次运行显示动画
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = True

    splash_placeholder = st.empty()
    with splash_placeholder.container():
        st.markdown("""
            <style>
                .splash-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 90vh;
                    background-color: #E8F5E9;
                    font-family: "Segoe UI", sans-serif;
                }
                .splash-title {
                    font-size: 38px;
                    font-weight: bold;
                    color: #2E7D32;
                    margin-bottom: 10px;
                }
                .splash-sub {
                    font-size: 18px;
                    color: #388E3C;
                    margin-bottom: 20px;
                }
                .loader {
                    border: 6px solid #C8E6C9;
                    border-top: 6px solid #388E3C;
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>

            <div class="splash-container">
                <div class="splash-title">YOLOv11 垃圾分类检测系统</div>
                <div class="splash-sub">正在启动，请稍候...</div>
                <div class="loader"></div>
            </div>
        """, unsafe_allow_html=True)

    time.sleep(1.5)
    splash_placeholder.empty()


# 初始化数据库
init_db()

# 美化样式注入
st.markdown("""
    <style>
        /* 主标题美化 */
        .main-title {
            font-size: 40px;
            color: #2C6E49;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            background-color: #E0F2F1;
        }

        /* 小标题 */
        .section-title {
            font-size: 24px;
            color: #00796B;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        /* 分隔线 */
        .custom-hr {
            border: 1px solid #B2DFDB;
            margin: 20px 0;
        }

        /* 检测结果表格字体大小调整 */
        .dataframe th, .dataframe td {
            font-size: 16px !important;
        }

        /* Streamlit按钮 hover 效果 */
        button[kind="primary"] {
            background-color: #26A69A;
            color: white;
            border-radius: 6px;
        }
        button[kind="primary"]:hover {
            background-color: #00796B;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# 初始化会话状态
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
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误！")

    with tab2:
        st.header("用户注册")
        new_username = st.text_input("新用户名")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认密码", type="password")
        if st.button("注册"):
            if new_password != confirm_password:
                st.error("两次输入的密码不一致，请重新输入！")
            else:
                if register_user(new_username, new_password):
                    st.success("注册成功，请登录！")
    st.stop()

# 主界面布局
st.title(XX)
st.markdown(f"欢迎 {st.session_state.username}！")
st.markdown("上传图片、视频和" + YY + "模型权重文件进行目标检测")
st.markdown(f"<div class='main-title'>{XX}</div>", unsafe_allow_html=True)

# 初始化会话状态变量
if 'model' not in st.session_state:
    st.session_state.model = None
if 'current_weights' not in st.session_state:
    st.session_state.current_weights = None


def convert_to_h264_opencv(input_path):
    output_path = input_path.replace(".mp4", "_converted.mp4")
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 编码
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"视频流转换完成: {output_path}")
    return output_path


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


def history_query():
    st.markdown("<div class='section-title'>历史记录查询 📊</div>", unsafe_allow_html=True)
    # 检查用户是否已登录
    if "username" not in st.session_state or not st.session_state["username"]:
        st.warning("请先登录以查看历史记录")
        return

    st.markdown("**提示：** 在此查询您的历史检测记录，可根据日期范围、文件名和类别进行筛选。")
    username = st.session_state["username"]
    table_name = f"results_{username}"

    # 连接数据库
    conn = mysql.connector.connect(**DB_CONFIG)
    if not conn:
        return
    cursor = conn.cursor()

    # 获取类别选项（供筛选使用）
    class_options = []
    try:
        cursor.execute(f"SELECT DISTINCT class FROM {table_name}")
        class_options = [row[0] for row in cursor.fetchall()]
    except mysql.connector.Error as e:
        err_msg = str(e).lower()
        # 如果表不存在，说明没有历史记录
        if "doesn't exist" in err_msg or "does not exist" in err_msg:
            st.info("当前没有历史记录。")
        else:
            st.error(f"获取类别列表失败：{e}")
        cursor.close()
        conn.close()
        return

    # 筛选条件表单
    with st.form(key="filter_form"):
        st.subheader("筛选条件")
        today = date.today()
        # 默认日期范围：一年内
        default_start = today.replace(year=today.year - 1) if today.year > 1 else today
        start_date = st.date_input("开始日期", value=default_start)
        end_date = st.date_input("结束日期", value=today)
        image_name = st.text_input("图像文件名包含", "")
        selected_classes = st.multiselect("类别", options=class_options)
        submitted = st.form_submit_button("查询")

    # 如果用户没有点击查询按钮，则退出
    if not submitted:
        cursor.close()
        conn.close()
        return

    # 检查日期范围合理性
    if start_date > end_date:
        st.error("开始日期必须早于结束日期")
        cursor.close()
        conn.close()
        return

    # 构建 SQL 查询
    query = f"SELECT * FROM {table_name} WHERE 1=1"
    params = []
    # 添加日期范围过滤
    query += " AND DATE(detection_time) BETWEEN %s AND %s"
    params.append(start_date.strftime("%Y-%m-%d"))
    params.append(end_date.strftime("%Y-%m-%d"))
    # 添加文件名过滤
    if image_name:
        query += " AND image_name LIKE %s"
        params.append(f"%{image_name}%")
    # 添加类别过滤
    if selected_classes:
        placeholders = ", ".join(["%s"] * len(selected_classes))
        query += f" AND class IN ({placeholders})"
        params.extend(selected_classes)

    try:
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            st.subheader("查询结果")
            st.dataframe(df)
            # CSV下载按钮
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name=f"{username}_history.csv",
                mime="text/csv"
            )
        else:
            st.info("没有符合条件的记录。")
    except mysql.connector.Error as e:
        st.error(f"查询失败：{e}")
    finally:
        cursor.close()
        conn.close()


# 侧边栏配置
st.sidebar.header("模型设置")
uploaded_weights = st.sidebar.file_uploader("上传" + YY + "权重文件", type=["pt"])

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
tab1, tab2, tab3, tab4 = st.tabs(["图像检测", "视频检测", "摄像头检测", "历史记录查询"])

with tab1:
    st.markdown("<div class='section-title'>图像检测 📷</div>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")

    if uploaded_image:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_image)

        with col1:
            st.image(image, caption="原始图像", use_container_width=True)

        if st.button("执行图像检测"):
            if st.session_state.model is None:
                st.error("请先加载模型！")
            else:
                start_time = time.time()
                result_img, detection_info = st.session_state.model.detect_image(image)
                elapsed_time = time.time() - start_time

                if result_img is not None:
                    with col2:
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(result_img, caption="检测结果", use_container_width=True)
                        st.success(f"检测完成！耗时: {elapsed_time:.2f}秒")

                    if detection_info:
                        result_df = {
                            "类别": [d['class'] for d in detection_info],
                            "置信度": [f"{d['confidence']:.2f}" for d in detection_info],
                            "位置": [f"{d['bbox']}" for d in detection_info]
                        }
                        st.dataframe(result_df, use_container_width=True)

                        # 将检测结果存储到数据库
                        try:
                            conn = mysql.connector.connect(**DB_CONFIG)
                            cursor = conn.cursor()
                            table_name = f"results_{st.session_state.username}"
                            detection_time = datetime.datetime.now()
                            image_name = uploaded_image.name

                            for d in detection_info:
                                insert_query = f"""
                                                    INSERT INTO `{table_name}` (image_name, class, confidence, bbox, detection_time)
                                                    VALUES (%s, %s, %s, %s, %s)
                                                """
                                bbox_str = ','.join(map(str, d['bbox']))
                                cursor.execute(insert_query,
                                               (image_name, d['class'], d['confidence'], bbox_str, detection_time))

                            conn.commit()
                        except Error as e:
                            st.error(f"保存检测结果失败: {e}")
                        finally:
                            if conn.is_connected():
                                cursor.close()
                                conn.close()
                else:
                    st.error("图像检测失败")
# 自定义分隔线（插入在模块之间）
st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
with tab2:
    st.markdown("<div class='section-title'>视频检测 🎞️</div>", unsafe_allow_html=True)
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
                out_put = convert_to_h264_opencv(processed_path)
                elapsed_time = time.time() - start_time

                if processed_path:
                    st.success(f"视频处理完成！耗时: {elapsed_time:.2f}秒")

                    # 尝试使用Streamlit的原生视频播放器
                    try:
                        with open(out_put, "rb") as f:
                            video_bytes = f.read()
                            st.markdown(
                                f"""
                                <video controls autoplay muted playsinline width="640" style="border: 1px solid #ccc; border-radius: 8px;">
                                    <source src="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" type="video/mp4">
                                    您的浏览器不支持视频播放。
                                </video>
                                """,
                                unsafe_allow_html=True
                            )
                            st.info("✅ 视频播放完毕，可重新检测或下载。")

                    except Exception as video_error:
                        st.error(f"视频播放失败: {str(video_error)}")
                        st.warning("处理后的视频可能格式不兼容Streamlit播放器，但处理成功。请在应用外查看视频文件。")

                        # 提供下载链接作为备选方案
                        with open(out_put, "rb") as f:
                            st.write("处理后的视频下载链接:")
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
                        os.unlink(out_put)
                    except:
                        pass  # 忽略清理错误

# 自定义分隔线（插入在模块之间）
st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-title'>摄像头检测 🎥</div>", unsafe_allow_html=True)
    st.markdown("请确保您的设备已连接摄像头，建议使用桌面版运行此功能。")

    run_cam = st.checkbox("启用摄像头实时检测")
    FRAME_WINDOW = st.empty()

    if run_cam:
        if st.session_state.model is None:
            st.error("请先加载模型！")
        else:
            cap = cv2.VideoCapture(0)
            detector = st.session_state.model

            st.info("摄像头已启用，点击取消勾选以停止运行。")

            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("无法读取摄像头画面")
                    break

                # OpenCV 读取为 BGR，需要转换为 RGB 供模型输入
                frame_rgb = frame

                # 模型预测
                results = detector.model.predict(
                    source=frame_rgb,
                    imgsz=detector.img_size,
                    conf=detector.conf_thres,
                    iou=detector.iou_thres,
                    device=detector.device
                )

                # 获取检测结果并绘制（OpenCV画框是BGR）
                result_img = results[0].plot()

                # 再次转换为 RGB 用于 st.image 显示
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                # 显示在页面上
                FRAME_WINDOW.image(result_rgb)

            cap.release()

with tab4:
    history_query()


# 设备信息
st.sidebar.markdown("---")
device_info = f"运行设备: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}"
if torch.cuda.is_available():
    device_info += f"\n{torch.cuda.get_device_name(0)}"
st.sidebar.code(device_info)

# 使用说明
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ 操作指南")
st.sidebar.markdown("""
1. 上传模型权重文件（`.pt`）  
2. 调整图像尺寸、置信度、IoU 阈值  
3. 选择检测模式：
   - 📷 图像检测  
   - 🎞️ 视频检测  
   - 🎥 摄像头检测（需开启权限）  
4. 上传或启用对应文件/设备  
5. 点击执行检测按钮  
6. 在“历史记录查询”中筛选并查看检测结果  
""")
