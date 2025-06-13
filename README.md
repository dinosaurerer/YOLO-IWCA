
<div align="center">

# ♻️ YOLOv11 垃圾分类检测系统

</div>

基于 Streamlit + YOLOv11 + OpenCV + MySQL 构建的智能垃圾检测平台，支持图像、视频、摄像头实时检测，并提供历史结果查询与模型管理功能。

<div align="center">
    <img src="https://img.shields.io/badge/Framework-Streamlit-brightgreen" />
    <img src="https://img.shields.io/badge/Model-YOLOv11-blue" />
    <img src="https://img.shields.io/badge/Backend-MySQL-orange" />
    <img src="https://img.shields.io/badge/UI-Responsive-lightgrey" />
</div>

---

## 🖼️ 系统界面预览

<img src="screenshots/s1.png" width="100%" alt="系统主界面">

## 🌍 实际场景应用

- 🗑️ **垃圾分类站点接入**
  - 系统可对接垃圾回收站或小区垃圾房的固定监控摄像头
  - 实时检测居民投放的垃圾类别，自动播报分类提示，辅助规范垃圾投放行为
  - 管理员可通过后台查看历史记录，实现数据统计与违规追踪

- 📱 **个人用户移动识别**
  - 支持部署在移动端设备（如手机、平板）中进行本地识别
  - 用户可通过拍照识别垃圾种类，获得对应的投放建议

- 🏫 **校园 / 企业宣传与实践**
  - 应用于学校环保课程或企业绿色活动中，提升公众垃圾分类意识
  - 支持演示模式与统计模式，用于教育、宣传与考核等不同场景

- 🧠 **智能设备集成扩展**
  - 作为智能垃圾桶前端识别模块，与机械投口联动，实现自动分类投放


## 🚀 功能亮点

### 🎨 **智能可视化界面**
  - 提供简洁直观的用户界面，支持图像 / 视频 / 摄像头实时检测展示
  - 实时渲染目标框、类别与置信度，支持主题色切换与动态按钮反馈

### 🧠 **改进型 YOLOv11-MSCA 模型**
  - 在原 YOLOv11 模型基础上引入多尺度通道注意力（MSCA）机制
  - 显著提升模型对小目标与复杂背景的检测精度，同时保持较高运行效率

### 🔊 **智能语音提示**
  - 系统可根据检测结果自动播报提示语
  - 例如检测到可回收物品时，语音提示：“请投放至蓝色可回收垃圾桶”

### 📷 **多源输入支持**
  - 支持图片、视频、摄像头以及批量文件夹检测
  - 视频检测支持自动转码并生成结果视频，可预览和下载

### 👤 **用户账户系统**
  - 支持用户注册与登录，实现账户隔离与私有记录管理

### 📦 **检测结果记录与筛选**
  - 所有检测结果自动存入数据库，支持：
    - 按日期筛选
    - 按垃圾类别过滤
    - 通过图像名关键词搜索

### ⬇️ **一键导出历史数据**
  - 支持将检测记录导出为 `.csv` 文件，方便统计分析与数据存档

### ⚙️ **自动适配运行环境**
  - 系统可自动检测并使用 GPU（如有）或切换至 CPU，优化运行效率


---



## 📦 安装与运行

### 1️⃣ 安装依赖环境

```bash
git clone https://github.com/dinosaurerer/YOLO-IWCA.git
cd YOLO-IWCA

# 建议使用虚拟环境
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

pip install -r requirements.txt
````

<details>
<summary>📄 requirements.txt </summary>

```txt
streamlit
ultralytics
opencv-python
numpy
Pillow
mysql-connector-python
```

</details>

---

### 2️⃣ 配置数据库（MySQL）

> 请确保你已安装 MySQL 并启动服务。

#### 修改数据库连接信息：

在 `UI.py` 中配置：

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "你的密码",
    "database": "dbwaste"
}
```

#### 创建数据库（一次性操作）：

```sql
CREATE DATABASE dbwaste;
```

系统将自动创建 `users` 表和用户专属的 `results_用户名` 检测结果表。

#### cmd中启动sql连接服务：
```bash
net start mysql
```

---

### 3️⃣ 启动应用

```bash
streamlit run UI.py
```

### 4️⃣ 关闭应用

```bash
在终端中按 Ctrl + C
```

---

## 🧭 使用指南

### 左侧控制栏设置参数：

* 上传模型权重 `.pt`
* 选择图像尺寸（默认 640）
* 设置置信度和 IoU 阈值

### 主界面操作：

| 模块        | 功能说明                      |
| --------- | ------------------------- |
| 📷 图像检测   | 上传单张图片并实时显示检测结果           |
| 🎞️ 视频检测  | 上传 `.mp4`/`.avi` 视频并处理后播放 |
| 🎥 摄像头检测  | 打开设备摄像头进行实时目标检测           |
| 📊 历史记录查询 | 根据用户名记录并支持筛选导出 `.csv`     |

---

## 🔐 用户系统说明

* 用户注册时系统自动为每个用户创建私有检测结果表；
* 检测记录永久保存，可随时按日期/类别查询；
* 数据库隔离，避免用户间干扰。

---

## 📂 项目结构说明

```
YOLO-IWCA/
├── UI.py                          # 主界面脚本，运行后启动智能垃圾分类助手的图形界面
├── getvideo.py                    # 视频处理脚本，从videodata目录读取视频帧并生成处理后的视频
├── output_video.mp4               # 处理后的示例视频输出结果
├── Intelligent Waste Classification Assistant.pdf  # 项目说明书或展示文档
├── README.md                      # 项目使用说明文档
├── screenshots/                   # 系统界面截图资源
│   └── s1.png                     # 示例截图（主界面）
├── test/                          # 测试文件夹
├── ultralytics/                   # YOLOv11 模型依赖（添加MSCA-attention）
├── videodata/                     # 视频帧图片数据目录
│   ├── fimg_228.jpg
│   ├── fimg_234.jpg
│   ├── ...
│   └── fimg_242.jpg
├── weights/                       # 模型权重目录
│   ├── yolo11-365-noaug.pt        # 自训练的 YOLOv11 模型权重（无数据增强）
│   ├── yolo11n.pt                 # YOLOv11 nano 模型权重
│   └── yolov8n.pt                 # YOLOv8 nano 模型权重（可对比测试）

```

---

## 💡 常见问题 FAQ  
<small>我们团队在项目开发时遇到的一些问题</small>  

### Q1: 启动时为什么看到“启动动画”反复出现？

A: 系统使用 `st.session_state` 控制动画仅首次显示，请确保你没有人为清空 session。

### Q2: 视频播放太大，页面撑爆？

A: 系统自动将处理后视频大小限定为 `640px` 并居中展示。

### Q3: 摄像头打不开？

A: 请确认是否允许浏览器/系统访问摄像头，建议使用桌面版运行 (`streamlit run`)。

---

## 📜 致谢

本系统基于以下开源工具开发：

* [Streamlit](https://streamlit.io/)
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [MySQL](https://www.mysql.com/)

---

## 📬 联系方式

如果你喜欢本项目或有改进建议，欢迎联系我：

* 📧 Email: [dinosaurerer@love.com](mailto:konglongxiao723@gmail.com)
* 💻 GitHub: [@dinosaurerer](https://github.com/dinosaurerer)

---

<div align="center">
    <b>🌿 让 AI 帮助城市更清洁 🌿</b>
</div>


---
