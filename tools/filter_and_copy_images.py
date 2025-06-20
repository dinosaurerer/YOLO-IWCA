import os
import shutil

# 配置路径
label_dir = r"C:\Users\lenovo\Desktop\labels"
image_root = r"D:\PyCharmWorkSpace\YOLO-IWCA\output"
target_image_dir = r"C:\Users\lenovo\Desktop\images"

# 四个类别的子文件夹
categories = [
    "recyclable_waste",
    "hazardous_waste",
    "kitchen_waste",
    "other_waste"
]

# 创建目标图片文件夹（如果不存在）
os.makedirs(target_image_dir, exist_ok=True)

# 读取所有标签文件名（不带扩展名）
label_names = set()
for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        label_names.add(os.path.splitext(file)[0])

# 统计
copied = 0
skipped = 0

# 遍历四个类别文件夹
for category in categories:
    category_path = os.path.join(image_root, category)
    for file in os.listdir(category_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_without_ext = os.path.splitext(file)[0]
            if name_without_ext in label_names:
                src_path = os.path.join(category_path, file)
                dst_path = os.path.join(target_image_dir, file)
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                skipped += 1

print(f"✅ 处理完成：复制了 {copied} 张图片，跳过了 {skipped} 张没有对应标签的图片。")
