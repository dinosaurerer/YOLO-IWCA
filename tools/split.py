import os
import shutil
import random

# 设置路径
origin_image_dir = r"C:\Users\lenovo\Desktop\images"
origin_label_dir = r"C:\Users\lenovo\Desktop\labels"

target_image_train_dir = r"C:\Users\lenovo\Desktop\dataset\images\train"
target_image_val_dir = r"C:\Users\lenovo\Desktop\dataset\images\val"
target_label_train_dir = r"C:\Users\lenovo\Desktop\dataset\labels\train"
target_label_val_dir = r"C:\Users\lenovo\Desktop\dataset\labels\val"

# 创建目标文件夹
os.makedirs(target_image_train_dir, exist_ok=True)
os.makedirs(target_image_val_dir, exist_ok=True)
os.makedirs(target_label_train_dir, exist_ok=True)
os.makedirs(target_label_val_dir, exist_ok=True)

# 获取所有有对应标签的图片文件
image_files = [
    f for f in os.listdir(origin_image_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png')) and
    os.path.isfile(os.path.join(origin_label_dir, os.path.splitext(f)[0] + ".txt"))
]

# 打乱顺序
random.shuffle(image_files)

# 划分 8:2
num_total = len(image_files)
num_train = int(num_total * 0.8)

train_files = image_files[:num_train]
val_files = image_files[num_train:]

# 拷贝函数
def copy_files(files, img_dst, lbl_dst):
    for img_file in files:
        name, ext = os.path.splitext(img_file)
        lbl_file = name + ".txt"

        src_img_path = os.path.join(origin_image_dir, img_file)
        src_lbl_path = os.path.join(origin_label_dir, lbl_file)

        dst_img_path = os.path.join(img_dst, img_file)
        dst_lbl_path = os.path.join(lbl_dst, lbl_file)

        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_lbl_path, dst_lbl_path)

# 执行复制
copy_files(train_files, target_image_train_dir, target_label_train_dir)
copy_files(val_files, target_image_val_dir, target_label_val_dir)

print(f"✅ 数据集划分完成：训练集 {len(train_files)} 张，验证集 {len(val_files)} 张。")
