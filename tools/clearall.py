# 递归清空文件夹下所有文件夹里的文件
import os

def clear_all_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
    print("✅ 所有文件夹中的文件已清空。")


if __name__ == "__main__":
    clear_all_folders(r"C:\Users\lenovo\Desktop\dataset")