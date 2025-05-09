import os
import torch
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    workers = 0
    batch = 16
    data_name = "yolov8"
    data_path = 'data.yaml'

    # 确保路径是绝对路径
    data_path = os.path.abspath(data_path)
    print(f"数据路径(绝对路径): {data_path}")

    directory_path = os.path.dirname(data_path)
    print(f"目录路径: {directory_path}")

    # 检查YAML文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在!")
        exit(1)

    # 加载模型
    model = YOLO('yolov8n.pt')

    train_name_path = 'yolov8'
    print(f"训练输出路径: {train_name_path}")

    # 检查是否有检查点可以恢复训练
    last_checkpoint = f"{train_name_path}/weights/last.pt"
    best_checkpoint = f"{train_name_path}/weights/best.pt"

    if os.path.exists(last_checkpoint):
        model = YOLO(last_checkpoint)
        print(f"加载检查点: {last_checkpoint}")
    elif os.path.exists(best_checkpoint):
        model = YOLO(best_checkpoint)
        print(f"加载最佳检查点: {best_checkpoint}")
    else:
        print(f"未找到检查点文件: {last_checkpoint}，将使用预训练模型")

    # 直接使用默认尺寸640
    imgsz = 640
    print(f"使用默认图像尺寸: {imgsz}")

    # 开始训练
    try:
        results = model.train(
            data=data_path,  # 数据集配置文件
            device=device,  # 训练设备 (cuda/cpu)
            workers=workers,  # 数据加载的工作线程数
            imgsz=imgsz,  # 训练图像尺寸
            epochs=1,  # 训练周期数，通用设置100-300
            patience=20,  # 早停耐心值，连续20个周期没有改进则停止
            batch=batch,  # 批次大小
            optimizer='AdamW',  # 优化器选择 (SGD/Adam/AdamW)
            lr0=0.001,  # 初始学习率
            lrf=0.01,  # 最终学习率 = lr0 * lrf
            cos_lr=True,  # 使用余弦学习率调度
            weight_decay=0.0005,  # 权重衰减，减少过拟合
            warmup_epochs=3.0,  # 预热周期
            warmup_momentum=0.8,  # 预热动量
            warmup_bias_lr=0.1,  # 预热偏置学习率
            multi_scale=True,  # 多尺度训练，提高模型对不同尺寸目标的鲁棒性
            rect=False,  # 矩形训练 (batch中使用相似宽高比的图像来减少填充)
            amp=True,  # 混合精度训练，提高训练速度
            fraction=1.0,  # 数据集分数 (0.0-1.0)
            resume=False,  # 是否从中断处恢复训练
            pretrained=True,  # 使用预训练权重
            dropout=0.0,  # dropout率
            label_smoothing=0.0,  # 标签平滑值
            seed=0,  # 随机种子，保证结果可重现
            verbose=True,  # 详细输出
            cache=False,  # 是否缓存图像
            save=True,  # 保存训练结果
            val=True,  # 在训练期间验证
            overlap_mask=True,  # 掩码重叠（用于分割任务）
            mask_ratio=4,  # 掩码下采样比例（用于分割任务）
            nbs=64,  # 标称批次大小
            name=data_name  # 训练结果保存名称
        )

        print(f"训练完成! 结果已保存到 {train_name_path} 目录")
        print(f"最佳模型路径: {train_name_path}/weights/best.pt")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback

        traceback.print_exc()