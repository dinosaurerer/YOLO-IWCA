import matplotlib.pyplot as plt
# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei' # 选择一个支持中文的字体

# 模拟数据
strategies = ['Backbone+MSCA', 'Neck+MSCA', 'Detect+MSCA']
metrics = [0.954, 0.928, 0.971]  # mAP@50 values

# 颜色、样式配置
colors = ['#4E79A7', '#F28E2B', '#E15759']
bar_width = 0.6

# 创建图表
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(strategies, metrics, width=bar_width, color=colors)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 图表美化
ax.set_title('MSCA Attention 集成策略对比（mAP@50）', fontsize=14, fontweight='bold')
ax.set_ylabel('mAP@50 (↑)', fontsize=12)
ax.set_ylim(0.90, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

# 显示图像
plt.show()
