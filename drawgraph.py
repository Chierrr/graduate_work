import re
import matplotlib.pyplot as plt

# 假设log文件的格式如下，每行包含一个loss值
# Epoch 1, Loss: 0.8976
# Epoch 2, Loss: 0.7895
# ...

# 定义log文件的路径
log_file_path = 'logs/alkaid/train.log'

# 初始化一个列表来存储loss值
losses = []

# 打开log文件并读取内容
with open(log_file_path, 'r') as file:
    for line in file:
        # 使用正则表达式匹配方括号内的数字
        match = re.search(r'\[([\d\.\s,]+)\]', line)
        if match:
            # 提取出匹配到的数字字符串
            numbers_str = match.group(1)
            # 将数字字符串按逗号分割成列表
            numbers = [float(num) for num in numbers_str.split(',')]
            # 计算前6个数字的和
            loss_sum = numbers[1]
            # 将loss值的和添加到列表中
            losses.append(loss_sum)

            # 确保我们提取到了loss值
if not losses:
    print("没有找到loss值。")
else:
    # 使用matplotlib绘制loss值的折线图
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()