import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV数据
train_df = pd.read_csv('data/fashion-mnist_train.csv')
test_df = pd.read_csv('data/fashion-mnist_test.csv')

# 提取数据和标签
x_train = train_df.iloc[:, 1:].values  # 图像像素数据
y_train = train_df.iloc[:, 0].values   # 标签数据

x_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# 将图像数据归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 将数据 reshape 为 28x28 的图像
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# 选择类别0（T-shirt/top）和类别1（Trouser）进行二元分类
binary_class_train_indices = np.where((y_train == 0) | (y_train == 1))
binary_class_test_indices = np.where((y_test == 0) | (y_test == 1))

x_train_binary = x_train[binary_class_train_indices]
y_train_binary = y_train[binary_class_train_indices]
x_test_binary = x_test[binary_class_test_indices]
y_test_binary = y_test[binary_class_test_indices]

# 将类别0和1映射为二进制标签
y_train_binary = np.where(y_train_binary == 0, 0, 1)
y_test_binary = np.where(y_test_binary == 0, 0, 1)

# 添加L2正则化和Dropout
l2_reg = tf.keras.regularizers.l2(0.001)  # L2正则化系数

# 构建二元分类模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=l2_reg),  # 第一层
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_reg),   # 第二层
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2_reg),   # 第三层
    tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=l2_reg),   # 第四层
    tf.keras.layers.Dense(1, activation='sigmoid')  # 输出层，Sigmoid 激活函数
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train_binary, y_train_binary, epochs=10, validation_data=(x_test_binary, y_test_binary))

# 评估模型
test_loss, test_acc = model.evaluate(x_test_binary, y_test_binary)
print(f"测试准确率：{test_acc}")

# 显示部分预测结果
predictions = model.predict(x_test_binary)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test_binary[i], cmap=plt.cm.binary)
    plt.xlabel(f"预测: {int(predictions[i] > 0.5)} | 实际: {y_test_binary[i]}")
plt.show()