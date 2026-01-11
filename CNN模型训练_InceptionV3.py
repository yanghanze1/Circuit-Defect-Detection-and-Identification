import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 提取影像路径和标签
image_paths = data['image_path'].values
labels = data['label'].values

# 将标签转换为one-hot编码
labels = to_categorical(labels)

# 影像尺寸
IMG_SIZE = (224, 224)

def load_and_preprocess_image(img_path):
    # 加载影像
    img = load_img(img_path, target_size=IMG_SIZE)
    # 将影像转换为数组
    img_array = img_to_array(img)
    # 归一化
    img_array /= 255.0
    return img_array

# 加载所有影像
images = np.array([load_and_preprocess_image(img_path) for img_path in image_paths])

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 加载InceptionV3模型，并排除顶部的全连接层
base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 增加全局平均池化层
x = base_model_inception.output
x = GlobalAveragePooling2D()(x)

# 增加全连接层作为输出层
predictions = Dense(len(np.unique(data['label'])), activation='softmax')(x)

# 定义最终的模型
model_inception = Model(inputs=base_model_inception.input, outputs=predictions)

# 冻结所有卷积层的权重
for layer in base_model_inception.layers:
    layer.trainable = False

model_inception.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_inception.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# 解冻顶层的几层
for layer in base_model_inception.layers[-4:]:
    layer.trainable = True

# 重新编译模型，以较低的学习率进行微调
model_inception.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 继续训练模型
model_inception.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

# 评估模型在验证集上的性能
val_loss, val_accuracy = model_inception.evaluate(X_val, y_val)
print(f'Validation Loss (InceptionV3): {val_loss}')
print(f'Validation Accuracy (InceptionV3): {val_accuracy}')

# 保存模型
model_inception.save('my_trained_model_inception.h5')
