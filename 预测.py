import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 加载模型
model = tf.keras.models.load_model('my_trained_model.h5')

def load_and_preprocess_image(img_path):
    # 加载影像
    img = load_img(img_path, target_size=(224, 224))
    # 将影像转换为数组
    img_array = img_to_array(img)
    # 归一化
    img_array /= 255.0
    return img_array

# 加载并预处理新影像
new_image_path = 'path/to/new/image.jpg'
new_image = load_and_preprocess_image(new_image_path)

# 进行预测
prediction = model.predict(np.expand_dims(new_image, axis=0))
predicted_class = np.argmax(prediction, axis=1)

# 打印预测结果
print(f'Predicted class: {predicted_class[0]}')
