import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# 读取Excel文件
file_path = '31-50組瑕疵辨認___楊漢澤___修改版.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'         # 替换为你的工作表名称
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 移除含有NaN标签的行
data = data.dropna(subset=['label'])

# 拆分特征和标签
X = data.drop(columns=['label'])
y = data['label']

# 查看原始数据的类别分布
print('原始数据集的类别分布:', Counter(y))

# 设置k_neighbors小于或等于少数类样本数减1
n_minority_samples = Counter(y).most_common()[-1][1]
k_neighbors = min(5, n_minority_samples - 1)

# 使用调整后的k_neighbors参数进行数据平衡
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 查看平衡后数据的类别分布
print('使用SMOTE后数据集的类别分布:', Counter(y_resampled))

# 将生成的新样本数据合并到原始数据中
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['label'] = y_resampled

# 只保留生成的新样本（因为原始样本已经包含在原始数据中）
new_samples = resampled_data[len(data):]

# 将新样本添加到原始数据中
final_data = pd.concat([data, new_samples], ignore_index=True)

# 保存最终数据到新的Excel文件
final_data.to_excel('balanced_data.xlsx', index=False)
