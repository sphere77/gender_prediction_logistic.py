import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 数据集
height = [158, 150, 160, 172, 175, 180]
weight = [50, 45, 51, 60, 62, 80]

X_train = np.array([[h, w] for h, w in zip(height, weight)])
y_train = np.array([0, 0, 0, 1, 1, 1])

# 2. 画图
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='crimson', label='女性', s=70)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='royalblue', label='男性', s=70)

plt.xlabel('身高 (cm)')
plt.ylabel('体重 (kg)')
plt.title('身高-体重 性别分布')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 3. 训练模型
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

# 4. 评估
y_train_pred = logreg.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

print("="*60)
print(f"训练集准确率：{train_accuracy * 100:.2f}%")
print(f"特征权重（身高、体重）：{logreg.coef_[0]}")
print(f"偏置项：{logreg.intercept_[0]}")
print("="*60)
print("输入身高体重进行预测，输入 q 退出")
print("="*60)

# 5. 循环预测
while True:
    height_input = input("请输入身高（cm）：")
    if height_input.lower() == 'q':
        print("程序退出")
        break

    try:
        height_num = float(height_input)
    except ValueError:
        print("输入错误，请输入数字\n")
        continue

    if not (50 <= height_num <= 250):
        print("身高必须在 50-250 cm\n")
        continue

    weight_input = input("请输入体重（kg）：")
    if weight_input.lower() == 'q':
        print("程序退出")
        break

    try:
        weight_num = float(weight_input)
    except ValueError:
        print("输入错误，请输入数字\n")
        continue

    if not (20 <= weight_num <= 300):
        print("体重必须在 20-300 kg\n")
        continue

    # 预测
    new_sample = [[height_num, weight_num]]
    pred = logreg.predict(new_sample)
    prob = logreg.predict_proba(new_sample)

    print("-"*60)
    print(f"输入：身高 {height_num} cm，体重 {weight_num} kg")
    print(f"预测性别：{'女性' if pred[0] == 0 else '男性'}")
    print(f"置信度：女性 {prob[0][0]*100:.2f}%，男性 {prob[0][1]*100:.2f}%")
    print("-"*60, "\n")