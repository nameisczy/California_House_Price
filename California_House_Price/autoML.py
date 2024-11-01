import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

# 导入数据
train_data = pd.read_csv("kaggle/California_House_Price/train.csv")
test_data = pd.read_csv("kaggle/California_House_Price/test.csv")

# 数据预处理
# 去掉无关列
train_data = train_data.drop(columns=['Id'])
test_data = test_data.drop(columns=['Id'])

# 处理日期
train_data['Last Sold On'] = pd.to_datetime(train_data['Last Sold On'], format='%Y-%m-%d')
train_data['Listed On'] = pd.to_datetime(train_data['Listed On'], format='%Y-%m-%d')
test_data['Last Sold On'] = pd.to_datetime(test_data['Last Sold On'], format='%Y-%m-%d')
test_data['Listed On'] = pd.to_datetime(test_data['Listed On'], format='%Y-%m-%d')

train_data['Last Sold Year'] = train_data['Last Sold On'].dt.year
train_data['Last Sold Month'] = train_data['Last Sold On'].dt.month
train_data['Last Sold Day'] = train_data['Last Sold On'].dt.day
train_data['Listed Year'] = train_data['Listed On'].dt.year
train_data['Listed Month'] = train_data['Listed On'].dt.month
train_data['Listed Day'] = train_data['Listed On'].dt.day

test_data['Last Sold Year'] = test_data['Last Sold On'].dt.year
test_data['Last Sold Month'] = test_data['Last Sold On'].dt.month
test_data['Last Sold Day'] = test_data['Last Sold On'].dt.day
test_data['Listed Year'] = test_data['Listed On'].dt.year
test_data['Listed Month'] = test_data['Listed On'].dt.month
test_data['Listed Day'] = test_data['Listed On'].dt.day

train_data = train_data.drop(['Last Sold On', 'Listed On'], axis=1)
test_data = test_data.drop(['Last Sold On', 'Listed On'], axis=1)

# 用AutoGluon训练
predictor = TabularPredictor(label='Sold Price').fit(train_data, presets="best_quality", ag_args_fit={'num_gpus': 1})

# 预测
predictions = predictor.predict(test_data)

# 保存结果
submission = pd.DataFrame({'Id': test_data.index, 'Sold Price': predictions})
submission.to_csv('submission.csv', index=False)

# 可视化训练日志
predictor.fit_summary()

# 可视化预测结果
plt.plot(predictions, label='Predictions')
plt.xlabel('Index')
plt.ylabel('Sold Price')
plt.legend()
plt.show()
