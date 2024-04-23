# pip install --force-reinstall "scikit-learn==1.0.2"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

print(data.head())
print(data.describe())
print(data.info())

plt.figure(figsize=(10, 6))
sns.distplot(data['PRICE'])
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data['PRICE'])
plt.title("Boxplot of House Prices")
plt.xlabel("Price")
plt.show()

correlation = data.corr()
print(correlation['PRICE'])

plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM', 'PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    plt.scatter(data[col], data['PRICE'], marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel("House prices in $1000")
plt.show()

X = data.drop(columns='PRICE')
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Linear Regression RMSE:", rmse)
print("Linear Regression R^2 Score:", r2)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
ann_viz(model, title="Neural Network Architecture")

history = model.fit(X_train, y_train, epochs=100, validation_split=0.05, verbose=0)

fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Scattergl(y=history.history['loss'], name='Train'), row=1, col=1)
fig.add_trace(go.Scattergl(y=history.history['val_loss'], name='Validation'), row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_yaxes(title_text="Loss", row=1, col=1)

fig.add_trace(go.Scattergl(y=history.history['mae'], name='Train'), row=1, col=2)
fig.add_trace(go.Scattergl(y=history.history['val_mae'], name='Validation'), row=1, col=2)
fig.update_xaxes(title_text="Epoch", row=1, col=2)
fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=2)

fig.update_layout(height=500, width=1000, title_text="Training History")
fig.show()
