#labels (i = arrival_delay)
# 0; <0 
# 1; 0<i<20
# 2; 20<i<40
# 3; 40<i<60
# 4; i <= 60
df.label

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing


# Preprocess the data
train = pd.read_csv('flight-delays-prediction-challeng2021/flights_train.csv')

df = pd.concat([pd.get_dummies(train['DAY_OF_WEEK']),
           train[['SCHEDULED_TIME', 'ARRIVAL_DELAY']]], axis=1)
df.columns = ['DAY_1', 'DAY_2', 'DAY_3', 'DAY_4', 'DAY_5', 'DAY_6',
              'DAY_7', 'SCHEDULED_TIME', 'ARRIVAL_DELAY']

min_max_scaler = preprocessing.MinMaxScaler()
df['SCHEDULED_TIME_NORM'] = min_max_scaler.fit_transform(df.SCHEDULED_TIME.values.reshape(-1,1))

def labels(arrival_delay):
       labels = []
       for i in arrival_delay:
              if i < 0:
                     labels.append(0)
              elif i < 20:
                     labels.append(1)
              elif i < 40:
                     labels.append(2)
              elif i < 60:
                     labels.append(3)
              else:
                     labels.append(4)
       return labels

df['labels'] = labels(df.ARRIVAL_DELAY)
del df['SCHEDULED_TIME'], df['ARRIVAL_DELAY']

x_train = df.iloc[:,:9]
y_train = x_train.pop('labels').values
x_train = x_train.values

# Create validation set
x_val = x_train[-100000:]
y_val = y_train[-100000:]
x_train = x_train[:-100000]
y_train = y_train[:-100000]
del df



#Define model
inputs = tf.keras.Input(shape=(8,))
x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(10, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)


# compile model
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# fit model
model.fit(
    x_train,
    y_train,
    batch_size=1000, 
    epochs=4,
    validation_data=(x_val, y_val),
) # len(x_train)

prediction = tf.argmax(model(x_train), axis = 1).numpy()


from collections import Counter
Counter(y_train)  # Counter({0: 1598269, 1: 660536, 2: 198383, 4: 169749, 3: 91616})
Counter(prediction)










