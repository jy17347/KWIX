import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

actions = [
          'legraise',
          'dumbbell',
          'babel']

path_dir = './created_dataset'
folder_list = os.listdir(path_dir)

data = np.concatenate([
    np.load(path_dir + '/' + folder_list[3]),
    np.load(path_dir + '/' + folder_list[4]),
    np.load(path_dir + '/' + folder_list[5])
], axis=0)

print(data.shape)

x_data = data[:,:,:-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)

y_data = to_categorical(labels, num_classes=len(actions)) #onehot encoding

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2022)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


model = Sequential([
    LSTM(64, activation='relu',return_sequences=True, input_shape=x_train.shape[1:3]),
    LSTM(128, activation='relu',return_sequences=True,),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(actions), activation='softmax')    
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)