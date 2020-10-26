# import numpy as np
# np.random.seed(1337)  # for reproducibility
# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Flatten
# from keras.optimizers import Adam
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train)
# X_train = X_train.reshape(-1,1,28,28)#如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
# X_test = X_test.reshape(-1,1,28,28)#彩色照片通道数有3 黑白只有1
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(X_train.shape)
# print(X_train)
#
# model=Sequential()
#
# #用卷积神经网络处理一组彩色图片时，Caffe/Theano 使用的数据格式是channels_first即：
# #（样本数，通道数，行数（高），列数（宽））
# #Tensforflow 使用的数据格式是channels_last即：
# #（样本数，行数（高），列数（宽），通道数）
# #输入
# model.add(Convolution2D(
#     batch_input_shape=(64, 1, 28, 28),
#     filters=32,
#     kernel_size=5,
#     strides=1,
#     padding='same',      # Padding method
#     data_format='channels_first',
# ))
# model.add(Activation('relu'))
#
# #output (32,28,28)
# model.add(MaxPooling2D(
#     pool_size=2,
#     strides=2,
#     padding='same',    # Padding method
#     data_format='channels_first',
# ))
#
# #output(32,14,14)
# model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
# #output(64,14,14)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# #output(64,7,7)
# #经过以上处理之后数据shape为（64，7，7），需要将数据抹平成一维，再添加全连接层1 64*64*7/3
#
# #全连接层1
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# #output(1024)
#
# #全连接层2 输出层
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
# adam=Adam(lr=1e-4)
#
# model.compile(optimizer=adam,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, nb_epoch=1, batch_size=64)
#
# loss,acc=model.evaluate(X_test,y_test)
# print(loss)
# print(acc)


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=1, batch_size=64,)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)







# 10000/10000 [==============================] - 43s 4ms/step
#
# test loss:  0.10048586780130864
#
# test accuracy:  0.9689