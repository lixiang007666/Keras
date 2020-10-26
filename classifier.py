import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
#optimizers.RMSprop 优化器采用 RMSprop，加速神经网络训练方法。
from keras.optimizers import RMSprop

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_train)
print(X_train.shape[0])
print(y_train.shape)
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (60,000, )

# data pre-processing
#参数-1就是不知道行数或者列数多少的情况下使用的参数
#reshape函数是对narray的数据结构进行维度变换，
#假设一个数据对象narray的总元素个数为N， 如果我们给出一个维度为（m，-1）时，我们就理解为将对象变换为一个二维矩阵，矩阵的第一维度大小为m，第二维度大小为N/m。
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   #normalize特征标准化
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your neural net
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

#接下来用 RMSprop 作为优化器，它的参数包括学习率等，可以通过修改这些参数来看一下模型的效果。
# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#优化器，可以是默认的，也可以是我们在上一步定义的。 损失函数，分类和回归问题的不一样，用的是交叉熵。 metrics，里面可以放入需要计算的 cost，accuracy，score 等。
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)