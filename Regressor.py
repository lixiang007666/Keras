import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model

# create some data
X = np.linspace(-1, 1, 200)#start:返回样本数据开始点 stop:返回样本数据结束点 num:生成的样本数据量，默认为50 np.linspace主要用来创建等差数列
print(X)
np.random.shuffle(X)#random.shuffle(a)：用于将一个列表中的元素打乱
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, 200) #  正态分布 numpy.random.normal(loc=0.0, scale=1.0, size=None) loc均值 scale标准差 size输出值的维度。
print(Y)

# plot data
plt.scatter(X, Y)
plt.show()


X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points

#定义模型
model = Sequential()
#参数有两个，一个是输入数据和输出数据的维度，本代码的例子中 x 和 y 是一维的
model.add(Dense(output_dim=1, input_dim=1))
#如果需要添加下一个神经层的时候，不用再定义输入的纬度，因为它默认就把前一层的输出作为当前层的输入。在这个例子里，只需要一层就够了。

#激活模型
# choose loss function and optimizing method
#目标函数：mse->ce
#激活函数：sigmod—>relu
#优化（梯度下降方法）：->GD等->Adam
#避免过适应->Dropout
#层间连接：通常是全联接->形式多样：权值共享、跨层的反馈
model.compile(loss='mse', optimizer='sgd')

#训练的时候用 model.train_on_batch 一批一批的训练 X_train, Y_train。默认的返回值是 cost，每100步输出一下结果。
# training
print('Training -----------')
for step in range(301):
    #model.train_on_batch() 在训练集数据的一批数据上进行训练
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)


#save
print("test before save:\n")
print(model.predict(X_test[0:2]))
model.save("Reg_model.h5")
del model

#load
print("test after save:\n")
model=load_model("Reg_model.h5")
print(model.predict(X_test[0:2]))



# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test,batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()