from numpy.lib.npyio import load
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

#数据预处理
def load_data_npz(path="mnist.npz"):
    f = np.load(path)
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f['x_test'], f['y_test']
    f.close()
    return x_train,y_train,x_test,y_test

X_train, Y_train, X_test, Y_test = load_data_npz()
print("x_train:{}".format(X_train.shape))
print("y_train:{}".format(Y_train.shape))
print("x_test:{}".format(X_test.shape))
print("y_test:{}".format(Y_test.shape))

print(X_train[0],Y_train[0])

train_x,train_y = [],[]
for i in range(len(X_train)):
    temp = X_train[i].astype(np.float32)/255
    temp = temp.reshape(28,28,1)
    train_x.append(temp),train_y.append(Y_train[i])
test_x,test_y = [],[]
for i in range(len(X_test)):
    temp = X_test[i].astype(np.float32)/255
    temp = temp.reshape(28,28,1)
    test_x.append(temp),test_y.append(Y_test[i])

train_x = (np.array(train_x))
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_x = (np.array(test_x))
test_y = tf.keras.utils.to_categorical(test_y, 10)
print("img handled!")
print(test_x.shape)


#模型本身
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="Input", input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="PreHandle"))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3), activation="relu", padding="same", name="Handle"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="MaxPooling"))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=(5,5), activation="relu", padding="same", name="Final"))
model.add(tf.keras.layers.Flatten(name="Flatten"))
model.add(tf.keras.layers.Dense(100, activation="relu", name="FullChain"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation="softmax", name="Result"))
model.summary()

#加载模型
model.load_weights("Model.h5")
print("model loaded!")


#打开这个用来做训练
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./Model.h5",
                                       monitor='accuracy',
                                       mode='max',
                                       save_best_only=True
                                       ),
	    tf.keras.callbacks.TensorBoard(update_freq='batch')
        ]


#编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), #学习率；learning_rate = 0.001默认
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#打开这个用来做训练
history = model.fit(train_x,train_y,
                     batch_size=4,
                     epochs=10,
                     validation_split=0.3, #获取20%来做反馈训练
                     callbacks=callbacks
                     )

#用来做测试以及最终预测的
loss,accuracy = model.evaluate(test_x, test_y, verbose=0)
predictions = model.predict(test_x, batch_size=4)

#显示部分
col = 10
row = 10
for i in range(col):
    for j in range(row):
        plt.subplot(col,row,i*col+j+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(np.argmax(predictions[col*i+j]))
        plt.subplots_adjust(wspace=0,hspace=1)
        plt.imshow(X_test[i*col+j], cmap="gray")
plt.show()

#显示下一个batch的100个数字和识别结果
# for i in range(col):
#     for j in range(row):
#         plt.subplot(col,row,i*col+j+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.title(np.argmax(predictions[col*i+j+100]))
#         plt.subplots_adjust(wspace=0,hspace=1)
#         plt.imshow(X_test[i*col+j+100], cmap="gray")
# plt.show()

print('test loss', loss)
print('test accuracy', accuracy)