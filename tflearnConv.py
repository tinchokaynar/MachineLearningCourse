from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.models.dnn import DNN
import tflearn.datasets.mnist as mnist

x, y, test_x, test_y = mnist.load_data(one_hot=True)

x = x.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None, 28, 28, 1], name='input')
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')

convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', name='targets')

model = DNN(convnet)

# model.fit({'input': x}, {'targets': y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id='mnist')
# model.save('tflearn.model')
model.load('tflearn.model')
print(model.predict([test_x[2]]))
