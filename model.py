import tensorflow as tf


class Network:
    def __init__(self):
        # 创建两个占位符，x为输入网络的图像，y_为输入网络的图像类型
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])

        # 第一层卷积层
        # 初始化W为[5, 5, 1, 32]的张量，表示剪辑和大小为5x5，第一层网路的输入和输出神经元个数为1和32
        self.W_conv1 = self.weight_variable([5, 5, 1, 32])
        # 初始化b为[32]，即输出大小
        self.b_conv1 = self.bias_variable([32])

        # # 把输入x(二维张量，shape为[batch, 784])编程4d的x_image， x_image的shape应该是[batch, 29, 28, 1]
        # -1表示自动推测这个维度的size
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # 把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
        # h_pool1的输出即为第一层网络输出，shape为[batch, 14, 14, 1]
        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # 第二层卷积层
        # 卷积核大小依然为5x5，这层的输入和输出神经单元个数俄日32和64
        self.W_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv2 = self.weight_variable([64])

        # h_pool2即为第二层网络输出，shape为[batch， 7， 7， 1]
        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # 第三层全连接层
        # 这层是拥有1024个神经元的全连接层
        # W的第1维size为7x7x64， 7x7是h_pool2输出的size， 64是第2层输出神经元个数
        self.W_fc1 = self.weight_variable([7*7*64, 1024])
        self.b_fc1 = self.bias_variable([1024])

        # 计算前需要把第2层的输出reshape成[batch， 7x7x64]的张量
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout层
        # 为了减少过拟合，在输出层前加入dropout
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # 输出层
        # 最后，添加一个softmax层
        # 可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率
        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        # 预测值和真是只之间的交叉熵
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))

        # train op，使用ADAM优化器来做梯度下降，学习率为0.0001
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # 评估模型，tf.argmax能给出某个tensor对象在某一维度上数据最大值的索引
        # 因为标签是由0，1组成了one_hot vector，返回的索引就是数值为1的位置
        self.correct_predict = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))

        # 计算正确预测项的比例，因为tf.equal返回的是布尔值
        # 使用tf.cast把布尔值转化成浮点数，然后用tf.reduce_mean求平均值
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predict, "float"))

    # 权重初始化函数
    def weight_variable(self, shape):
        # 输出服从截尾正态分布的随机值
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 偏置初始化函数
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积op
    # x是一个4维张量，shape为[batch, height, width, channels]
    # 卷积核移动步长为1，填充类型为SAME可以不丢弃任何像素点
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # # 创建池化op
    # 采用最大池化，也就是取窗口中的最大值作为结果
    # x是一个4维张量，shape为[batch, height, width, channels]
    # ksize表示pool窗口大小为2x2，也就是高2宽2
    # strides表示在height和width维度上的步长都是2
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    

