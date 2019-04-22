import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import Network


class Train:
    def __init__(self):
        self.net = Network()
        self.mnist = input_data.read_data_sets("MNIST_data\\", one_hot=True)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver(max_to_keep=1)

    def train(self):
        step = 15000

        for i in range(step):
            batch = self.mnist.train.next_batch(50)
            if i % 100 == 0:
                # 每100次输出一次日志
                train_accuracy = self.net.accuracy.eval(feed_dict={
                    self.net.x: batch[0], self.net.y_: batch[1], self.net.keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.net.train_step.run(feed_dict={
                self.net.x: batch[0], self.net.y_: batch[1], self.net.keep_prob: 0.5})
        print("test accuracy %g" % self.net.accuracy.eval(feed_dict={
            self.net.x: self.mnist.test.images, self.net.y_: self.mnist.test.labels, self.net.keep_prob: 1.0}))


if __name__ == "__main__":
    app = Train()
    app.train()
