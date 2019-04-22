import tensorflow as tf
from model import Network
import numpy as np
import cv2


class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.ckpt_dir = 'ckpt'
        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("未保存任何模型")

    def predict(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        flatten_img = np.reshape(img, 784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x, self.net.keep_prob: 1.0})

        print(image_path)
        print('     -> Predit digit', np.argmax(y[0]))


if __name__ == "__main__":
    app = Predict()
    app.predict('test_images\\0.png')
