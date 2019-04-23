# 简介
本代码利用tensorflow复现MNIST手写检测

网络结构定义在model.py中

模型的训练定义在train.py中

模型加载与检测定义在predict.py中

tf.py为最初学习的时候以过程模式为主体写成的代码，网络结构定义训练显示准确度一气呵成，是神经网络里的豪杰（

后来为了能够在加载模型的时候复用网络结构定义代码段才将里面的定义和训练分别放在不同的类里

# 之后可能会添加的功能：
-加载一个模型并继续训练

-将tf模型转换为tflite格式并为其创建android应用程序
