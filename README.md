1. cmake时，发现官网文档与当前github代码不匹配。使用低版本，才能成功cmake与make
2. mnnconvert(python)，安装时，windows的0.0.6不能正常使用。在Linux中pip install mnnconvert-0.0.7才能正常转换
3. MNN的Input维度不能>=5，并且在整个.mnn模型中不能有维度>=5的操作。
为了方便部署，我的pb模型中包含了数据预处理和后处理（转为真实框，score分数阈值，iou阈值），其中有维度(-1,13,13,1,cls+5)，MNN提示这里有错误
4. 为了验证mnn可行性，我将后处理与预处理拆开，.mnn文件中包含模型输入（1，416，416，3），输出（"conv2d_10/BiasAdd","conv2d_13/BiasAdd"）,
使用MNN验证工具无异常，但是使用时无报错，无结果。使用clock发现网络并没有正常运行。

附：使用过程中，使用了tensorflow中tf.image.pad_to_bounding_box，freeze之后多了很多莫名的输入输出。转换成mnn之后报错，于是查看其实现过程，使用
tf.pad代替，实现功能
