#  GAN_for_colorization

**使用GAN训练实现黑白图片上色**

**python工具包**
>任意版本均可
>pytorch
>numpy
>cv2

**数据集**

*[数据集地址](http://www.seeprettyface.com/mydataset_page3.html "数据集地址")，选择其中任意即可，也可从其他位置获取。解包后人工分出训练集和测试集的图片，将训练集的图片保存到datas/train/image下，测试集保存到datas/test/image下*

**使用方法**

*依次运行pre_train.py、train.py、test.py，结果会保存在datas/test/generated下*
