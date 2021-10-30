#  GAN_for_colorization

**使用GAN训练实现黑白图片上色**

大概是写完了，大概率有BUG


**python工具包**
python = 3.6
pytorch
numpy 
cv2

**使用前**

主目录下新建文件夹datas
─datas
│  ├─test
│  │  ├─generated
│  │  ├─gray
│  │  ├─image
│  │  └─rgb
│  └─train
│      ├─gray
│      ├─image
│      └─rgb


**数据集**

[数据集地址](http://www.seeprettyface.com/mydataset_page3.html "数据集地址")，选择其中任意即可。解包后人工分出训练集和测试集的图片，将训练集的图片保存到datas/train/image下，测试集保存到datas/test/image下

**使用方法**

依次运行pre_train.py、train.py、test.py

##readme待补充##
