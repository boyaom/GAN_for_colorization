import cv2
import os
import numpy as np
import pickle as pkl

def pre_train(train_or_test):
	path = './datas/' + train_or_test + '/image'
	files = os.listdir(path)
	with open(train_or_test + '.txt', 'w') as f:
		f.write('\n'.join(files))
		f.close()
	files_path = [os.path.join(path, x).replace('\\', '/') for x in files]
	nums = len(files_path)
	X = []
	for i in range(nums):
		filename = files_path[i]
		img = cv2.imread(filename)
		img = cv2.resize(img, (256, 256))
		img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) #y=gray, u=0.436b-0.289g-0.147r, v=-0.100b-0.515g+0.615r
		img_yuv = img_yuv/256
		X.append(img_yuv)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray = 0.114*b + 0.587*g + 0.299*r
		cv2.imwrite(filename.replace('image', 'rgb'), img)
		cv2.imwrite(filename.replace('image', 'gray'), img_gray)
	X = np.array(X)
	with open('X_' + train_or_test + '.pkl', 'wb') as f:
		pkl.dump(X, f)

pre_train('train')
pre_train('test')