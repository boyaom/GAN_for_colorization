import numpy as np
import torch
import cv2
import torch.nn as nn
# import torchvision.models as models
from model import NetG
from torch.autograd import Variable
from torch.utils.data import DataLoader
from train import MyDataSet

BATCHSIZE = 20
NUM_WORKERS = 0
# cuda = False
cuda = torch.cuda.is_available()
print(cuda)
device = torch.device("cuda:0" if cuda else "cpu")

if __name__ == '__main__':
	test_dataset = MyDataSet('X_test.pkl')
	test_num = len(test_dataset)
	print("Loading data...")
	test_dataloader = DataLoader(test_dataset, batch_size = BATCHSIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory = cuda)
	print("Done")
	with open('test.txt', 'r') as f:
		files = f.read().split('\n')
		f.close()
	
	G = NetG()
	# D = models.resnet18(pretrained=False)
	# D.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
	if cuda:
		G = G.cuda()
		# D = D.cuda()
	G.load_state_dict(torch.load('G_weights.pth'))
	# D.load_state_dict(torch.load('D_weights.pth'))
	for i, (y, uv) in enumerate(test_dataloader):
		yvar = Variable(y).to(device)
		# uvgen = Variable(uv).to(device)
		uvgen = G(yvar)
		gen_imgs = torch.cat([yvar, uvgen], dim = 1).cpu().detach().numpy()
		
		for j in range(y.size(0)):
			number = j+i*BATCHSIZE
			filename = 'datas/test/generated/' + files[number]
			print(filename, end="...")
			gen_img_yuv = gen_imgs[j,...].transpose(1, 2, 0) * 256
			gen_img_yuv = gen_img_yuv.astype(np.uint8)
			gen_img_bgr = cv2.cvtColor(gen_img_yuv, cv2.COLOR_YUV2BGR)
			cv2.imwrite(filename, gen_img_bgr)
			print("Done")