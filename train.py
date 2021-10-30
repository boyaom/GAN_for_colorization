import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torchvision.models as models
from model import NetG
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

BATCHSIZE = 20
EPOCH = 200
NUM_WORKERS = 0
PIXEL_WISE_WEIGHT = 1000.0
G_SPEED = 1
CHECK_POINT = 100
LR_G = 0.001
LR_D = 0.001
# cuda = False
cuda = torch.cuda.is_available()
print(cuda)
device = torch.device("cuda:0" if cuda else "cpu")
adversarial_loss = nn.BCELoss()

class MyDataSet(Dataset):
	def __init__(self, file):
		with open(file, 'rb') as f:
			self.X = pkl.load(f)
	def __len__(self):
		return len(self.X)
	def __getitem__(self, idx):
		y = self.X[idx,...,0]
		u = self.X[idx,...,1]
		v = self.X[idx,...,2]
		return torch.Tensor(y[np.newaxis,:]), torch.Tensor(np.stack([u,v],axis=0))

if __name__ == '__main__':
	
	train_dataset = MyDataSet('X_train.pkl')
	train_num = len(train_dataset)
	print("Loading data...")
	train_dataloader = DataLoader(train_dataset, batch_size = BATCHSIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory = cuda)
	print("Done")
	G = NetG()
	D = models.resnet18(pretrained=False)
	D.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
	
	if cuda:
		G = G.cuda()
		D = D.cuda()
		adversarial_loss = adversarial_loss.cuda()
	optimizer_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))

	print("Training...")
	i = 0
	for epoch in range(EPOCH):
		for y, uv in train_dataloader:
			try:
				print(i)
				valid = Variable(torch.Tensor(y.size(0), 1).fill_(1.0), requires_grad=False).to(device)
				fake = Variable(torch.Tensor(y.size(0), 1).fill_(0.0), requires_grad=False).to(device)
				
				yvar = Variable(y).to(device)
				uvvar = Variable(uv).to(device)
				real_imgs = torch.cat([yvar, uvvar], dim = 1)
				
				optimizer_G.zero_grad()
				uvgen = G(yvar)
				gen_imgs = torch.cat([yvar.detach(), uvgen], dim = 1)
				
				g_loss_gan = adversarial_loss(D(gen_imgs), valid)
				g_loss = g_loss_gan + PIXEL_WISE_WEIGHT * torch.mean((uvvar - uvgen)**2)
				
				if i % G_SPEED == 0:
					g_loss.backward()
					optimizer_G.step()
				
				optimizer_D.zero_grad()
				
				real_loss = adversarial_loss(D(real_imgs), valid)
				fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
				d_loss = (real_loss + fake_loss) / 2
				d_loss.backward()
				optimizer_D.step()
				i+= 1
				if i % CHECK_POINT == 0:
					print("Epoch: %d: [D loss: %f] [G total loss: %f] [G GAN loss: %f]" % (epoch, d_loss.item(), g_loss.item(), g_loss_gan.item()))
					torch.save(D.state_dict(), 'D_weights_' + str(epoch) + '.pth')
					torch.save(G.state_dict(), 'G_weights_' + str(epoch) + '.pth')
			except KeyboardInterrupt:
				torch.save(D.state_dict(), 'D_weights.pth')
				torch.save(G.state_dict(), 'D_weights.pth')
	torch.save(D.state_dict(), 'D_weights.pth')
	torch.save(G.state_dict(), 'D_weights.pth')
	print("Done")