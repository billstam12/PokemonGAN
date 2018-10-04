import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#Generate the data from the dataset in my folder

def pokemon_data():
	compose = transforms.Compose(
		[transforms.Resize(64),
		 transforms.CenterCrop(64),
		 transforms.ToTensor(),
		 transforms.Normalize((.5, .5, .5), (.5, .5, .5))
		])
	out_dir = 'mgan_dataset'
	return datasets.ImageFolder(root=out_dir,transform=compose)

data = pokemon_data()
poke_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
# Num batches
num_batches = len(poke_loader)

def view_image(img):
	single_image = img[0][0]
	single_image= (single_image*0.5)+0.5
	single_image = single_image.clamp(0,1)
	single_image = single_image.numpy()
	
	# move the dimensions around to get them right
	single_image = np.transpose(single_image, (1, 2,0))
   
	# plot image
	print('image size:' ,single_image.shape)
	plt.imshow(single_image)
	plt.axis('off')
	plt.show()


#print a single image
for (img) in poke_loader:
	#view_image(img)
	break;


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class G(nn.Module):
	def __init__(self):
		super(G, self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
			nn.Tanh()
		)

	def forward(self, input):
		output = self.main(input)
		return output

netG = G() 
netG = netG.apply(weights_init)



class D(nn.Module):
	def __init__(self):
		super(D, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, 1, bias = False),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(64, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(128, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(256, 512, 4, 2, 1, bias = False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(512, 1, 4, 1, 0, bias = False),
			nn.Sigmoid()
		)

	def forward(self, input):
		output = self.main(input)
		return output.view(-1)	


netD = D() 
netD = netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.99))

import matplotlib.pyplot as plt

def denorm_monsters(x):
    renorm = (x*0.5)+0.5
    return renorm.clamp(0,1)

imm = 0
def plot_figure(noise):
	global imm
	plt.figure();
	imgs = netG(noise);
	result = denorm_monsters(imgs.data)
	result = make_grid(result)
	result = transforms.Compose([transforms.ToPILImage()])(result)
	plt.imshow(result)
	plt.axis('off')

	imm += 1
	savefile = "results/im" + str(imm) + '.jpg'  
	plt.savefig(savefile)

fixed_noise = Variable(torch.randn(64, 100, 1, 1))
plot_figure(fixed_noise)

for epoch in range(100):
	for i, poke in enumerate(tqdm(poke_loader,0)):
		netD.zero_grad()
		#Get real image
		real, _ = poke
		input = Variable(real)
		target = Variable(torch.ones(input.size()[0]))
		output = netD(input)
		errD_real = criterion(output, target)
		errD_real.backward()

		noise = Variable(torch.randn(input.size()[0], 100, 1, 1))

		
		fake = netG(noise)
		target = Variable(torch.zeros(input.size()[0]))
		output = netD(fake.detach())
		errD_fake = criterion(output, target)   
		errD_fake.backward()

		optimizerD.step()

		
		netG.zero_grad()
		target = Variable(torch.ones(input.size()[0]))
		output = netD(fake)
		errG = criterion(output, target)
		errG.backward()
		optimizerG.step()
		
	print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 100, i, len(poke_loader), errD_real.data[0], errG.data[0]))
	plot_figure(fixed_noise)

