# PokemonGAN

## Use pokemon sprite data to create a GAN that generates new ones

In this script I created a pokemon generating GAN and let it run for 60 epochs, just for me to test how GANs work in a fun way.
To achieve this I used pytorch (I don't have a compatible GPU with cuda so I run the script in my CPU)\

The code below was used to view the images in the dataset, after they were resized and normalized so they could be fed in the models better.

```python

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
	view_image(img)
	break;

```
What I got from this is images like this one:
![](https://i.imgur.com/2slzYax.png)

## Generator

The generator model is the one that will be generating the fake data that will be provided to the discriminator so the latter can choose which
data is real and which one is fake. That way the Generator will better itself by trying to minimize it's loss function (make fake images indistinguishable
from real ones) and the discriminator will up his game by trying to maximize the Generator's loss function (by making correct choices). This is a minimax game.


```python
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
```
*5 layers of a combination of 3 sublayers were used.*

First a **Convolutional** layer the a batch normalizing layer with a Relu loss function. As we can see the generator starts
with a size 100 and kernel 4 and outputs a tensor sized 512. The next layer gets the 512 size tensor and outputs a 256, halving in each step
till the last layer produces the 3 channel tensor which is the produced Image. In the last Layer I used a Tanh activation function as per the 
literature.


## Discriminator

```python

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

```

The Discriminator acts exactly opposite from the Generator as it takes a 3 channel tensor and produces a 1 channel output of 0 or 1.
Depending on if it thinks the image is real or fake.

The weight initializers, loss functions and the training algorithm are fairly standard and are in the script

## Results

Here is a .gif file of the result of each of the 60 epochs
![](https://media.giphy.com/media/dZsMbLy9QILb7tfEeC/giphy.gif)
!https://media.giphy.com/media/dZsMbLy9QILb7tfEeC/giphy.gif
