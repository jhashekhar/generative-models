import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

datasets = torchvision.datasets.CIFAR10(root='cifar10',
                                        train=True,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), ]))
dataloader = torch.utils.data.DataLoader(datasets, batch_size=16, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU())

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(32, 16, 4, stride = 2, padding = 1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
        nn.Sigmoid())

  def forward(self, x):
    z = self.encoder(x)
    out = self.decoder(z)
    return z, out


autoencoder = Autoencoder().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(autoencoder.parameters())

num_epochs = 25
for epoch in range(num_epochs):
  autoencoder.train()
  train_loss = 0.0
  for data in dataloader:
    img, _ = data
    img = img.to(device)
    optimizer.zero_grad()
    z, out = autoencoder(img)
    #print(out.shape, img.shape)
    loss = criterion(out, img)
    optimizer.zero_grad()
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  torchvision.utils.save_image(out.view(16, 3, 32, 32), 'sample_{}.png'.format(epoch), nrow=4, padding=2)
  print("Epoch [{}/{}]   loss: {}".format(epoch+1, num_epochs, train_loss/len(dataloader)))
