import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def loss_fn(x, means_x, means_z, log_var_z):
    BCE = torch.nn.functional.binary_cross_entropy(means_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var_z - means_z.pow(2) - log_var_z.exp())

    return (BCE + KLD) / x.size(0)

def transformation(x):
    PIL_tensor=transforms.ToTensor()
    return PIL_tensor(x).reshape(-1,1).squeeze()

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(encoder,self).__init__()
        self.l1=nn.Linear(input_dim, hidden_dim)
        self.linear_means = nn.Linear(hidden_dim, output_dim)
        self.linear_log_var = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x=self.l1(x)
        x=F.relu(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

class decoder (nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(decoder,self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = self.l1(z)
        z = F.relu(z)
        z = self.l2(z)
        out=torch.sigmoid(z)
        return out

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder=encoder(input_dim, hidden_dim, output_dim)
        self.decoder=decoder(output_dim, hidden_dim, input_dim)
        self.hidden_dim=hidden_dim

    def forward(self,x):
        means_z, log_var_z = self.encoder(x)
        z = self.sample_z(means_z, log_var_z)
        means_x= self.decoder(z)
        return means_x, means_z, log_var_z

    def sample_z(self,means, log_var):
        # log_var = log(std^2) = 2*log(std)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means+eps*std

BATCH_SIZE=1000
EPOCHS=75
INPUT_DIM=28*28
HIDDEN_DIM=100
OUTPUT_DIM=28*28
LEARNING_RATE=1e-3

writer=SummaryWriter()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transformation)

train_loader= torch.utils.data.DataLoader(dataset=mnist_trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader= torch.utils.data.DataLoader(dataset=mnist_testset, batch_size=BATCH_SIZE, shuffle=False)

model=VAE(INPUT_DIM,HIDDEN_DIM,OUTPUT_DIM)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
i=0

images=[]
for epoch in range(EPOCHS):
    print("epoch : ", epoch)
    for x,y in train_loader :
        i+=1
        # x.shape : BATCH_SIZE 784
        # y.shape : BATCH_SIZE
        # plt.imshow(x[0,0,:,:])

        # print(torch.max(x))
        # print(x.shape)
        # print(x[0,0,:,:])

        means_x, means_z, log_var_z=model(x)
        loss=loss_fn(x,means_x, means_z, log_var_z)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    writer.add_scalars("Loss", {"Train loss": loss}, i)

    for x, y in test_loader:
        means_x, means_z, log_var_z = model(x)
        loss = loss_fn(x, means_x, means_z, log_var_z)

    writer.add_scalars("Loss", {"Test loss": loss}, i)

    if epoch%5==0 :
        if epoch==0:
            images.append(x[0,:])
        else :
            images.append(means_x[0,:])

for j in range(len(images)):
    plt.subplot(3,5,j+1)
    plt.imshow(images[j].reshape((28,28)).detach().numpy())
    plt.axis('off')

plt.show()



