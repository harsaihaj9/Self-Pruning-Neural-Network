
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)*0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_w = self.weight * gates
        return F.linear(x, pruned_w, self.bias)
    def gate_l1(self):
        return torch.sigmoid(self.gate_scores).sum()
    def sparsity(self, thr=1e-2):
        g = torch.sigmoid(self.gate_scores)
        return (g < thr).float().mean().item()*100, g.detach().cpu().flatten()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.fc1=PrunableLinear(3*32*32,512)
        self.fc2=PrunableLinear(512,256)
        self.fc3=PrunableLinear(256,10)
    def forward(self,x):
        x=self.flatten(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)
    def sparsity_loss(self):
        return self.fc1.gate_l1()+self.fc2.gate_l1()+self.fc3.gate_l1()
    def total_sparsity(self):
        vals=[]
        s=[]
        for l in [self.fc1,self.fc2,self.fc3]:
            sp,v=l.sparsity()
            s.append(sp); vals.append(v)
        return sum(s)/len(s), torch.cat(vals)

def train_eval(lmbda=1e-4, epochs=2, batch_size=128):
    tfm=transforms.Compose([transforms.ToTensor()])
    train_ds=datasets.CIFAR10(root='./data', train=True, download=True, transform=tfm)
    test_ds=datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=2)
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=2)
    model=Net().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for xb,yb in train_dl:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            out=model(xb)
            loss=F.cross_entropy(out,yb)+lmbda*model.sparsity_loss()
            loss.backward()
            opt.step()
    model.eval(); correct=tot=0
    with torch.no_grad():
        for xb,yb in test_dl:
            xb,yb=xb.to(device),yb.to(device)
            pred=model(xb).argmax(1)
            correct+=(pred==yb).sum().item(); tot+=yb.size(0)
    acc=100*correct/tot
    sp, vals = model.total_sparsity()
    return acc, sp, vals

if __name__=="__main__":
    lambdas=[1e-5,1e-4,1e-3]
    rows=[]; best_vals=None; best_acc=0
    for lam in lambdas:
        acc,sp,vals=train_eval(lam)
        rows.append([lam,acc,sp])
        if acc>best_acc:
            best_acc=acc; best_vals=vals
    df=pd.DataFrame(rows, columns=["Lambda","Test Accuracy","Sparsity Level (%)"])
    print(df)
    df.to_csv("results.csv", index=False)
    plt.hist(best_vals.numpy(), bins=50)
    plt.title("Gate Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
