from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from model import Net

def train(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss=0
    train_acc=0
    for X,y in dataloader:

        y_pred =model(X)

        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val(model: nn.Module(),
         dataloader: DataLoader,
         loss_fn: nn.Module,
         ):
    model.eval()
    val_loss ,val_acc = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            val_acc = (y_pred_class == y).sum().item()/len(y_pred_class)
        val_loss = val_loss/len(dataloader)
        val_acc = val_acc/len(dataloader)
    return val_loss, val_acc

def main (Model: nn.Module,
         train_dataloader: DataLoader,
         val_dataloader: DataLoader,
         optimizer: torch.optim.Optimizer,
         loss_fn: nn.Module = nn.CrossEntropyLoss(),
         epochs: int = 5
         ):
    result = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    for i in range(epochs):
        train_loss, train_acc = train(Model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = val(Model, val_dataloader, loss_fn)
        print(
            f"Epoch: {i + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )
        result['train_loss'].append(train_loss)
        result['val_loss'].append(val_loss)
        result['train_acc'].append(train_acc)
        result['val_acc'].append(val_acc)
    return result
if __name__ == '__main__':
    train_path = r'C:\Users\admin\Downloads\Data\train'
    val_path = r'C:\Users\admin\Downloads\Data\val'
    train_trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((23, 100))
        ]

    )
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_trans
    )
    val_dataset = datasets.ImageFolder(
        root=val_path,
        transform=train_trans
    )
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    model_1 =Net(num_class=5)
    Optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.01)
    main(model_1,train_dataloader, val_dataloader, Optimizer,loss_fn= nn.CrossEntropyLoss(), epochs=20)




