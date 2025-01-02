import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss, crossview_contrastive_Loss, elbo
from scipy.optimize import linear_sum_assignment
import dataloader as loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def make_pseudo_label(model, device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    model.eval()
    scaler = MinMaxScaler()
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, _, hs = model.forward(xs)
        for v in range(view):
            hs[v] = hs[v].cpu().detach().numpy()
            hs[v] = scaler.fit_transform(hs[v])

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    new_pseudo_label = []
    for v in range(view):
        Pseudo_label = kmeans.fit_predict(hs[v])
        Pseudo_label = Pseudo_label.reshape(data_size, 1)
        Pseudo_label = torch.from_numpy(Pseudo_label)
        new_pseudo_label.append(Pseudo_label)

    return new_pseudo_label
def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long().to(device)
    new_y = new_y.view(new_y.size()[0])
    return new_y

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

        xrs, zs, _, _ = model(xs)
        optimizer.zero_grad()
        loss_list = []

        for v in range(view):
            loss_list.append(1*criterion(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))



def contrastive_train(epoch):
    tot_loss, rec_loss, cl_loss, L_shared, L_private, L_SL = 0., 0., 0., 0., 0., 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        loss_list = []
        xrs, zs, qs, ps = model.forward2(xs)

        loss_value, losses_A, losses_B = elbo(qs[0], ps[0], ps[1])
        temp = 0.01 * (loss_value + losses_A + losses_B)
        loss_list.append(temp)

        rec_loss = mes(xs[v], xrs[v])
        loss_list.append(rec_loss)

        cl_loss = 0.001 * crossview_contrastive_Loss(zs[0], zs[1], 9)
        loss_list.append(cl_loss)

        rec_loss += rec_loss.item()
        cl_loss += cl_loss.item()
        L_shared += L_shared.item()
        L_private += L_private.item()
        L_SL += L_SL.item()

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Training Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)),
          'rec_loss {}'.format(rec_loss),'cl_loss {}'.format(cl_loss),
          'L_shared {}'.format(L_shared), 'L_private {}'.format(L_private),'L_SL {}'.format(L_SL),
          )


def fine_tuning(epoch, new_pseudo_label):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    tot_loss = 0.
    cross_entropy = torch.nn.CrossEntropyLoss()
    for batch_idx, (xs, _, idx) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, qs, _ = model(xs)
        loss_list = []
        for v in range(view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = match(p, q)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    if len(data_loader) == 0:
        print('Last fine tuning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss ))
    else:
        print('Fine tuning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))




parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='0', type=int,
                    help='choiced of dataset: 0-BBCSport, 1-Reuters_dim10, 2-CCV, 3-MNIST-USPS'
                         '4-Caltech-2V, 5-Caltech-3V, 6-Caltech-4V, 7-Caltech-5V')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--temperature_f", type=float, default=0.5)
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--mse_epochs", type=int, default=100)
parser.add_argument("--con_epochs", type=int, default=100)
parser.add_argument("--tune_epochs", type=int, default=50)
parser.add_argument("--feature_dim", type=int, default=512)
parser.add_argument("--high_feature_dim", type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)



args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dataset = ['BBCSport', 'Reuters_dim10', 'Caltech101', 'MNIST_USPS', 'Caltech-2V', 'Caltech-3V', 'Caltech-4V', 'Caltech-5V', 'CCV', 'Caltech101-20', 'Scene-15', 'LandUse']
dataset = Dataset[args.dataset]

dataset, dims, view, data_size, class_num = loader.load_data(dataset)


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

setup_seed(args.seed)

model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

epoch = 1
while epoch <= args.mse_epochs:
    pretrain(epoch)
    epoch += 1
while epoch <= args.mse_epochs + args.con_epochs:
    contrastive_train(epoch)
    epoch += 1
while epoch <= args.mse_epochs + args.con_epochs + args.tune_epochs:
    fine_tuning(epoch, make_pseudo_label(model, device))
    if epoch == args.mse_epochs + args.con_epochs + args.tune_epochs:
        acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
        print(Dataset[args.dataset])
    epoch += 1

