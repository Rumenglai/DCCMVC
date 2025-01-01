import torch
import torch.nn as nn
import probtorch
from torch.nn.functional import normalize
EPS = 1e-9

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.zPrivate_dim = 50
        self.zShared_dim = 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        self.fc = nn.Linear(512, 2 * (self.zPrivate_dim + self.zShared_dim)+8)

    def forward(self, x):
        return self.encoder(x)
    def forward2(self, x):
        hiddens = self.encoder(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :self.zPrivate_dim]
        logvarPrivate = stats[:, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[:, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q_private = torch.distributions.Normal(muPrivate, stdPrivate)
        q_shared = torch.distributions.Normal(muShared, stdShared)

        return {'privateA': q_private, 'sharedA': q_shared}, hiddens

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.zPrivate_dim = 50
        self.zShared_dim = 10
        self.dec_hidden = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
                        nn.Linear(feature_dim, 2000),
                        nn.ReLU(),
                        nn.Linear(2000, 500),
                        nn.ReLU(),
                        nn.Linear(500, 500),
                        nn.ReLU(),
                        nn.Linear(500, input_dim)
                    )
        self.dec_image = nn.Linear(500, input_dim)

    def forward(self, x):
        return self.decoder(x)
    def forward2(self, z_private, z_shared, x):
        hiddens = self.dec_hidden(torch.cat([z_private, z_shared], -1))
        images_mean = self.dec_image(hiddens).squeeze(0)

        p = probtorch.Trace()
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
               images_mean, x, name='x')

        return p, images_mean

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = nn.ModuleList([
            Encoder(input_size[v], feature_dim).to(device) for v in range(view)
        ])
        self.decoders = nn.ModuleList([
            Decoder(input_size[v], feature_dim).to(device) for v in range(view)
        ])

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim * view, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)

            zs.append(z)
            hs.append(h)
            qs.append(q)
            xrs.append(xr)
        return xrs, zs, qs, hs

    def forward2(self, xs):
        xrs = []
        qs = []
        zs = []
        ps = []
        for v in range(self.view):
            x = xs[v]
            q, z = self.encoders[v].forward2(x)
            p, xr = self.decoders[v].forward2(q['privateA'].sample(), q['sharedA'].sample(), x)
            zs.append(z)
            xrs.append(xr)
            qs.append(q)
            ps.append(p)
        return xrs, zs, qs, ps

    def forward_cluster(self, xs):
        qs = []
        preds = []
        cat = torch.tensor([]).cuda()
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            cat = torch.cat((cat, z), 1)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)

        cat_pre = self.head(cat)
        return qs, preds, cat_pre
