import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class EffectorTracking():
    def __init__(self, model, loss, optimiser):
        self.model = model

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.criterion = loss

        if optimiser is not None:
            if optimiser['name'] == 'Adam':
                self.optimiser = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=optimiser['learning_rate'],
                    weight_decay=optimiser['weight_decay']
                )
            elif optimiser['name'] == 'SGD':
                self.optimiser = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=optimiser['learning_rate'],
                    weight_decay=optimiser['weight_decay']
                )
        else:
            self.optimiser = None

        self.data = None

    def train_step(self, print_debug=None):
        assert self.data is not None, "Set input first"
        self.set_train()
        self.optimiser.zero_grad()
        img, pos_gt = self.data
        img = img.to(self.device)
        pos_gt = pos_gt.to(self.device)

        pos_pred = self.model(img)

        # print(pos_gt.type())
        # print(pos_pred.type())

        self.loss = self.criterion(pos_pred, pos_gt)
        loss_value = self.loss.data.cpu().numpy().astype(float)
        self.loss.backward()
        self.optimiser.step()

        stats = None
        if print_debug:
            pass

        return loss_value, stats

    def validate(self, loader):
        self.set_test()
        loss_cum = 0.0
        with torch.no_grad():
            for data in tqdm(loader):
                self.set_input(data)
                img, pos_gt = self.data
                img = img.to(self.device)
                pos_gt = pos_gt.to(self.device)

                pos_pred = self.model(img)

                loss = self.criterion(pos_pred, pos_gt)
                loss_value = loss.data.cpu().numpy().astype(float)
                loss_cum += loss_value * img.shape[0]

        loss_cum = loss_cum / len(loader.dataset)
        return {
            'acc_sum': 1 - loss_cum,
            'loss': loss_cum
        }

    def get_prediction(self, img):
        return self.model(img)

    def set_input(self, data):
        self.data = data

    def unset_input(self):
        self.data = None

    def set_train(self):
        if not self.model.training:
            self.model.train()

    def set_test(self):
        if self.model.training:
            self.model.eval()

    def save_checkpoint(self, path, epoch=None, num_iter=None):
        checkpoint = {
            'state_dict': self.model.cpu().state_dict(),
            'epoch': epoch,
            'num_iter': num_iter
        }
        if self.model.training:
            checkpoint['optim_state_dict'] = self.optimiser.state_dict()
        torch.save(checkpoint, path)
        self.model = self.model.to(self.device)

    def load_checkpoint(self, path, zero_train=False):
        print("Loading checkpoint from:\t{}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimiser is not None:
            if self.model.training and not zero_train:
                self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def __str__(self):
        return "Effector position predictor"


class EffectorTrackingModel(nn.Module):
    def __init__(self, img_enc):
        super(EffectorTrackingModel, self).__init__()
        self.img_enc = img_enc
        out_size = self.img_enc.out_size

        self.lin_head = nn.Sequential(
            nn.Linear(out_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, img):
        img_enc = self.img_enc(img)
        pos_pred = self.lin_head(img_enc)

        return pos_pred
