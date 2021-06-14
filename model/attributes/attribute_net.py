import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class AttributeTrainer():
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
        img, colour, material, name, hor, ver, pos, bb = self.data
        img = img.to(self.device)
        colour = colour.to(self.device)
        material = material.to(self.device)
        name = name.to(self.device)
        hor = hor.to(self.device)
        ver = ver.to(self.device)
        pos = pos.to(self.device)
        bb = bb.to(self.device)

        pred = self.model(img)
        # print(pred)

        # print(pos_gt.type())
        # print(pos_pred.type())

        self.loss = self.criterion(
            pred,
            (
                colour,
                material,
                name,
                hor,
                ver,
                pos,
                bb
            )
        )

        self.get_acc(
            pred,
            (
                colour,
                material,
                name,
                hor,
                ver,
                pos,
                bb
            )
        )

        loss_value = self.loss.data.cpu().numpy().astype(float)
        self.loss.backward()
        self.optimiser.step()

        stats = None
        if print_debug:
            stats = self.get_acc(
                pred,
                (
                    colour,
                    material,
                    name,
                    hor,
                    ver,
                    pos,
                    bb
                )
            )

        return loss_value, stats

    def validate(self, loader):
        self.set_test()
        loss_cum = 0.0
        acc_dict_cum = {
            'colour': 0.0,
            'material': 0.0,
            'name': 0.0,
            'hor': 0.0,
            'ver': 0.0,
            'pos': 0.0,
            'bb': 0.0,
            'tot_acc': 0.0,
        }
        with torch.no_grad():
            for data in tqdm(loader):
                self.set_input(data)
                img, colour, material, name, hor, ver, pos, bb = self.data
                img = img.to(self.device)
                colour = colour.to(self.device)
                material = material.to(self.device)
                name = name.to(self.device)
                hor = hor.to(self.device)
                ver = ver.to(self.device)
                pos = pos.to(self.device)
                bb = bb.to(self.device)

                pred = self.model(img)

                acc_d = self.get_acc(
                    pred,
                    (
                        colour,
                        material,
                        name,
                        hor,
                        ver,
                        pos,
                        bb
                    )
                )

                for k, v in acc_d.items():
                    acc_dict_cum[k] += v * img.shape[0]

                loss = self.criterion(
                    pred,
                    (
                        colour,
                        material,
                        name,
                        hor,
                        ver,
                        pos,
                        bb
                    )
                )
                loss_value = loss.data.cpu().numpy().astype(float)
                loss_cum += loss_value * img.shape[0]

        loss_cum = loss_cum / len(loader.dataset)
        for k, v in acc_dict_cum.items():
            acc_dict_cum[k] = v / len(loader.dataset)
        final_dict = {}
        final_dict['acc_combined'] = acc_dict_cum['tot_acc'] + 1 - (acc_dict_cum['pos'] + acc_dict_cum['pos']) / 2
        for k, v in acc_dict_cum.items():
            final_dict[k] = v
        final_dict['loss'] = loss_cum

        return final_dict

    def get_acc(self, pred, gt):
        tot_corr = 0
        tot_num = 0
        colour_gt, material_gt, name_gt, hor_gt, ver_gt, pos_gt, bb_gt = gt
        colour_pred, material_pred, name_pred, hor_pred, ver_pred, pos_pred, bb_pred = pred

        _, colour_pred = torch.max(colour_pred, dim=1)
        colour_bool = colour_pred == colour_gt
        tot_corr += torch.sum(colour_bool)
        tot_num += torch.numel(colour_bool)
        colour_acc = torch.sum(colour_bool) / torch.numel(colour_bool)

        _, material_pred = torch.max(material_pred, dim=1)
        material_bool = material_pred == material_gt
        tot_corr += torch.sum(material_bool)
        tot_num += torch.numel(material_bool)
        material_acc = torch.sum(material_bool) / torch.numel(material_bool)

        _, name_pred = torch.max(name_pred, dim=1)
        name_bool = name_pred == name_gt
        tot_corr += torch.sum(name_bool)
        tot_num += torch.numel(name_bool)
        name_acc = torch.sum(name_bool) / torch.numel(name_bool)

        _, hor_pred = torch.max(hor_pred, dim=1)
        hor_bool = hor_pred == hor_gt
        tot_corr += torch.sum(hor_bool)
        tot_num += torch.numel(hor_bool)
        hor_acc = torch.sum(hor_bool) / torch.numel(hor_bool)

        _, ver_pred = torch.max(ver_pred, dim=1)
        ver_bool = ver_pred == ver_gt
        tot_corr += torch.sum(ver_bool)
        tot_num += torch.numel(ver_bool)
        ver_acc = torch.sum(ver_bool) / torch.numel(ver_bool)

        tot_acc = tot_corr / tot_num

        l1_pos = F.l1_loss(pos_pred, pos_gt)
        l1_bb = F.l1_loss(bb_pred, bb_gt)

        acc_dict = {
            'colour': colour_acc,
            'material': material_acc,
            'name': name_acc,
            'hor': hor_acc,
            'ver': ver_acc,
            'pos': l1_pos,
            'bb': l1_bb,
            'tot_acc': tot_acc,
        }

        return acc_dict

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
        # if self.optimiser is not None:
        #     if self.model.training and not zero_train:
        #         self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def __str__(self):
        return "Attribute extraction"


class AttributeModel(nn.Module):
    def __init__(self, img_enc):
        super(AttributeModel, self).__init__()
        self.img_enc = img_enc
        out_size = self.img_enc.out_size

        self.lin_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(out_size, 128),
            nn.ReLU(inplace=True),
        )

        self.colour_head = nn.Linear(128, 11)
        self.material_head = nn.Linear(128, 6)
        self.name_head = nn.Linear(128, 15)
        self.hor_head = nn.Linear(128, 2)
        self.ver_head = nn.Linear(128, 2)
        self.pos_head = nn.Linear(128, 3)
        self.bb_head = nn.Linear(128, 3)

    def forward(self, img):
        img_enc = self.img_enc(img)
        lin_out = self.lin_head(img_enc)

        col = self.colour_head(lin_out)
        mat = self.material_head(lin_out)
        name = self.name_head(lin_out)
        hor = self.hor_head(lin_out)
        ver = self.ver_head(lin_out)
        pos = self.pos_head(lin_out)
        bb = self.bb_head(lin_out)

        return col, mat, name, hor, ver, pos, bb
