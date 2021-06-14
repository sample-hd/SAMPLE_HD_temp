import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, gt, mask):
        # print(pred[0, 2210:, 0])
        mask = ~mask
        # print(pred.shape)
        # print(gt.shape)
        # print(mask.shape)
        # print(mask.shape[0] * mask.shape[1])
        # print(torch.sum(mask))
        # print(mask[3][:])
        # repeat_size = [1] * len(mask.shape) + [pred.shape[-1]]
        # print(repeat_size)
        mask_stack = torch.stack(pred.shape[-1] * [mask], dim=len(mask.shape))
        # print(mask_stack.shape)
        pred_masked = torch.masked_select(pred, mask_stack)
        # print(pred_masked[0:10])
        gt_masked = torch.masked_select(gt, mask_stack)
        # print(gt_masked[0:10])
        loss = F.l1_loss(pred_masked, gt_masked)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, gt, mask):
        # print(pred[0, 2210:, 0])
        mask = ~mask
        mask_stack = torch.stack(pred.shape[-1] * [mask], dim=len(mask.shape))
        pred_masked = torch.masked_select(pred, mask_stack)
        gt_masked = torch.masked_select(gt, mask_stack)
        loss = F.mse_loss(pred_masked, gt_masked)
        return loss


class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()

    def forward(self, pred, gt, seq_len):
        # print(pred[0, 2210:, 0])
        # gt = gt.transpose(1, 2)
        pred = pred.transpose(1, 2)
        # GT manipulation
        # print(gt.shape)
        # print(pred.shape)
        # print(gt[:, 0:20, :])
        gt[gt > 0.1] = 1
        gt[gt < -0.1] = -1
        gt[torch.logical_and(gt <= 0.1, gt >= -0.1)] = 0
        gt = (gt + 1).long()
        # print(gt.type())
        # pred_masked = torch.masked_select(pred, mask_stack)
        gt_masked = self.mask_indices(gt, seq_len)
        # print(gt_masked[:, 700:720, :])
        ce = F.cross_entropy(pred, gt_masked, ignore_index=-100,
                             weight=torch.Tensor([0.48, 0.04, 0.48]).to(gt.device))
        # print(ce)

        return ce

    def mask_indices(self, gt, seq_len):
        for i, slen in enumerate(seq_len):
            gt[i, slen:, :] = -100
        return gt


class TokenMaskLoss(nn.Module):
    def __init__(self, mask_factor, token_loss, mask_loss,
                 masked=False, token_limit=False, pos_weight=1.0):
        super(TokenMaskLoss, self).__init__()
        if pos_weight == 1.0:
            pos_weight = None
        else:
            pos_weight = torch.tensor(pos_weight)
        self.mask_factor = mask_factor
        if token_loss == 'NLL':
            self.token_l = nn.NLLLoss()
        if mask_loss == 'BCE':
            # self.mask_l = nn.BCELoss()
            self.mask_l = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.masked = masked
        self.token_limit = token_limit

    def forward(self, token_pred, token_gt, mask_pred, mask_gt,
                token_len=None, mask_bools=None):
        # print(token_pred.shape)
        # print(token_gt)
        # # exit()
        # print(mask_pred.shape)
        # print(mask_gt.shape)
        # print(mask_bools)
        # print(mask_gt.min())

        if self.token_limit:
            token_pred, token_gt = self.get_length_remap(token_pred, token_gt, token_len, masks=False)
            if not self.masked:
                mask_pred, mask_gt = self.get_length_remap(mask_pred, mask_gt, token_len, masks=True)
        if self.masked:
            mask_pred, mask_gt = self.get_mask_remap(mask_pred, mask_gt, mask_bools)
        # print(token_pred.shape)
        # print(token_gt)
        # print(mask_pred.shape)
        # print(mask_gt.shape)
        # print(mask_bools)
        tok_l = self.token_l(token_pred, token_gt)
        mask_l = self.mask_l(mask_pred, mask_gt)

        loss = tok_l + torch.tensor(self.mask_factor) * mask_l
        # print(tok_l)
        # print(mask_l)

        return loss

    def get_length_remap(self, pred, gt, lengths, masks=False):
        shape_idx = gt.shape[1]
        mask = torch.zeros((lengths.shape[0], shape_idx), dtype=torch.bool)
        for i, l in enumerate(lengths):
            # print(l)
            line_mask = torch.zeros((gt.shape[1]), dtype=torch.bool)
            line_mask[0:l] = True
            # print(line_mask)
            mask[i] = line_mask
        gt = gt[mask]
        if masks:
            pred = pred[mask]
        else:
            pred = pred.transpose(1, 2)[mask]

        return pred, gt

    def get_mask_remap(self, pred, gt, bools):
        # mask = torch.zeros_like(gt, dtype=torch.bool)
        # print(bools.shape)
        gt = gt[bools]
        pred = pred[bools]

        # print(gt.shape)
        # print(pred.shape)

        return pred, gt


class MSE_Multi(nn.Module):
    def __init__(self, c2_weight):
        super(MSE_Multi, self).__init__()
        self.c2_weight = c2_weight
        self.l_c1 = nn.MSELoss()
        self.l_c2 = nn.MSELoss()

    def forward(self, c1_pred, c1_gt, c2_pred, c2_gt):
        loss = self.l_c1(c1_pred, c1_gt) + self.c2_weight * self.l_c2(c2_pred, c2_gt)
        return loss


class AnnotationLoss(nn.Module):
    def __init__(self, pos_type, factor):
        super(AnnotationLoss, self).__init__()
        self.factor = factor
        if pos_type == 'L1':
            self.pos_loss = nn.L1Loss()
        elif pos_type == 'L2':
            self.pos_loss = nn.MSELoss()
        else:
            raise NotImplementedError()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_tup, gt_tup):
        ce = self.ce(pred_tup[0], gt_tup[0])
        ce += self.ce(pred_tup[1], gt_tup[1])
        ce += self.ce(pred_tup[2], gt_tup[2])
        ce += self.ce(pred_tup[3], gt_tup[3])
        ce += self.ce(pred_tup[4], gt_tup[4])
        ce = ce / 5
        pos = self.pos_loss(pred_tup[5], gt_tup[5])
        pos += self.pos_loss(pred_tup[6], gt_tup[6])
        pos = pos / 2
        loss = ce + self.factor * pos

        return loss




