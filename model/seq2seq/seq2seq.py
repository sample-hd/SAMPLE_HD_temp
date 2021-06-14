import torch
import torch.nn as nn


class Seq2seqTrainer():
    def __init__(self, model, loss, optimiser):
        self.model = model

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
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
        instr, ins_len, prog, prog_len = self.data
        instr = instr.to(self.device)
        prog = prog.to(self.device)

        token_pred = self.model(instr, prog, ins_len)

        prog_gt = prog[:, 1:]
        prog_len = prog_len - 1
        # print(prog_gt.shape)
        token_pred = token_pred[:, :-1, :].transpose(1, 2)
        # print(token_pred.shape)
        self.loss = self.criterion(token_pred, prog_gt)
        loss_value = self.loss.data.cpu().numpy().astype(float)
        self.loss.backward()
        self.optimiser.step()
        stats = None
        if print_debug:
            stats = {}
            _, token_idx = torch.max(token_pred, 1)
            # print(token_idx.shape)
            # print(token_idx)
            bool_tokens = token_idx == prog_gt
            acc_with_null = torch.sum(bool_tokens) / torch.numel(bool_tokens)
            stats['acc_w_null'] = acc_with_null
            len_mask = torch.zeros_like(bool_tokens, dtype=torch.bool)
            for i in range(len_mask.shape[0]):
                len_mask[i, 0:prog_len[i]] = True
            acc_without_null = torch.sum(bool_tokens[len_mask]) / torch.numel(bool_tokens[len_mask])
            stats['acc_wo_null'] = acc_without_null

        return loss_value, stats

    def validate(self, loader):
        self.set_test()
        loss_cum = 0.0
        acc_wo_cum = 0.0
        acc_wo_cum_sample = 0.0
        acc_w_cum = 0.0
        acc_w_cum_sample = 0.0
        no_null_no_sum = 0
        acc_prog_lvl_cum = 0.0
        acc_prog_lvl_cum_sample = 0.0
        sum_corr_sample = 0.0
        with torch.no_grad():
            end_tok = loader.dataset.vocab["program_token_to_idx"]["<END>"]
            for data in loader:
                self.set_input(data)
                instr, ins_len, prog, prog_len = self.data
                instr = instr.to(self.device)
                # ins_len = ins_len.to(self.device)
                instr = instr.to(self.device)
                prog = prog.to(self.device)

                token_pred = self.model(instr, prog, ins_len)

                prog_gt = prog[:, 1:]
                prog_len = prog_len - 1
                # print(prog_gt.shape)
                token_pred = token_pred[:, :-1, :].transpose(1, 2)

                loss = self.criterion(token_pred, prog_gt)
                _, token_idx = torch.max(token_pred, 1)
                bool_tokens = token_idx == prog_gt
                acc_with_null = torch.sum(bool_tokens) / torch.numel(bool_tokens)
                acc_w_cum += acc_with_null * instr.shape[0]

                len_mask = torch.zeros_like(bool_tokens, dtype=torch.bool)
                for i in range(len_mask.shape[0]):
                    len_mask[i, 0:prog_len[i]] = True
                acc_without_null = torch.sum(bool_tokens[len_mask]) / torch.numel(bool_tokens[len_mask])
                no_null_no = torch.numel(bool_tokens[len_mask])
                no_null_no_sum += no_null_no
                acc_wo_cum += acc_without_null * no_null_no

                loss_value = loss.data.cpu().numpy().astype(float)
                loss_cum += loss_value * instr.shape[0]

                # print(token_idx)

                bool_tokens_null_true = bool_tokens
                bool_tokens_null_true[~len_mask] = True
                prog_correct = torch.all(bool_tokens_null_true, 1)
                # print(instr)
                # print(token_idx)
                # print(prog_gt)
                # print(bool_tokens_null_true)
                # print(prog_correct)
                acc_prog_lvl = torch.sum(prog_correct) / torch.numel(prog_correct)
                acc_prog_lvl_cum += acc_prog_lvl * instr.shape[0]

                prog_start = prog[:, 0].unsqueeze(1)

                token_pred = self.model.sample(instr, prog_start, ins_len, end_tok)
                prog_gt_len = prog_gt.shape[1]
                token_pred = token_pred[:, :prog_gt_len]
                # print(token_pred.shape)
                # print(prog_gt.shape)

                bool_tokens = token_pred == prog_gt
                acc_with_null_sample = torch.sum(bool_tokens) / torch.numel(bool_tokens)
                acc_w_cum_sample += acc_with_null_sample * instr.shape[0]

                acc_without_null_sample = torch.sum(
                    bool_tokens[len_mask]) / torch.numel(bool_tokens[len_mask])
                acc_wo_cum_sample += acc_without_null_sample * no_null_no

                # print(token_pred)
                # print(token_pred==token_idx)

                bool_tokens_null_true = bool_tokens
                bool_tokens_null_true[~len_mask] = True
                bool_tokens_null_true[:, 0] = True
                prog_correct = torch.all(bool_tokens_null_true, 1)
                acc_prog_lvl_sample = torch.sum(prog_correct) / torch.numel(prog_correct)
                sum_corr_sample += torch.sum(prog_correct)
                acc_prog_lvl_cum_sample += acc_prog_lvl_sample * instr.shape[0]

        loss_cum = loss_cum / len(loader.dataset)
        acc_w_cum = acc_w_cum / len(loader.dataset)
        acc_w_cum_sample = acc_w_cum_sample / len(loader.dataset)
        acc_wo_cum = acc_wo_cum / no_null_no_sum
        acc_wo_cum_sample = acc_wo_cum_sample / no_null_no_sum
        acc_prog_lvl_cum = acc_prog_lvl_cum / len(loader.dataset)
        acc_prog_lvl_cum_sample = acc_prog_lvl_cum_sample / len(loader.dataset)
        return {
            'acc_sum': acc_w_cum + acc_prog_lvl_cum + acc_prog_lvl_cum_sample,
            'tok_acc_w_null': acc_w_cum,
            'tok_acc_w_null_sample': acc_w_cum_sample,
            'tok_acc_wo_null': acc_wo_cum,
            'tok_acc_wo_null_sample': acc_wo_cum_sample,
            'tok_acc_prog': acc_prog_lvl_cum,
            'tok_acc_prog_sample': acc_prog_lvl_cum_sample,
            'sum_corr_sample': sum_corr_sample,
            'loss': loss_cum
        }

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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.optimiser is not None:
            if self.model.training and not zero_train:
                self.optimiser.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] if not zero_train else None
        num_iter = checkpoint['num_iter'] if not zero_train else None
        return epoch, num_iter

    def __str__(self):
        return "Seq2seq model"


class Seq2seqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inp, out, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(inp, input_lengths)
        # print(encoder_outputs.shape)
        # print(encoder_hidden.shape)
        decoder_outputs, decoder_hidden = self.decoder(out, encoder_outputs, encoder_hidden)

        return decoder_outputs

    def sample(self, inp, prog_start, ins_len, end_tok):
        encoder_outputs, encoder_hidden = self.encoder(inp, ins_len)
        token_pred = self.decoder.sample(
            prog_start, encoder_outputs, encoder_hidden, end_tok)

        return token_pred
