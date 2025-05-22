import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_sparse import transpose
from torch_geometric.utils import is_undirected, sort_edge_index
from collections import defaultdict
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import rdmolops, Draw
import random
import pandas as pd
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torchcontrib.optim import SWA
from torch_geometric.utils import unbatch_edge_index
from itertools import combinations
from tqdm.rich import tqdm
import math

from torch.utils.tensorboard import SummaryWriter
import pickle
import time

from copy import copy


class BaseTrainer(object):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        self.method_name = None
        self.checkpoints_path = None

        self.model = model
        self.explainer = explainer
        self.dataloader = dataloader
        self.cfg = cfg

        self.device = device

        self.best_valid_score = 0.0
        self.lowest_valid_loss = float('inf')
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(os.path.join(save_dir, 'checkpoints'))

    def set_method_name(self, method_name):
        self.method_name = method_name
        self.checkpoints_path = os.path.join(self.save_dir, 'checkpoints',
                                             f'{self.method_name}_{self.cfg.dataset_name}.pth')

    def _train_batch(self, data):
        raise NotImplementedError

    def _valid_batch(self, data):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def valid(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def process_data(data, use_edge_attr):
        if not use_edge_attr:
            data.edge_attr = None
        if data.get('edge_label', None) is None:
            data.edge_label = torch.zeros(data.edge_index.shape[1])
        return data

    @torch.inference_mode()
    def test_by_sample_ensemble(self, dataset_name, method_name, ensemble_numbers=[0]):
        if 'cal' in method_name:
            accumulated_metrics = defaultdict(list)
            att_dict = defaultdict(list)
            exp_label_dict = {}
            clf_dict = defaultdict(list)
            clf_label_dict = {}
            clf_label_dict_bool = defaultdict(list)

            for index in ensemble_numbers:
                test_batch_loss_list = []
                accumulated_info = defaultdict(list)
                new_checkpoints_path = f'{self.checkpoints_path[:-4]}_{index}.pth'
                self.load_model(new_checkpoints_path)
                self.model.eval()
                self.explainer.eval()

                for data_index, data in enumerate(self.dataloader['test_by_sample']):
                    data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)

                    emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                             batch=data.batch)
                    att_log_logit = self.explainer(emb, data.edge_index, data.batch)
                    att = self.concrete_sample(att_log_logit, training=True)

                    c_edge_att = self.process_att_to_edge_att(data, att)
                    s_edge_att = 1 - c_edge_att

                    c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=c_edge_att)
                    s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch,
                                               edge_atten=s_edge_att)

                    c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
                    s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

                    num = s_emb.shape[0]
                    l = [i for i in range(num)]
                    random_idx = torch.tensor(l)
                    csi_emb = s_emb[random_idx] + c_emb

                    csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

                    info = dict()
                    info['clf_logits'] = csi_clf_logits.detach()
                    info['clf_label'] = data.y.data
                    info['edge_att'] = c_edge_att.squeeze().detach()
                    info['exp_label'] = data.edge_label.data

                    att_dict[data_index].append(info['edge_att'])
                    exp_label_dict[data_index] = data.edge_label.data

                    clf_dict[data_index].append(info['clf_logits'])
                    clf_label_dict[data_index] = info['clf_label']

                    preds = (info['clf_logits'].sigmoid() > 0.5).float()

                    if preds == data.y.data:
                        clf_label_dict_bool[data_index].append(
                            1 - torch.abs(info['clf_logits'].sigmoid() - data.y.data))
                    else:
                        clf_label_dict_bool[data_index].append(0)

                    for key, value in info.items():
                        accumulated_info[key].append(value)

                    # test_batch_loss_list.append(loss.item())
                    test_batch_loss_list.append(0)
                test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
                test_metrics = self.calculate_metrics(accumulated_info)
                for key, value in test_metrics.items():
                    accumulated_metrics[key].append(value)
                # print(test_metrics)
            average_metrics = {key: (np.mean(values), np.std(values)) for key, values in accumulated_metrics.items()}
            print(f"10 runs result: {average_metrics}")

            att_list = []
            exp_label_list = []

            for key in att_dict.keys():
                exp_label_list.append(exp_label_dict[key])
                att_list.append(torch.mean(torch.stack(att_dict[key]), dim=0))

            acc_list = []
            for key in clf_dict.keys():
                clf = torch.mean(torch.stack(clf_dict[key]), dim=0)
                clf_label = clf_label_dict[key]
                clf_preds = self.get_preds(clf, self.cfg.multi_label)
                clf_labels = clf_label
                acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
                acc_list.append(acc)

            exp_att = torch.cat(att_list)
            exp_labels = torch.cat(exp_label_list)
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            print(f"ensemble acc: {np.mean(acc_list)}, ensemble auc: {roc_auc}")
        else:
            accumulated_metrics = defaultdict(list)
            att_dict = defaultdict(list)
            exp_label_dict = {}
            clf_dict = defaultdict(list)
            clf_label_dict = {}
            clf_label_dict_bool = defaultdict(list)

            for index in ensemble_numbers:
                test_batch_loss_list = []
                accumulated_info = defaultdict(list)
                new_checkpoints_path = f'{self.checkpoints_path[:-4]}_{index}.pth'
                self.load_model(new_checkpoints_path)
                self.model.eval()
                self.explainer.eval()

                for data_index, data in enumerate(self.dataloader['test_by_sample']):
                    data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)

                    emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                             batch=data.batch)
                    att_log_logit = self.explainer(emb, data.edge_index, data.batch)
                    att = self.concrete_sample(att_log_logit, training=False)
                    edge_att = self.process_att_to_edge_att(data, att)

                    clf_logits = self.model(x=data.x,
                                            edge_index=data.edge_index,
                                            edge_attr=data.edge_attr,
                                            batch=data.batch,
                                            edge_atten=edge_att)
                    # loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
                    info = dict()
                    info['clf_logits'] = clf_logits.detach()
                    info['clf_label'] = data.y.data
                    info['edge_att'] = edge_att.squeeze().detach()
                    info['exp_label'] = data.edge_label.data

                    att_dict[data_index].append(info['edge_att'])
                    exp_label_dict[data_index] = data.edge_label.data

                    clf_dict[data_index].append(info['clf_logits'])
                    clf_label_dict[data_index] = info['clf_label']

                    preds = (info['clf_logits'].sigmoid() > 0.5).float()

                    if preds == data.y.data:
                        clf_label_dict_bool[data_index].append(
                            1 - torch.abs(info['clf_logits'].sigmoid() - data.y.data))
                    else:
                        clf_label_dict_bool[data_index].append(0)

                    for key, value in info.items():
                        accumulated_info[key].append(value)

                    # test_batch_loss_list.append(loss.item())
                    test_batch_loss_list.append(0)
                test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
                test_metrics = self.calculate_metrics(accumulated_info)
                for key, value in test_metrics.items():
                    accumulated_metrics[key].append(value)
                # print(test_metrics)
            average_metrics = {key: (np.mean(values), np.std(values)) for key, values in accumulated_metrics.items()}
            print(f"10 runs result: {average_metrics}")

            att_list = []
            exp_label_list = []

            for key in att_dict.keys():
                exp_label_list.append(exp_label_dict[key])
                att_list.append(torch.mean(torch.stack(att_dict[key]), dim=0))

            acc_list = []
            for key in clf_dict.keys():
                clf = torch.mean(torch.stack(clf_dict[key]), dim=0)
                clf_label = clf_label_dict[key]
                clf_preds = self.get_preds(clf, self.cfg.multi_label)
                clf_labels = clf_label
                acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
                acc_list.append(acc)

            exp_att = torch.cat(att_list)
            exp_labels = torch.cat(exp_label_list)
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            print(f"ensemble acc: {np.mean(acc_list)}, ensemble auc: {roc_auc}")
        return np.mean(acc_list), roc_auc

    @torch.inference_mode()
    def calculate_shd(self, dataset_name, method_name, ensemble_numbers=[0]):
        # if not os.path.exists(f'./outputs/temp_data/shd_{method_name}_{dataset_name}.pt'):
        import copy
        ori_data = []
        for data in self.dataloader['test_by_sample']:
            ori_data.append(copy.deepcopy(data))
            # print(ori_data)
            # exit()
        for model_index in ensemble_numbers:
            new_checkpoints_path = f'{self.checkpoints_path[:-4]}_{model_index}.pth'
            self.load_model(new_checkpoints_path)
            self.model.eval()
            self.explainer.eval()

            for data_index, data in enumerate(self.dataloader['test_by_sample']):
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                         batch=data.batch)
                att_log_logit = self.explainer(emb, data.edge_index, data.batch)
                att = self.concrete_sample(att_log_logit, training=False)
                edge_att = self.process_att_to_edge_att(data, att)
                ori_data[data_index][f'edge_att_{model_index}'] = edge_att.squeeze().detach()
                # print(ori_data[data_index])
                # exit()

        from itertools import combinations
        from tqdm.rich import tqdm
        all_scores = []
        all_scores_std = []

        one = torch.tensor(1)
        zero = torch.tensor(0)

        for data in tqdm(ori_data):
            edge_atts = [data[f'edge_att_{i}'] for i in range(len(ensemble_numbers))]

            assert all(len(att) == len(edge_atts[0]) for att in edge_atts), "All edge_att must have the same length"

            edge_atts_bin = [torch.where(att > 0.5, one, zero) for att in edge_atts]

            n = len(edge_atts_bin)
            scores = []

            combinations_list = combinations(range(n), 1)
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    # print(pair)
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    edge_atts_first_bin = torch.where(edge_atts_first > 0.5, one, zero)
                    edge_atts_second_bin = torch.where(edge_atts_second > 0.5, one, zero)
                    hamming_dist = torch.sum(edge_atts_first_bin != edge_atts_second_bin).item()
                    score = hamming_dist
                    scores.append(score)

            average_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores.append(average_score)
            all_scores_std.append(std_score)

        print(f"before ensemble (n=1) shd: {np.mean(all_scores)}, {np.mean(all_scores_std)}")

        all_scores = []
        all_scores_std = []

        one = torch.tensor(1)
        zero = torch.tensor(0)

        for data in tqdm(ori_data):
            edge_atts = [data[f'edge_att_{i}'] for i in range(len(ensemble_numbers))]

            assert all(len(att) == len(edge_atts[0]) for att in edge_atts), "All edge_att must have the same length"

            edge_atts_bin = [torch.where(att > 0.5, one, zero) for att in edge_atts]

            n = len(edge_atts_bin)
            scores = []

            combinations_list = combinations(range(n), int(0.5 * n))
            for pair in combinations(combinations_list, 2):
                if set(pair[0]).isdisjoint(pair[1]):
                    # print(pair)
                    edge_atts_first = torch.stack([edge_atts[i] for i in pair[0]]).mean(dim=0)
                    edge_atts_second = torch.stack([edge_atts[i] for i in pair[1]]).mean(dim=0)
                    edge_atts_first_bin = torch.where(edge_atts_first > 0.5, one, zero)
                    edge_atts_second_bin = torch.where(edge_atts_second > 0.5, one, zero)
                    hamming_dist = torch.sum(edge_atts_first_bin != edge_atts_second_bin).item()
                    score = hamming_dist
                    scores.append(score)

            average_score = np.mean(scores)
            std_score = np.std(scores)
            all_scores.append(average_score)
            all_scores_std.append(std_score)

        print(f"after ensemble (n=5) shd: {np.mean(all_scores)}, {np.mean(all_scores_std)}")
        exit()


class ATTTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):

        ce_loss = self.criterion(logits, labels)
        loss = ce_loss * self.ce_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        from datetime import datetime
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            # valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=False)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        edge_label_counter_list, edge_att_counter_list = [], []
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)

            edge_att_counter = torch.sum(test_batch_info['edge_att'] > 0.5).item() / test_batch_info['edge_att'].size(0)

            edge_label_counter = torch.sum(test_batch_info['exp_label'] == 1).item() / test_batch_info[
                'exp_label'].size(0)
            edge_att_counter_list.append(edge_att_counter)
            edge_label_counter_list.append(edge_label_counter)

            # print(torch.sum(test_batch_info['exp_label'] == 1).item())

            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        info_for_hist = {}
        info_for_hist['att'] = edge_att_counter_list
        info_for_hist['label'] = edge_label_counter_list

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class SIZETrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef
        self.sparsity_mask_coef = cfg.sparsity_mask_coef
        self.sparsity_ent_coef = cfg.sparsity_ent_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    def sparsity(self, edge_mask, eps=1e-6):
        sparsity = 0.
        # sparsity += self.sparsity_mask_coef * edge_mask.mean()
        sparsity += self.c * edge_mask.mean()
        # ent = -edge_mask * torch.log(edge_mask + eps) - (1 - edge_mask) * torch.log(1 - edge_mask + eps)
        # sparsity += self.sparsity_ent_coef * ent.mean()
        return sparsity

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def get_sparsity_c(self, current_epoch):
        c = self.sparsity_mask_coef * (current_epoch+1) * 2 / 100
        if c > self.sparsity_mask_coef:
            c = self.sparsity_mask_coef
        return c
    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):
        ce_loss = self.criterion(logits, labels)
        reg_loss = self.sparsity(att)
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        from datetime import datetime
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)
            self.c = self.get_sparsity_c(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            # valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=False)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        edge_label_counter_list, edge_att_counter_list = [], []
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)

            edge_att_counter = torch.sum(test_batch_info['edge_att'] > 0.5).item() / test_batch_info['edge_att'].size(0)

            edge_label_counter = torch.sum(test_batch_info['exp_label'] == 1).item() / test_batch_info[
                'exp_label'].size(0)
            edge_att_counter_list.append(edge_att_counter)
            edge_label_counter_list.append(edge_label_counter)

            # print(torch.sum(test_batch_info['exp_label'] == 1).item())

            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        info_for_hist = {}
        info_for_hist['att'] = edge_att_counter_list
        info_for_hist['label'] = edge_label_counter_list

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class GSATTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval
        self.ce_loss_coef = cfg.ce_loss_coef
        self.reg_loss_coef = cfg.reg_loss_coef

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, att, logits, labels):

        ce_loss = self.criterion(logits, labels)
        reg_loss = (att * torch.log(att / self.r + 1e-6)
                    + (1 - att) * torch.log((1 - att) / (1 - self.r + 1e-6) + 1e-6)).mean()
        loss = ce_loss * self.ce_loss_coef + reg_loss * self.reg_loss_coef
        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)

        att = self.concrete_sample(att_log_logit, training=True)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)

        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        edge_att = self.process_att_to_edge_att(data, att)
        clf_logits = self.model(x=data.x,
                                edge_index=data.edge_index,
                                edge_attr=data.edge_attr,
                                batch=data.batch,
                                edge_atten=edge_att)
        loss = self.__loss__(att=att, logits=clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        from datetime import datetime
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)

            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (self.r == self.final_r) and (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            # valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=False)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        edge_label_counter_list, edge_att_counter_list = [], []
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)

            edge_att_counter = torch.sum(test_batch_info['edge_att'] > 0.5).item() / test_batch_info['edge_att'].size(0)

            edge_label_counter = torch.sum(test_batch_info['exp_label'] == 1).item() / test_batch_info[
                'exp_label'].size(0)
            edge_att_counter_list.append(edge_att_counter)
            edge_label_counter_list.append(edge_label_counter)

            # print(torch.sum(test_batch_info['exp_label'] == 1).item())

            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        info_for_hist = {}
        info_for_hist['att'] = edge_att_counter_list
        info_for_hist['label'] = edge_label_counter_list

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


class CALTrainer(BaseTrainer):
    def __init__(self, model, explainer, dataloader, cfg, device, save_dir):
        super().__init__(model, explainer, dataloader, cfg, device, save_dir)
        self.method_name = None

        self.learn_edge_att = cfg.learn_edge_att
        self.r = cfg.init_r
        self.init_r = cfg.init_r
        self.final_r = cfg.final_r
        self.decay_r = cfg.decay_r
        self.decay_interval = cfg.decay_interval

        self.csi_loss_coef = cfg.csi_loss_coef
        self.c_loss_coef = cfg.c_loss_coef
        self.s_loss_coef = cfg.s_loss_coef

        self.num_class = cfg.num_class

        if cfg.num_class == 2 and not cfg.multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        elif cfg.num_class > 2 and not cfg.multi_label:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.explainer.parameters()),
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)

    @staticmethod
    def get_preds(logits, multi_label):
        if multi_label:
            preds = (logits.sigmoid() > 0.5).float()
        elif logits.shape[1] > 1:  # multi-class
            preds = logits.argmax(dim=-1).float()
        else:  # binary
            preds = (logits.sigmoid() > 0.5).float()
        return preds

    def get_r(self, current_epoch):
        # r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        # if r < self.final_r:
        #     r = self.final_r
        r = self.final_r
        return r

    def save_model(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'explainer_state_dict': self.explainer.state_dict()
        }
        torch.save(state, path)
        print(f"model saved -- {path}")

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.explainer.load_state_dict(state['explainer_state_dict'])
        print(f"model loaded -- {path}")

    @staticmethod
    def convert_node_att_to_edge_att(node_att, edge_index):
        src_att = node_att[edge_index[0]]
        dst_att = node_att[edge_index[1]]
        edge_att = src_att * dst_att
        return edge_att

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    @staticmethod
    def concrete_sample(att_log_logit, temp=1, training=False):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = att_log_logit.sigmoid()
        return att_bern

    def process_att_to_edge_att(self, data, att):
        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.convert_node_att_to_edge_att(att, data.edge_index)

        return edge_att

    def __loss__(self, c_logits, s_logits, csi_logits, labels):

        if self.num_class == 2:
            s_prob = torch.sigmoid(s_logits)
            s_loss = (s_prob * torch.log(s_prob / self.r + 1e-6)
                      + (1 - s_prob) * torch.log((1 - s_prob) / (1 - self.r + 1e-6) + 1e-6)).mean()
        else:
            log_s_probs = torch.log_softmax(s_logits, dim=-1)
            uniform_target = torch.ones_like(s_logits, dtype=torch.float).cuda() / self.num_class
            s_loss = F.kl_div(log_s_probs, uniform_target, reduction='batchmean')

        c_loss = self.criterion(c_logits, labels)

        csi_loss = self.criterion(csi_logits, labels)

        loss = csi_loss * self.csi_loss_coef + c_loss * self.c_loss_coef + s_loss * self.s_loss_coef

        return loss

    def _train_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=True)

        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        info = dict()

        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def _valid_batch(self, data, is_test=True):
        emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        att_log_logit = self.explainer(emb, data.edge_index, data.batch)
        att = self.concrete_sample(att_log_logit, training=False)
        c_edge_att = self.process_att_to_edge_att(data, att)
        s_edge_att = 1 - c_edge_att

        c_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=c_edge_att)
        s_emb = self.model.get_emb(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch,
                                   edge_atten=s_edge_att)

        c_clf_logits = self.model.get_pred_from_c_emb(emb=c_emb, batch=data.batch)
        s_clf_logits = self.model.get_pred_from_s_emb(emb=s_emb, batch=data.batch)

        num = s_emb.shape[0]
        l = [i for i in range(num)]
        random_idx = torch.tensor(l)
        csi_emb = s_emb[random_idx] + c_emb

        csi_clf_logits = self.model.get_pred_from_csi_emb(emb=csi_emb, batch=data.batch)

        loss = self.__loss__(c_logits=c_clf_logits, s_logits=s_clf_logits, csi_logits=csi_clf_logits, labels=data.y)
        info = dict()
        if is_test:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
            info['edge_att'] = c_edge_att.squeeze().detach()
            info['exp_label'] = data.edge_label.data
        else:
            info['clf_logits'] = csi_clf_logits.detach()
            info['clf_label'] = data.y.data
        return loss.item(), info

    def calculate_metrics(self, accumulated_info):
        metrics = dict()
        if accumulated_info['edge_att']:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            exp_att = torch.cat(accumulated_info['edge_att'])
            exp_labels = torch.cat(accumulated_info['exp_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            try:
                roc_auc = roc_auc_score(exp_labels.cpu(), exp_att.cpu())
            except ValueError:
                roc_auc = 0
            metrics['acc'] = acc
            metrics['roc_auc'] = roc_auc
        else:
            clf_preds = self.get_preds(torch.cat(accumulated_info['clf_logits']), self.cfg.multi_label)
            clf_labels = torch.cat(accumulated_info['clf_label'])
            acc = (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
            metrics['acc'] = acc
        return metrics

    def train(self):
        self.writer = SummaryWriter(
            f"./outputs/logs/{self.method_name}/{self.cfg.dataset_name}")

        from datetime import datetime
        self.timestamp = datetime.timestamp(datetime.now())

        for e in range(self.cfg.epochs):
            train_batch_loss_list = []
            accumulated_info = defaultdict(list)
            self.model.train()
            self.explainer.train()
            self.r = self.get_r(e)
            all_att, all_exp = [], []

            for data in self.dataloader['train']:
                data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
                train_batch_loss, train_batch_info = self._train_batch(data, is_test=True)

                att = train_batch_info['edge_att'].data.cpu()
                exp = train_batch_info['exp_label'].data.cpu()
                all_att.append(att)
                all_exp.append(exp)

                train_batch_loss_list.append(train_batch_loss)
                for key, value in train_batch_info.items():
                    accumulated_info[key].append(value)
            train_loss = torch.FloatTensor(train_batch_loss_list).mean().item()
            self.writer.add_scalar(f'train_loss/{self.timestamp}', train_loss, e)

            train_metrics = self.calculate_metrics(accumulated_info)

            all_att = torch.cat(all_att)
            all_exp = torch.cat(all_exp)
            bkg_att_weights, signal_att_weights = all_att, all_att
            if np.unique(all_exp).shape[0] > 1:
                bkg_att_weights = all_att[all_exp == 0]
                signal_att_weights = all_att[all_exp == 1]
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/bkg_att_weights',
                                      bkg_att_weights, e)
            self.writer.add_histogram(f'train_histogram/{self.timestamp}/signal_att_weights',
                                      signal_att_weights, e)

            valid_loss, valid_metrics = self.valid(e)
            print(f"epoch: {e}, "
                  f"train loss: {train_loss:.4f}, train acc {train_metrics['acc']:.4f}, "
                  f"valid loss: {valid_loss:.4f}, valid acc {valid_metrics['acc']:.4f}")
            if (e > 10) and (
                    (valid_metrics['acc'] > self.best_valid_score) or
                    (valid_metrics['acc'] == self.best_valid_score and valid_loss < self.lowest_valid_loss)):
                self.save_model(self.checkpoints_path)
                self.best_valid_score = valid_metrics['acc']
                self.lowest_valid_loss = valid_loss

    @torch.inference_mode()
    def valid(self, e=None):
        valid_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.model.eval()
        self.explainer.eval()

        all_att, all_exp = [], []

        for data in self.dataloader['valid']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)

            # valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=False)
            valid_batch_loss, valid_batch_info = self._valid_batch(data, is_test=True)  # 0102 for att visualization
            att = valid_batch_info['edge_att'].data.cpu()
            exp = valid_batch_info['exp_label'].data.cpu()
            all_att.append(att)
            all_exp.append(exp)

            valid_batch_loss_list.append(valid_batch_loss)
            for key, value in valid_batch_info.items():
                accumulated_info[key].append(value)
        valid_loss = torch.FloatTensor(valid_batch_loss_list).mean().item()
        self.writer.add_scalar(f'valid_loss/{self.timestamp}', valid_loss, e)

        valid_metrics = self.calculate_metrics(accumulated_info)

        all_att = torch.cat(all_att)
        all_exp = torch.cat(all_exp)
        bkg_att_weights, signal_att_weights = all_att, all_att
        if np.unique(all_exp).shape[0] > 1:
            bkg_att_weights = all_att[all_exp == 0]
            signal_att_weights = all_att[all_exp == 1]
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/bkg_att_weights',
                                  bkg_att_weights, e)
        self.writer.add_histogram(f'valid_histogram/{self.timestamp}/signal_att_weights',
                                  signal_att_weights, e)

        return valid_loss, valid_metrics

    @torch.inference_mode()
    def test(self):
        test_batch_loss_list = []
        accumulated_info = defaultdict(list)
        self.load_model(self.checkpoints_path)
        self.model.eval()
        self.explainer.eval()
        edge_label_counter_list, edge_att_counter_list = [], []
        for data in self.dataloader['test']:
            data = self.process_data(data=data, use_edge_attr=self.cfg.use_edge_attr).to(self.device)
            test_batch_loss, test_batch_info = self._valid_batch(data, is_test=True)

            edge_att_counter = torch.sum(test_batch_info['edge_att'] > 0.5).item() / test_batch_info['edge_att'].size(0)
            edge_label_counter = torch.sum(test_batch_info['exp_label'] == 1).item() / test_batch_info[
                'exp_label'].size(0)
            edge_att_counter_list.append(edge_att_counter)
            edge_label_counter_list.append(edge_label_counter)

            test_batch_loss_list.append(test_batch_loss)
            for key, value in test_batch_info.items():
                accumulated_info[key].append(value)
        test_loss = torch.FloatTensor(test_batch_loss_list).mean().item()
        test_metrics = self.calculate_metrics(accumulated_info)

        info_for_hist = {}
        info_for_hist['att'] = edge_att_counter_list
        info_for_hist['label'] = edge_label_counter_list

        print(
            f"test loss: {test_loss:.4f}, test acc {test_metrics['acc']:.4f}, att roc-auc {test_metrics['roc_auc']:.4f}")
        return test_metrics


def get_trainer(method_name, model, explainer, dataloader, cfg, device, save_dir):
    trainer = None
    if method_name == 'att':
        trainer = ATTTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    elif method_name == 'size':
        trainer = SIZETrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'gsat':
        trainer = GSATTrainer(model=model,
                              explainer=explainer,
                              dataloader=dataloader,
                              cfg=cfg,
                              device=device,
                              save_dir=save_dir)
    elif method_name == 'cal':
        trainer = CALTrainer(model=model,
                             explainer=explainer,
                             dataloader=dataloader,
                             cfg=cfg,
                             device=device,
                             save_dir=save_dir)
    trainer.set_method_name(method_name)
    return trainer
