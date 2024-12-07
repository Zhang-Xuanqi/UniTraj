# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.data.storage import NodeStorage

from unitraj.models.qcnet.losses import MixtureNLLLoss, NLLLoss
from unitraj.models.qcnet.metrics import Brier, MR, minADE, minAHE, minFDE, minFHE
from unitraj.models.qcnet.modules import QCNetDecoder, QCNetEncoder

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNet(pl.LightningModule):

    def __init__(self, config) -> None:
        super(QCNet, self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.dataset = config.dataset
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.output_head = config.output_head
        self.num_historical_steps = config.num_historical_steps
        self.num_future_steps = config.num_future_steps
        self.num_modes = config.num_modes
        self.num_recurrent_steps = config.num_recurrent_steps
        self.num_freq_bands = config.num_freq_bands
        self.num_map_layers = config.num_map_layers
        self.num_agent_layers = config.num_agent_layers
        self.num_dec_layers = config.num_dec_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.pl2pl_radius = config.pl2pl_radius
        self.time_span = config.time_span
        self.pl2a_radius = config.pl2a_radius
        self.a2a_radius = config.a2a_radius
        self.num_t2m_steps = config.num_t2m_steps
        self.pl2m_radius = config.pl2m_radius
        self.a2m_radius = config.a2m_radius
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.T_max = config.T_max
        self.submission_dir = config.submission_dir
        self.submission_file_name = config.submission_file_name
        self.input_keys = config.input_keys

        self.encoder = QCNetEncoder(
            dataset=self.dataset,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_historical_steps=self.num_historical_steps,
            pl2pl_radius=self.pl2pl_radius,
            time_span=self.time_span,
            pl2a_radius=self.pl2a_radius,
            a2a_radius=self.a2a_radius,
            num_freq_bands=self.num_freq_bands,
            num_map_layers=self.num_map_layers,
            num_agent_layers=self.num_agent_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )
        self.decoder = QCNetDecoder(
            dataset=self.dataset,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            output_head=self.output_head,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            num_modes=self.num_modes,
            num_recurrent_steps=self.num_recurrent_steps,
            num_t2m_steps=self.num_t2m_steps,
            pl2m_radius=self.pl2m_radius,
            a2m_radius=self.a2m_radius,
            num_freq_bands=self.num_freq_bands,
            num_layers=self.num_dec_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * self.output_dim + ['von_mises'] * self.output_head,
                                       reduction='none')

        self.Brier = Brier(max_guesses=6)
        self.minADE = minADE(max_guesses=6)
        self.minAHE = minAHE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minFHE = minFHE(max_guesses=6)
        self.MR = MR(max_guesses=6)

        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def training_step(self,
                      data,
                      batch_idx):
        # if isinstance(data, Batch):
        #     data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        # if isinstance(data, Batch):
        #     data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['predict_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] == 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group('QCNet')
    #     parser.add_argument('--dataset', type=str, required=True)
    #     parser.add_argument('--input_dim', type=int, default=2)
    #     parser.add_argument('--hidden_dim', type=int, default=128)
    #     parser.add_argument('--output_dim', type=int, default=2)
    #     parser.add_argument('--output_head', action='store_true')
    #     parser.add_argument('--num_historical_steps', type=int, required=True)
    #     parser.add_argument('--num_future_steps', type=int, required=True)
    #     parser.add_argument('--num_modes', type=int, default=6)
    #     parser.add_argument('--num_recurrent_steps', type=int, required=True)
    #     parser.add_argument('--num_freq_bands', type=int, default=64)
    #     parser.add_argument('--num_map_layers', type=int, default=1)
    #     parser.add_argument('--num_agent_layers', type=int, default=2)
    #     parser.add_argument('--num_dec_layers', type=int, default=2)
    #     parser.add_argument('--num_heads', type=int, default=8)
    #     parser.add_argument('--head_dim', type=int, default=16)
    #     parser.add_argument('--dropout', type=float, default=0.1)
    #     parser.add_argument('--pl2pl_radius', type=float, required=True)
    #     parser.add_argument('--time_span', type=int, default=None)
    #     parser.add_argument('--pl2a_radius', type=float, required=True)
    #     parser.add_argument('--a2a_radius', type=float, required=True)
    #     parser.add_argument('--num_t2m_steps', type=int, default=None)
    #     parser.add_argument('--pl2m_radius', type=float, required=True)
    #     parser.add_argument('--a2m_radius', type=float, required=True)
    #     parser.add_argument('--lr', type=float, default=5e-4)
    #     parser.add_argument('--weight_decay', type=float, default=1e-4)
    #     parser.add_argument('--T_max', type=int, default=64)
    #     parser.add_argument('--submission_dir', type=str, default='./')
    #     parser.add_argument('--submission_file_name', type=str, default='submission')
    #     return parent_parser


    # def dict_to_heterodata(self,
    #                        data):
    #     input_keys = self.input_keys
    #     input_dict = data['input_dict']
        
    #     self.num_historical_steps = input_dict['obj_trajs'].shape[2]
        
    #     hetero_data = HeteroData()
        
    #     # for key in input_keys:
    #     #     if key not in ['scenario_id', 'city']:
    #     #         hetero_data[key] = NodeStorage()
    #     #     else:
    #     #         hetero_data[key] = []

    #     num_agents = input_dict["obj_trajs_pos"].shape[1]
    #     hetero_data['scenario_id'] = input_dict["scenario_id"]
        
    #     # deal with agent info
    #     hetero_data['agent']['num_nodes'] = num_agents
    #     # hetero_data['agent'].av_index

    #     valid_mask = torch.cat(
    #         (input_dict['obj_trajs_mask'], input_dict['obj_trajs_future_mask']), dim=-1
    #     ).squeeze(0)
    #     hetero_data['agent']['valid_mask'] = valid_mask
    #     num_steps = self.num_historical_steps + self.num_future_steps
    #     predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    #     predict_mask[:, :self.num_historical_steps] = False
    #     predict_mask[:, self.num_historical_steps:] = True
    #     predict_mask[~valid_mask.bool()] = False
    #     hetero_data['agent']['predict_mask'] = predict_mask
        
    #     # hetero_data['agent']['agent_id']
    #     # hetero_data['agent']['agent_type']

    #     obj_future_pos = input_dict['obj_trajs_future_state'][..., :3].squeeze(0)
    #     obj_future_pos[:, :, 2][:] = 0.
    #     hetero_data['agent']['position'] = torch.cat(
    #         [input_dict["obj_trajs_pos"].squeeze(0), obj_future_pos], dim=1
    #     )

    #     obj_his_velo = input_dict['obj_trajs'][..., 35:37].squeeze(0)
    #     obj_future_velo = input_dict['obj_trajs_future_state'][..., 2:].squeeze(0)
    #     obj_full_velo = torch.cat(
    #         [obj_his_velo, obj_future_velo], dim=1
    #     )
    #     hetero_data['agent']['velocity'] = F.pad(obj_full_velo, (0, 1), "constant", 0.0)
    #     hetero_data['agent']['heading_encoding'] = input_dict['obj_trajs'][..., 35:37].squeeze(0)
        
    #     # deal with map info
    #     map_polylines = input_dict['map_polylines'].clone()
    #     polylines_position = map_polylines[:, :, :, 0:3]  # shape: (1, 768, 30, 3)
    #     polylines_direction = map_polylines[:, :, :, 3:6]  # shape: (1, 768, 30, 3) #### change to 1
    #     hetero_data["map_point"]['position'] = polylines_position.view(1, -1, 3)
    #     hetero_data["map_point"]['orientation'] = polylines_direction.view(1, -1, 3)
        
    #     hetero_data['map_polylines']['map_center'] = input_dict['map_center'].squeeze(0)
    #     hetero_data['map_polylines']['position'] = input_dict['map_polylines'][..., 0:3].squeeze(0)
    #     hetero_data['map_polylines']['orientation'] = input_dict['map_polylines'][..., 3:6].squeeze(0)
    #     hetero_data['map_polylines']['lane_type'] = input_dict['map_polylines'][..., 9:29].squeeze(0)
    #     hetero_data['map_polylines']['mask'] = input_dict['map_polylines_mask'].squeeze(0)
    #     hetero_data['map_polylines']['center'] = input_dict['map_polylines_center'].squeeze(0)
        
    #     hetero_data_batch = Batch.from_data_list([hetero_data])
        
    #     return hetero_data_batch
    
    # def data_process(self, data):
    #     processed = list()
    #     splited_data = self.split_batch_data(data)
    #     for batch_data in splited_data:
    #         hetero_data = self.dict_to_heterodata(batch_data)
    #         if hetero_data is None:
    #             print(f"Warning: dict_to_heterodata returned None for batch_data: {batch_data}")
    #             continue
    #         processed.append(hetero_data)
    #     if not processed:
    #         raise ValueError("No data was processed, check the input data and processing steps.")
    #     return Batch.from_data_list(processed)
            
    # def split_batch_data(self, data):
    #     """
    #     split batch from data and return a list of batches
    #     """
    #     batch_size = data['input_dict']['obj_trajs'].shape[0]
    #     split_data_list = []
    #     for batch_idx in range(batch_size):
    #         split_data = dict()
    #         for key, value in data.items():
    #             if key == 'input_dict':
    #                 split_data[key] = dict()
    #                 for k, v in data['input_dict'].items():
    #                     if isinstance(v, torch.Tensor):
    #                         split_data[key][k] = v[batch_idx]
    #                     else:
    #                         split_data[key][k] = v
    #             # else:
    #             #     split_data[key] = value
    #         split_data_list.append(split_data)
    #     return split_data_list
    
    # def tf_map_center_to_global_coord(self, map_center, input):
        

    # def dict_to_heterodata(self, data):
    #     input_dict = data['input_dict']
    #     device = input_dict['obj_trajs'].device
    #     self.num_historical_steps = input_dict['obj_trajs'].shape[1]
        
    #     hetero_data = HeteroData()
    #     num_agents = input_dict["obj_trajs_pos"].shape[0]
    #     hetero_data['scenario_id'] = input_dict["scenario_id"]
    #     hetero_data['agent']['num_nodes'] = num_agents

    #     # concate past and future masks
    #     valid_mask = torch.cat(
    #         (input_dict['obj_trajs_mask'].clone(),
    #          input_dict['obj_trajs_future_mask'].clone()), dim=-1
    #     ).to(device)
    #     hetero_data['agent']['valid_mask'] = valid_mask
        
    #     num_steps = self.num_historical_steps + self.num_future_steps
    #     predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool, device=device)
    #     predict_mask[:, :self.num_historical_steps] = False
    #     predict_mask[:, self.num_historical_steps:] = True
    #     predict_mask[~valid_mask.bool()] = False
    #     hetero_data['agent']['predict_mask'] = predict_mask
        
    #     # agent positions
    #     obj_future_pos = input_dict['obj_trajs_future_state'][..., :3].clone().to(device)
    #     obj_future_pos[:, :, 2][:] = 0.
    #     hetero_data['agent']['position'] = torch.cat([input_dict["obj_trajs_pos"].clone(), obj_future_pos], dim=1).to(device)
    #     obj_his_velo = input_dict['obj_trajs'][..., 35:37].clone().to(device)
    #     obj_future_velo = input_dict['obj_trajs_future_state'][..., -2:].clone().to(device)
    #     obj_full_velo = torch.cat([obj_his_velo, obj_future_velo], dim=1)
    #     hetero_data['agent']['velocity'] = F.pad(obj_full_velo, (0, 1), "constant", 0.0).to(device)
    #     obj_his_heading_encoding = input_dict['obj_trajs'][..., 35:37].clone().to(device)
    #     obj_his_heading = torch.atan2(obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1]).to(device)
    #     obj_future_heading_encoding = input_dict['obj_trajs_future_state'][..., 2:4].clone().to(device)
    #     obj_future_heading = torch.atan2(obj_future_heading_encoding[..., 0], obj_future_heading_encoding[..., 1]).to(device)
    #     hetero_data['agent']['heading'] = torch.cat([obj_his_heading, obj_future_heading], dim=1).to(device)
    #     obj_type = input_dict['obj_trajs'][:, 0, 11:33]
    #     hetero_data['agent']['type'] = torch.argmax(obj_type, dim=1) 
        
    #     # map points
    #     map_polylines = input_dict['map_polylines'].clone().to(device)
    #     hetero_data['map_point']['num_nodes'] = map_polylines.shape[0] * (map_polylines.shape[1]-1)
    #     polylines_position = map_polylines[..., 0:3]  # shape: (768, 30, 3)
    #     polylines_direction = map_polylines[..., 3:6]  # shape: (768, 30, 3)
    #     hetero_data["map_point"]['position'] = polylines_position[:, :-1].reshape(-1, 3).clone().to(device)
    #     hetero_data["map_point"]['orientation'] = torch.atan2(polylines_direction[:, :-1, 0], polylines_direction[:, :-1, 1]).reshape(-1).to(device)
    #     hetero_data["map_point"]['magnitude'] = torch.norm(
    #         (map_polylines[:, 1:, 0:3] - map_polylines[:, :-1, 0:3]), p=2, dim=-1
    #     ).view(-1).to(device)
    #     hetero_data["map_point"]['type'] = torch.zeros_like(map_polylines[:, :-1, 0].reshape(-1), dtype=torch.uint8, device=device)
    #     hetero_data["map_point"]['side'] = torch.zeros_like(map_polylines[:, :-1, 0].reshape(-1), dtype=torch.uint8, device=device)
        
    #     # map polygons --> use polyline info instead
    #     hetero_data['map_polygon']['num_nodes'] = input_dict["map_polylines"].shape[0]
    #     hetero_data['map_polygon']['position'] = torch.mean(
    #         input_dict['map_polylines'][..., 0:3].clone(), dim=1
    #     ).to(device)
    #     start_points = input_dict['map_polylines'][..., 0:3][:, 0, :2].to(device)
    #     second_points = input_dict['map_polylines'][..., 0:3][:, 1, :2].to(device)
    #     hetero_data['map_polygon']['orientation'] = torch.atan2(
    #         (second_points - start_points)[:, 1], (second_points - start_points)[:, 0]
    #     ).to(device)
    #     pl_type_one_hot = map_polylines[:, 0, 9:29]
    #     hetero_data['map_polygon']['type'] = pl_type_one_hot.argmax(dim=-1).to(device)
    #     hetero_data['map_polygon']['is_intersection'] = torch.zeros(input_dict["map_polylines"].shape[0], device=device)
        
    #     # map polylines
    #     hetero_data['map_center'] = input_dict['map_center'].clone().to(device)
    #     hetero_data['map_polylines']['position'] = input_dict['map_polylines'][..., 0:3].clone().to(device)
    #     hetero_data['map_polylines']['orientation'] = input_dict['map_polylines'][..., 3:6].clone().to(device)
    #     hetero_data['map_polylines']['lane_type'] = input_dict['map_polylines'][..., 9:29].clone().to(device)
    #     hetero_data['map_polylines']['mask'] = input_dict['map_polylines_mask'].clone().to(device)
    #     hetero_data['map_polylines']['center'] = input_dict['map_polylines_center'].clone().to(device)
        
    #     # pt & pl indexing
    #     num_polylines = map_polylines.shape[0]
    #     num_points_per_polyline = map_polylines.shape[1] - 1
    #     point_to_polygon_edge_index = torch.stack([
    #         torch.arange(num_polylines * num_points_per_polyline, dtype=torch.long),
    #         torch.arange(num_polylines, dtype=torch.long).repeat_interleave(num_points_per_polyline)
    #     ], dim=0).to(device)
    #     hetero_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        
    #     polygon_to_polygon_edge_index = []
    #     for i in range(num_polylines - 1):
    #         polygon_to_polygon_edge_index.append(torch.tensor([[i, i+1], [i+1, i]]))
    #     polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1) if polygon_to_polygon_edge_index else torch.empty((2, 0), dtype=torch.long)
        
    #     return hetero_data
