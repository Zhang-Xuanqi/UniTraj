import os
import pickle
import torch
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from unitraj.models.smart.utils.log import Logging
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import BaseTransform
from unitraj.models.smart.datasets.preprocess import TokenProcessor
from .base_dataset import BaseDataset
from unitraj.datasets.common_utils import get_kalman_difficulty, get_trajectory_type
from unitraj.models.smart.utils import wrap_angle

class SMARTDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self.token_processor = TokenProcessor(2048)
        self.target_transform = WaymoTargetBuilder(11, 80)
        super().__init__(config, is_validation)
        self.logger = Logging().log(level='DEBUG')
        
    def postprocess(self, output):
        
        get_kalman_difficulty(output)
        
        get_trajectory_type(output)

        return self.smart_convert(output)
    
    def smart_convert(self, output):
        data = []
        for i in output:
            d = {
                'scenario_id': i['scenario_id'],
                'kalman_difficulty': i['kalman_difficulty'],
                'trajectory_type': i['trajectory_type'],
            }
            obj_trajs = i['obj_trajs']
            num_historical_steps = obj_trajs.shape[1]

            agent = {}
            agent['num_nodes'] = i['obj_trajs'].shape[0]
            agent['av_index'] = i['track_index_to_predict']

            valid_mask = np.concatenate(
                (i['obj_trajs_mask'], i['obj_trajs_future_mask'].astype(bool)), axis=-1
            )
            agent['valid_mask'] = valid_mask

            predict_mask = np.zeros_like(valid_mask, dtype=bool)
            predict_mask[:, num_historical_steps:] = True
            predict_mask[~valid_mask] = False
            agent['predict_mask'] = predict_mask

            agent['type'] = np.zeros(agent['num_nodes'], dtype=np.uint8)
            agent['category'] = np.argmax(i['obj_trajs'][:, 0, 6:11], axis=1)

            agent['position'] = np.concatenate([i['obj_trajs_pos'], i['obj_trajs_future_state'][..., :3]], axis=1).astype(np.float32)

            obj_his_heading_encoding = i['obj_trajs'][..., 23:25]
            obj_his_heading = np.arctan2(obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1])
            obj_future_heading = i['obj_trajs_future_state'][..., 5]
            agent['heading'] = np.concatenate([obj_his_heading, obj_future_heading], axis=1).astype(np.float32)

            velo_all = np.concatenate(
                [i['obj_trajs'][..., 25:27], i['obj_trajs_future_state'][..., 3:5]], axis=1
            )
            agent['velocity'] = np.pad(velo_all, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0.0).astype(np.float32)

            agent['center_objects_world'] = i['center_objects_world'].reshape(1, -1).astype(np.float32)
            
            agent['shape'] = np.concatenate([i['obj_trajs'][..., 3:6], i['obj_trajs_future_state'][..., 2:5]], axis=1).astype(np.float32)
            d['agent'] = agent

            d['map_polygon'] = {}
            map_polylines = i['map_polylines']
            polyline_mask = ~np.all(map_polylines == 0, axis=(1, 2))
            map_polylines = map_polylines[polyline_mask]
            d['map_polygon']['num_nodes'] = map_polylines.shape[0]
            d['map_polygon']['type'] = np.argmax(map_polylines[:, 0, 9:29], axis=-1)

            d['map_point'] = {}
            polylines_position = map_polylines[:, :-1, 0:3].reshape(-1, 3)
            polylines_orientation = map_polylines[:, :-1, 3:6].reshape(-1, 3)
            pt_valid_mask = (
                (map_polylines[:, :-1, 0] != 0) | (map_polylines[:, :-1, 1] != 0)
            ).flatten()
            pt_pos = polylines_position[pt_valid_mask]
            pt_ori = polylines_orientation[pt_valid_mask]
            d['map_point']['num_nodes'] = pt_pos.shape[0]
            d['map_point']['position'] = pt_pos.astype(np.float32)
            d['map_point']['orientation'] = np.arctan2(pt_ori[:, 1], pt_ori[:, 0]).astype(np.float32)

            pt_mag = np.linalg.norm(
                (map_polylines[:, 1:, 0:3] - map_polylines[:, :-1, 0:3]), axis=-1
            ).reshape(-1)[pt_valid_mask]
            d['map_point']['magnitude'] = pt_mag.astype(np.float32)
            d['map_point']['height'] = np.zeros_like(pt_mag).astype(np.float32)
            pl_type_one_hot_first = map_polylines[:, :-1, 9:29]
            d['map_point']['type'] = np.argmax(
                pl_type_one_hot_first, axis=-1
            ).reshape(-1)[pt_valid_mask]

            num_polylines, num_points_per_polyline = map_polylines.shape[0], map_polylines.shape[1] - 1
            point_index = np.arange(num_polylines * num_points_per_polyline, dtype=np.int64)
            polygon_index = np.repeat(np.arange(num_polylines, dtype=np.int64), num_points_per_polyline)
            valid_point_index = point_index[pt_valid_mask]
            valid_polygon_index = polygon_index[pt_valid_mask]
            new_point_index = np.arange(valid_point_index.shape[0], dtype=np.int64)
            edge_index = np.stack([new_point_index, valid_polygon_index], axis=0)
            d['map_point', 'to', 'map_polygon'] = {'edge_index': edge_index}
            d['center_objects_type'] = i['center_objects_type']
            
            d = self.target_transform(d)

            data.append(self.token_processor.preprocess(d))
        return data
    
    def collate_fn(self, data_list):
        if isinstance(data_list[0], List):
            data_list = data_list[0]
        batch_size = len(data_list)
        merged_data = HeteroData()
        
        single_keys = ['scenario_id', 'kalman_difficulty', 'trajectory_type', 'center_objects_type']
        all_keys = ['map_save', 'pt_token', 'agent', 'map_point', 'map_polygon', ('map_point', 'to', 'map_polygon')]
        sample = data_list[0]
        pt_offset = 0
        pl_offset = 0
        for key in all_keys:
            merged_data[key] = {}
            if key != ('map_point', 'to', 'map_polygon'):
                for sub_key in sample[key].keys():
                    if sub_key == 'num_nodes':
                        continue
                    try:
                        merged_data[key][sub_key] = torch.from_numpy(
                                np.concatenate([data[key][sub_key] for data in data_list], axis=0)
                            )
                    except Exception as e:
                        merged_data[key][sub_key] = torch.tensor(
                                [data[key][sub_key] for data in data_list]
                        )
            else:
                merged_edges = []
                for data in data_list:
                    edge_index = data[key]['edge_index']
                    edge_index[0] += pt_offset
                    edge_index[1] += pl_offset
                    merged_edges.append(edge_index)
                    pt_offset += edge_index[0].max().item() + 1
                    pl_offset += edge_index[1].max().item() + 1
                merged_data[key]['edge_index'] = torch.from_numpy(
                    np.concatenate(merged_edges, axis=1)
                )
            if key not in ['map_save', ('map_point', 'to', 'map_polygon')]:
                ptr = torch.tensor([0])
                num_nodes_list = [data[key]['num_nodes'] for data in data_list]
                ptr = torch.cat([ptr, torch.tensor(num_nodes_list, dtype=torch.int64)])
                merged_data[key]['ptr'] = ptr
                merged_data[key]['num_nodes'] = torch.tensor(sum(num_nodes_list))
                merged_data[key]['batch'] = torch.arange(len(ptr)).repeat_interleave(ptr) 
        return merged_data
    
class WaymoTargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int,
                 mode="train") -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.num_features = 3
        self.augment = False
        self.logger = Logging().log(level='DEBUG')

    def score_ego_agent(self, agent):
        av_index = agent['av_index']
        agent["category"][av_index] = 5
        return agent

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent['valid_mask']
        ego_pos = agent["position"][av_index]
        obstacle_mask = agent['type'] == 3
        distance = torch.norm(agent["position"][:, self.num_historical_steps-1, :2] - ego_pos[self.num_historical_steps-1, :2], dim=-1)  # keep the closest 100 vehicles near the ego car
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[:max_num]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        excluded = ["num_nodes", "av_index", "ego"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def score_nearby_vehicle(self, agent, max_num=10):
        av_index = agent['av_index']
        agent["category"] = torch.zeros_like(agent["category"])
        obstacle_mask = agent['type'] == 3
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1)
        distance[obstacle_mask] = 10e5
        sort_idx = distance.sort()[1]
        nearby_mask = torch.zeros(distance.shape[0])
        nearby_mask[sort_idx[1:max_num]] = 1
        nearby_mask = nearby_mask.bool()
        agent["category"][nearby_mask] = 3
        agent["category"][obstacle_mask] = 0

    def score_trained_vehicle(self, agent, max_num=10, min_distance=0):
        av_index = agent['av_index']
        agent["category"] = np.zeros_like(agent["category"])
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = np.linalg.norm(agent["position"][:, self.num_historical_steps, :2] - pos, axis=-1)
        distance_all_time = np.linalg.norm(agent["position"][:, :, :2] - agent["position"][av_index, :, :2], axis=-1)
        invalid_mask = distance_all_time < 150  # we do not believe the perception out of range of 150 meters
        agent["valid_mask"] = agent["valid_mask"] * invalid_mask
        # we do not predict vehicle  too far away from ego car
        closet_vehicle = distance < 100
        valid = agent['valid_mask']
        valid_current = valid[:, self.num_historical_steps:]
        valid_counts = valid_current.sum(axis=1)
        counts_vehicle = valid_counts >= 1
        no_backgroud = agent['type'] != 3
        vehicle2pred = closet_vehicle & counts_vehicle & no_backgroud
        if vehicle2pred.sum() > max_num:
            # too many still vehicle so that train the model using the moving vehicle as much as possible
            true_indices = np.nonzero(vehicle2pred)[0]
            selected_indices = np.random.choice(true_indices, max_num, replace=False)
            vehicle2pred[:] = False
            vehicle2pred[selected_indices] = True
        agent["category"][vehicle2pred] = 3

    def rotate_agents(self, position, heading, num_nodes, num_historical_steps, num_future_steps):
        origin = position[:, num_historical_steps - 1]
        theta = heading[:, num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target = origin.new_zeros(num_nodes, num_future_steps, 4)
        target[..., :2] = torch.bmm(position[:, num_historical_steps:, :2] -
                                    origin[:, :2].unsqueeze(1), rot_mat)
        his = origin.new_zeros(num_nodes, num_historical_steps, 4)
        his[..., :2] = torch.bmm(position[:, :num_historical_steps, :2] -
                                 origin[:, :2].unsqueeze(1), rot_mat)
        if position.size(2) == 3:
            target[..., 2] = (position[:, num_historical_steps:, 2] -
                              origin[:, 2].unsqueeze(-1))
            his[..., 2] = (position[:, :num_historical_steps, 2] -
                           origin[:, 2].unsqueeze(-1))
            target[..., 3] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 3] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        else:
            target[..., 2] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 2] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        return his, target

    def __call__(self, data) -> HeteroData:
        agent = data["agent"]
        self.score_ego_agent(agent)
        self.score_trained_vehicle(agent, max_num=32)
        return HeteroData(data)