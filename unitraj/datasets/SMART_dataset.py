import os
import pickle
import torch
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from unitraj.models.smart.utils.log import Logging
import numpy as np
from unitraj.models.smart.datasets.preprocess import TokenProcessor
from .base_dataset import BaseDataset
from unitraj.datasets.common_utils import get_kalman_difficulty, get_trajectory_type

class SMARTDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self.token_processor = TokenProcessor(2048)
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

            agent['position'] = np.concatenate([i['obj_trajs_pos'], i['obj_trajs_future_state'][..., :3]], axis=1)

            obj_his_heading_encoding = i['obj_trajs'][..., 23:25]
            obj_his_heading = np.arctan2(obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1])
            obj_future_heading = i['obj_trajs_future_state'][..., 2]
            agent['heading'] = np.concatenate([obj_his_heading, obj_future_heading], axis=1)

            velo_all = np.concatenate(
                [i['obj_trajs'][..., 25:27], i['obj_trajs_future_state'][..., 3:5]], axis=1
            )
            agent['velocity'] = np.pad(velo_all, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0.0)

            agent['center_objects_world'] = i['center_objects_world']
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
            d['map_point']['position'] = pt_pos
            d['map_point']['orientation'] = pt_ori

            pt_mag = np.linalg.norm(
                (map_polylines[:, 1:, 0:3] - map_polylines[:, :-1, 0:3]), axis=-1
            ).reshape(-1)[pt_valid_mask]
            d['map_point']['magnitude'] = pt_mag
            d['map_point']['height'] = np.zeros_like(pt_mag)
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
            
            d = self.convert_all_numpy_to_tensor(d)
            
            data.append(self.token_processor.preprocess(d))
        print(11111)
        return data
    
    def convert_all_numpy_to_tensor(self, data, dtype=None):
        if isinstance(data, dict):
            return {k: self.convert_all_numpy_to_tensor(v, dtype) for k, v in data.items()}

        elif isinstance(data, list):
            return [self.convert_all_numpy_to_tensor(item, dtype) for item in data]
        
        elif isinstance(data, tuple):
            return tuple(self.convert_all_numpy_to_tensor(item, dtype) for item in data)
        
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
            return tensor.to(dtype) if dtype else tensor

        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return torch.tensor(data, dtype=dtype if dtype else torch.int64)
        
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return torch.tensor(data, dtype=dtype if dtype else torch.float32)
        
        elif isinstance(data, np.str_):
            return str(data)
        
        else:
            return data
            
        # data = []
        # for i in output:
        #     d = {'scenario_id': i['scenario_id']}
        #     d['kalman_difficulty'] = i['kalman_difficulty']
        #     d['trajectory_type'] = i['trajectory_type']
        #     d['agent'] = {}
        #     d['agent']['num_nodes'] = i['obj_trajs'].shape[0]
        #     d['agent']['av_index'] = i['track_index_to_predict']
        #     valid_mask = torch.from_numpy(np.concatenate(
        #         (i['obj_trajs_mask'],
        #         i['obj_trajs_future_mask'].astype(bool)), axis=-1
        #     ))
        #     d['agent']['valid_mask'] = valid_mask
        #     predict_mask = torch.zeros_like(valid_mask, dtype=bool)
        #     num_historical_steps = i['obj_trajs'].shape[1]
        #     predict_mask[:, num_historical_steps:] = True
        #     predict_mask[~valid_mask] = False
        #     d['agent']['predict_mask'] = predict_mask
        #     d['agent']['type'] = torch.zeros(d['agent']['num_nodes'], dtype=torch.uint8)
        #     d['agent']['category'] = torch.from_numpy(
        #         np.argmax(i['obj_trajs'][:, 0, 6:11], axis=1)
        #     )
        #     d['agent']['position'] = torch.from_numpy(
        #         np.concatenate([i['obj_trajs_pos'], i['obj_trajs_future_state'][..., :3]], axis=1)
        #     )
        #     obj_his_heading_encoding = i['obj_trajs'][..., 23:25]
        #     obj_his_heading = np.arctan2(
        #         obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1]
        #     )
        #     obj_future_heading = i['obj_trajs_future_state'][..., 2]
        #     d['agent']['heading'] = torch.from_numpy(
        #         np.concatenate([obj_his_heading, obj_future_heading], axis=1)
        #     )
        #     velo_all = np.concatenate(
        #         [i['obj_trajs'][..., 25:27],
        #          i['obj_trajs_future_state'][..., 3:5]], axis=1
        #     )
        #     d['agent']['velocity'] = torch.from_numpy(np.pad(
        #         velo_all, pad_width=((0, 0), (0, 0), (0, 1)),
        #         mode='constant', constant_values=0.0
        #     ))
        #     d['agent']['center_objects_world'] = i['center_objects_world']
        #     # d['agent']['shape']

        #     d['map_polygon'] = {}
        #     map_polylines = i['map_polylines']
        #     polyline_mask = ~np.all(map_polylines == 0, axis=(1, 2))
        #     map_polylines = map_polylines[polyline_mask]
        #     d['map_polygon']['num_nodes'] = map_polylines.shape[0]
        #     d['map_polygon']['type'] = torch.from_numpy(np.argmax(
        #         map_polylines[:, 0, 9:29], axis=-1
        #     ))
        #     # d['map_polygon']['light_type']
            
        #     d['map_point'] = {}
        #     polylines_position = map_polylines[:, :-1, 0:3].reshape(-1, 3)
        #     polylines_orientation = map_polylines[:, :-1, 3:6].reshape(-1, 3)
        #     pt_valid_mask = (
        #         (map_polylines[:, :-1, 0] != 0) | (map_polylines[:, :-1, 1] != 0)
        #     ).flatten()
        #     pt_pos = polylines_position[pt_valid_mask]
        #     pt_ori = polylines_orientation[pt_valid_mask]
        #     d['map_point']['num_nodes'] = pt_pos.shape[0]
        #     d['map_point']['position'] = torch.from_numpy(pt_pos)
        #     d['map_point']['position'] = torch.from_numpy(pt_ori)
            
        #     pt_mag = np.linalg.norm(
        #         (map_polylines[:, 1:, 0:3] - map_polylines[:, :-1, 0:3]), axis=-1
        #     ).reshape(-1)[pt_valid_mask]
        #     d['map_point']['magnitude'] = torch.from_numpy(pt_mag)
        #     d['map_point']['height'] = torch.zeros_like(torch.tensor(pt_mag))
        #     pl_type_one_hot_first = map_polylines[:, :-1, 9:29]
        #     d['map_point']['type'] = torch.from_numpy(
        #         np.argmax(pl_type_one_hot_first, axis=-1)
        #     ).reshape(-1)[pt_valid_mask]
            
        #     num_polylines, num_points_per_polyline = map_polylines.shape[0], map_polylines.shape[1] - 1
        #     point_index = np.arange(num_polylines * num_points_per_polyline, dtype=np.int64)
        #     polygon_index = np.repeat(np.arange(num_polylines, dtype=np.int64), num_points_per_polyline)
        #     valid_point_index = point_index[pt_valid_mask]
        #     valid_polygon_index = polygon_index[pt_valid_mask]
        #     new_point_index = np.arange(valid_point_index.shape[0], dtype=np.int64)
        #     edge_index = np.stack([new_point_index, valid_polygon_index], axis=0)
        #     d['map_point', 'to', 'map_polygon'] = {}
        #     d['map_point', 'to', 'map_polygon']['edge_index'] = edge_index
            
        #     data.append(d)
        # return data

