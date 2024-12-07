import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.transforms import BaseTransform

from .base_dataset import BaseDataset
from unitraj.datasets.common_utils import get_kalman_difficulty, get_trajectory_type,\
    wrap_angle

class QCNetDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)
        
    def postprocess(self, output):

        # Add the trajectory difficulty
        get_kalman_difficulty(output)

        # Add the trajectory type (stationary, straight, right turn...)
        get_trajectory_type(output)

        assert len(output) == 1
        output = dict_to_heterodata(output[0])
        return output
    
    def collate_fn(self, data_list):
        batch_data = HeteroData()

        # scenario_id, map_center
        batch_data['scenario_id'] = [data[0]['scenario_id'] for data in data_list]
        batch_data['map_center'] = torch.tensor(np.stack([data[0]['map_center'] for data in data_list]), dtype=torch.float32)

        # agent, map_point, map_polygon, map_polylines
        for key in ['agent', 'map_point', 'map_polygon', 'map_polylines']:
            batch_data[key] = {}
            for data in data_list:
                for sub_key in data[0][key].keys():
                    if sub_key == 'num_nodes':
                            if sub_key not in batch_data[key]:
                                batch_data[key][sub_key] = data[0][key][sub_key]
                            else:
                                batch_data[key][sub_key] += data[0][key][sub_key]
                    else:
                        batch_data[key][sub_key] = torch.cat(
                            [torch.tensor(data[0][key][sub_key]) for data in data_list], dim=0
                        )
        # indexing
        batch_data['map_point', 'to', 'map_polygon'].edge_index = torch.tensor(
            np.stack([data[0]['map_point', 'to', 'map_polygon'].edge_index for data in data_list])
        )
        
        target_transform = TargetBuilder(50, 60)
        batch_data = target_transform(batch_data)

        return batch_data

class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] -
                                                     theta.unsqueeze(-1))
        return data

def dict_to_heterodata(input_dict):
    # input_dict = data['input_dict']
    # device = input_dict['obj_trajs'].device
    num_historical_steps = input_dict['obj_trajs'].shape[1]
    num_future_steps = input_dict['obj_trajs_future_state'].shape[1]
    
    hetero_data = HeteroData()
    num_agents = input_dict["obj_trajs_pos"].shape[0]
    hetero_data['scenario_id'] = input_dict["scenario_id"]
    hetero_data['agent']['num_nodes'] = num_agents

    # concate past and future masks
    valid_mask = np.concatenate(
        (input_dict['obj_trajs_mask'],
         input_dict['obj_trajs_future_mask']), axis=-1
    )
    hetero_data['agent']['valid_mask'] = valid_mask.astype(int)
    
    num_steps = num_historical_steps + num_future_steps
    predict_mask = np.zeros((num_agents, num_steps), dtype=np.bool_)
    predict_mask[:, :num_historical_steps] = False
    predict_mask[:, num_historical_steps:] = True
    predict_mask[~valid_mask.astype(bool)] = False
    hetero_data['agent']['predict_mask'] = predict_mask
    
    # agent positions
    obj_future_pos = input_dict['obj_trajs_future_state'][..., :3]
    obj_future_pos[:, :, 2][:] = 0.
    hetero_data['agent']['position'] = np.concatenate([input_dict["obj_trajs_pos"], obj_future_pos], axis=1)
    obj_his_velo = input_dict['obj_trajs'][..., 35:37]
    obj_future_velo = input_dict['obj_trajs_future_state'][..., -2:]
    obj_full_velo = np.concatenate([obj_his_velo, obj_future_velo], axis=1)
    hetero_data['agent']['velocity'] = np.pad(
        obj_full_velo, pad_width=( (0, 0), (0, 0), (0, 1) ), mode='constant', constant_values=0.0
    )
    obj_his_heading_encoding = input_dict['obj_trajs'][..., 35:37]
    obj_his_heading = np.arctan2(obj_his_heading_encoding[..., 0], obj_his_heading_encoding[..., 1])
    obj_future_heading = input_dict['obj_trajs_future_state'][..., 2]
    hetero_data['agent']['heading'] = np.concatenate([obj_his_heading, obj_future_heading], axis=1)
    obj_type = input_dict['obj_trajs'][:, 0, 11:33]
    hetero_data['agent']['type'] = np.argmax(obj_type, axis=1)
    agent_category = np.zeros(num_agents, dtype=np.int64)
    if input_dict['track_index_to_predict'] != 0:
        agent_category[input_dict['track_index_to_predict']] = 2
    hetero_data['agent']['category'] = agent_category

    # map points
    map_polylines = input_dict['map_polylines']
    polyline_mask = ~np.all(map_polylines == 0, axis=(1, 2))
    map_polylines = map_polylines[polyline_mask]
    hetero_data['map_point']['num_nodes'] = map_polylines.shape[0] * (map_polylines.shape[1]-1)
    polylines_position = map_polylines[..., 0:3]  # shape: (768, 30, 3)
    polylines_direction = map_polylines[..., 3:6]  # shape: (768, 30, 3)
    hetero_data["map_point"]['position'] = polylines_position[:, :-1].reshape(-1, 3)
    hetero_data["map_point"]['orientation'] = np.arctan2(polylines_direction[:, :-1, 0], polylines_direction[:, :-1, 1]).reshape(-1)
    hetero_data["map_point"]['magnitude'] = np.linalg.norm(
        (map_polylines[:, 1:, 0:3] - map_polylines[:, :-1, 0:3]), axis=-1
    ).reshape(-1)
    hetero_data["map_point"]['type'] = np.zeros_like(map_polylines[:, :-1, 0].reshape(-1), dtype=np.uint8)
    hetero_data["map_point"]['side'] = np.zeros_like(map_polylines[:, :-1, 0].reshape(-1), dtype=np.uint8)
    
    # map polygons --> use polyline info instead
    hetero_data['map_polygon']['num_nodes'] = map_polylines.shape[0]
    hetero_data['map_polygon']['position'] = np.mean(
        map_polylines[..., 0:3], axis=1
    )
    start_points = map_polylines[..., 0:3][:, 0, :2]
    second_points = map_polylines[..., 0:3][:, 1, :2]
    hetero_data['map_polygon']['orientation'] = np.arctan2(
        (second_points - start_points)[:, 1], (second_points - start_points)[:, 0]
    )
    pl_type_one_hot = map_polylines[:, 0, 9:29]
    hetero_data['map_polygon']['type'] = np.argmax(pl_type_one_hot, axis=-1)
    hetero_data['map_polygon']['is_intersection'] = np.zeros(map_polylines.shape[0], dtype=np.uint8)
    
    # map polylines
    hetero_data['map_center'] = input_dict['map_center']
    hetero_data['map_polylines']['position'] = map_polylines[..., 0:3]
    hetero_data['map_polylines']['orientation'] = map_polylines[..., 3:6]
    hetero_data['map_polylines']['lane_type'] = map_polylines[..., 9:29]
    hetero_data['map_polylines']['mask'] = input_dict['map_polylines_mask'][polyline_mask]
    hetero_data['map_polylines']['center'] = input_dict['map_polylines_center'][polyline_mask]
    
    # pt & pl indexing
    num_polylines, num_points_per_polyline = map_polylines.shape[0], map_polylines.shape[1] - 1
    point_index = np.arange(num_polylines * num_points_per_polyline, dtype=np.int64)
    polygon_index = np.repeat(np.arange(num_polylines, dtype=np.int64), num_points_per_polyline)
    hetero_data['map_point', 'to', 'map_polygon']['edge_index'] = np.stack([point_index, polygon_index], axis=0)
    
    return [hetero_data]
