# inspection(ok): normalize[0,1][-1,1]cc_obsmax
import pandas as pd
import torch


class OrderState:
    def __init__(self, pick_up_position, drop_off_position, pick_up_position_id, drop_off_position_id, price, env_config):
        self.status = 'wait_pair'  # wait_pair, wait_pick, picked_up, 'dropped'
        self.pick_up_position_id = pick_up_position_id
        self.drop_off_position_id = drop_off_position_id
        self.pick_up_position = pick_up_position
        self.drop_off_position = drop_off_position
        self.price = price
        self.waiting_steps = 0
        self.env_config = env_config

        # self.env_config = env_config

    def set_zero(self):
        self.pick_up_position = self.drop_off_position = [0, 0]
        self.pick_up_position_id = self.drop_off_position_id = 0

    def to_tuple(self, normalize=False):
        if normalize:
            return (#self.pick_up_position[0] / self.env_config['max_x'],
                    #self.pick_up_position[1] / self.env_config['max_y'],
                    self.drop_off_position[0] / self.env_config['max_x'],
                    self.drop_off_position[1] / self.env_config['max_y'],
                    )
        else:
            # return self.pick_up_position[0], self.pick_up_position[1], self.drop_off_position[0], self.drop_off_position[1]
            return self.drop_off_position[0], self.drop_off_position[1]


class CarState:
    def __init__(self, position, position_id, env_config):
        self.status = 'idle'  # idle, picking_up, dropping_off
        self.position = position
        self.position_id = position_id
        self.order = None
        self.path = []
        self.required_steps = None
        self.travel_distance = 0
        self.env_config = env_config

        # self.env_config = env_config

    def set_zero(self):
        self.position = [0, 0]
        self.position_id = 0

    def to_tuple(self, normalize=False):
        if normalize:
            return (self.position[0] / self.env_config['max_x'],
                        self.position[1] / self.env_config['max_y'],
                        )
        else:
            return self.position[0], self.position[1]


class JointState:
    def __init__(self, order_states, car_states):
        for order_state in order_states:
            assert isinstance(order_state, OrderState)
        for car_state in car_states:
            assert isinstance(car_state, CarState)

        self.order_states = order_states
        self.car_states = car_states

    def to_tensor(self, add_batchsize_dim=False, device=None, normalize=False):
        order_tensor = torch.tensor([order_state.to_tuple(normalize=True if normalize else False) for order_state in self.order_states],
                                           dtype=torch.float32)
        car_tensor = torch.tensor([car_state.to_tuple(normalize=True if normalize else False) for car_state in self.car_states],
                                           dtype=torch.float32)

        if add_batchsize_dim:
            order_tensor = order_tensor.unsqueeze(0)
            car_tensor = car_tensor.unsqueeze(0)

        if device is not None:
            order_tensor = order_tensor.to(device)
            car_tensor = car_tensor.to(device)

        # order car tensor
        return car_tensor, order_tensor
