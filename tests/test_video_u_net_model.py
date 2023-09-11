import unittest

import torch
from models.Video_U_Net import Video_U_Net


class TestVideoUNetModel(unittest.TestCase):
    def setUp(self):
        self.u_net = Video_U_Net()
        
        self.video_num = 1
        self.video_C = 3  # Channels
        self.video_F = 1  # Frames
        self.video_H = 64  # Height
        self.video_W = 64  # Width

        self.time_val = 1_000

        self.x_T = torch.randn(
            (
                self.video_num,
                self.video_C,
                self.video_F,
                self.video_H,
                self.video_W))
        self.time_step = torch.tensor([self.time_val], dtype=torch.long)
    
    def test_unet_forward_pass(self):
        x_0 = self.u_net(self.x_T, self.time_step)
        self.assertEqual(
            x_0.shape,
            (
                self.video_num,
                self.video_C,
                self.video_F,
                self.video_H,
                self.video_W))
