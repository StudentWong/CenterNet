import torch
from torch import nn
from MembershipBackend import MemberShip_Forward_Wrapper
from MembershipBackend import MemberShip_Input_Backward_Wrapper
from MembershipBackend import MemberShip_Center_Backward_Wrapper
from MembershipBackend import MemberShip_Lamda_Backward_Wrapper
from MembershipBackend import CenterLoss_Forward_Wrapper

if __name__ == "__main__":
    N = 100
    D = 128
    C = 16
    # x = torch.rand((N, D), dtype=torch.float).cuda()
    # c = torch.rand((D, C), dtype=torch.float).cuda()
    # gt = torch.rand((N, C), dtype=torch.float).cuda()
    # gts = torch.rand((N), dtype=torch.float).cuda()
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float).cuda()
    c = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float).cuda()
    gt = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]], dtype=torch.float).cuda()
    gts = torch.tensor([1, 2, 3], dtype=torch.float).cuda()
    y = CenterLoss_Forward_Wrapper(x, c, gt, gts)
    print(y)