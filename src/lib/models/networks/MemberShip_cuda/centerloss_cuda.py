import torch
from torch import nn
from MembershipBackend import CenterLoss_Forward_Wrapper
from MembershipBackend import CenterLoss_Input_Backward_Wrapper
from MembershipBackend import CenterLoss_Center_Backward_Wrapper


class _centerloss_matrix_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, gt, gt_s):
        # x: N*D*(w*h)
        # c: D*C
        # gt: N*C*(w*h)
        # gt_s: N*(w*h)

        # print(type(x))
        assert len(x.shape) == 3 and  len(c.shape) == 2 and len(gt.shape) == 3 and len(gt_s.shape) == 2
        assert x.shape[0] == gt.shape[0] and gt_s.shape[0] == x.shape[0]
        N = x.shape[0]
        assert x.shape[2] == gt.shape[2] and x.shape[2] == gt_s.shape[1]
        wh = x.shape[2]
        assert x.shape[1] == c.shape[0]
        D = x.shape[1]
        assert c.shape[1] == gt.shape[1]
        C = c.shape[1]

        x_squeeze = x.permute((0, 2, 1)).contiguous().view(N*wh, D)
        gt_squeeze = gt.permute((0, 2, 1)).contiguous().view(N*wh, C)
        assert gt_s.is_contiguous()
        gt_sum = gt_s.view(N*wh)


        assert isinstance(x_squeeze, torch.cuda.FloatTensor) \
        and isinstance(c, torch.cuda.FloatTensor) \
        and isinstance(gt_squeeze, torch.cuda.FloatTensor) \
        and isinstance(gt_sum, torch.cuda.FloatTensor) \
        and x_squeeze.is_contiguous() \
        and c.is_contiguous() \
        and gt_squeeze.is_contiguous()\
         and gt_sum.is_contiguous()

        # out = torch.zeros((N*wh, C), dtype=torch.float).cuda()

        
        out = CenterLoss_Forward_Wrapper(x_squeeze, c, gt_squeeze, gt_sum)
        # out = out.clamp(min=min_clip)
        ctx.save_for_backward(x_squeeze.detach(), c.detach(), gt_squeeze.detach(), gt_sum.detach(), torch.tensor(N, dtype=torch.int16), torch.tensor(D, dtype=torch.int16), torch.tensor(C, dtype=torch.int16), torch.tensor(wh, dtype=torch.int16))

        assert out.is_contiguous()
        output = out.view(N, wh, D).permute((0, 2, 1)).contiguous()


        return output

    @staticmethod
    def backward(ctx, grad_output):
        # x: N*D*(w*h)
        # c: D*C
        # lamda: D*C
        # output: N*C*(w*h)
        #grad_out = N*C*(w*h)

        grad_x = grad_c = grad_gt = grad_gts = None
        x_squeeze, c, gt_squeeze, gt_sum, N, D, C, wh = ctx.saved_variables

        N = int(N)
        D = int(D)
        C = int(C)
        wh = int(wh)
        assert len(grad_output.shape) == 3 and grad_output.shape[0] == N \
            and grad_output.shape[1] == D and grad_output.shape[2] == wh

        grad_output_squeeze = grad_output.permute((0, 2, 1)).contiguous().view(N*wh, D)
        # exit()
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros((N*wh, D), dtype=torch.float).cuda()
            CenterLoss_Input_Backward_Wrapper(grad_x, grad_output_squeeze, x_squeeze, c, gt_squeeze,gt_sum)
            assert grad_x.is_contiguous()
            grad_x = grad_x.view(N, wh, D).permute((0, 2, 1)).contiguous()

        if ctx.needs_input_grad[1]:
            grad_c = torch.zeros((D, C), dtype=torch.float).cuda()
            CenterLoss_Center_Backward_Wrapper(grad_c, grad_output_squeeze, x_squeeze, c, gt_squeeze, gt_sum)
            assert grad_c.is_contiguous()

        if ctx.needs_input_grad[2]:
            print("error")
            # grad_la = torch.zeros((D, C), dtype=torch.float).cuda()
            # MemberShip_Lamda_Backward_Wrapper(grad_la, grad_output_squeeze, x_squeeze, c, lamda, out)
            # assert grad_la.is_contiguous()
        if ctx.needs_input_grad[3]:
            print("error")

        # print(grad_x)
        return grad_x, grad_c, grad_gt, grad_gts


class CenterLoss_gt_cuda(nn.Module):
    '''nn.Module warpper for center loss'''

    def __init__(self):
        super(CenterLoss_gt_cuda, self).__init__()
        self.center_loss = _centerloss_matrix_function

    def forward(self, x, c, act, gt_hm):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)
        # gt_hm: N*C*W*H
        assert x.shape[0] == act.shape[0] \
               and gt_hm.shape[0] == act.shape[0] \
               and len(x.shape) == 3
        N = x.shape[0]
        assert x.shape[1] == c.shape[0] and len(c.shape) == 2
        D = x.shape[1]
        assert c.shape[1] == act.shape[1] and act.shape[1] == gt_hm.shape[1]
        C = c.shape[1]
        assert len(act.shape) == 3 \
               and x.shape[2] == act.shape[2] \
               and gt_hm.shape[2] * gt_hm.shape[3] == x.shape[2] \
               and gt_hm.is_contiguous()
        wh = x.shape[2]
        gt_hm = gt_hm.view(N, C, wh)

        # gt_hm = gt_hm.eq(1).float()

        gts = gt_hm.sum(dim=1)
        num_pos = gt_hm.sum()
        loss_matrix = self.center_loss.apply(x, c, gt_hm, gts)
        # print(loss_matrix)
        return loss_matrix.sum()/num_pos

class CenterLoss_gt_eq1_cuda(nn.Module):
    '''nn.Module warpper for center loss'''

    def __init__(self):
        super(CenterLoss_gt_eq1_cuda, self).__init__()
        self.center_loss = _centerloss_matrix_function

    def forward(self, x, c, act, gt_hm):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)
        # gt_hm: N*C*W*H
        assert x.shape[0] == act.shape[0] \
               and gt_hm.shape[0] == act.shape[0] \
               and len(x.shape) == 3
        N = x.shape[0]
        assert x.shape[1] == c.shape[0] and len(c.shape) == 2
        D = x.shape[1]
        assert c.shape[1] == act.shape[1] and act.shape[1] == gt_hm.shape[1]
        C = c.shape[1]
        assert len(act.shape) == 3 \
               and x.shape[2] == act.shape[2] \
               and gt_hm.shape[2] * gt_hm.shape[3] == x.shape[2] \
               and gt_hm.is_contiguous()
        wh = x.shape[2]
        gt_hm = gt_hm.view(N, C, wh)

        gt_hm = gt_hm.eq(1).float()

        gts = gt_hm.sum(dim=1)
        num_pos = gt_hm.sum()
        loss_matrix = self.center_loss.apply(x, c, gt_hm, gts)
        # print(loss_matrix)
        return loss_matrix.sum()/num_pos