import torch
from torch import nn
from MembershipBackend import MemberShip_Forward_Wrapper
from MembershipBackend import MemberShip_Input_Backward_Wrapper
from MembershipBackend import MemberShip_Center_Backward_Wrapper
from MembershipBackend import MemberShip_Lamda_Backward_Wrapper
# from MembershipBackend import CenterLoss_Forward_Wrapper

min_clip = 1e-6

class _menbership_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, lamda):
        # x: N*D*(w*h)
        # c: D*C
        # lamda: D*C
        # output: N*C*(w*h)

        # print(type(x))
        assert len(x.shape) == 3 and  len(c.shape) == 2 and len(lamda.shape) == 2
        N = x.shape[0]
        wh = x.shape[2]
        assert x.shape[1] == c.shape[0] and x.shape[1] == lamda.shape[0]
        D = x.shape[1]
        assert c.shape[1] == lamda.shape[1]
        C = c.shape[1]

        x_squeeze = x.permute((0, 2, 1)).contiguous().view(N*wh, D)
        assert isinstance(x_squeeze, torch.cuda.FloatTensor) \
        and isinstance(c, torch.cuda.FloatTensor) \
        and isinstance(lamda, torch.cuda.FloatTensor) \
        and x_squeeze.is_contiguous() \
        and c.is_contiguous() \
        and lamda.is_contiguous()

        out = torch.zeros((N*wh, C), dtype=torch.float).cuda()

        
        MemberShip_Forward_Wrapper(x_squeeze, c, lamda, out)
        out = out.clamp(min=min_clip)
        ctx.save_for_backward(x_squeeze.detach(), c.detach(), lamda.detach(), out.detach(), torch.tensor(N, dtype=torch.int16), torch.tensor(D, dtype=torch.int16), torch.tensor(C, dtype=torch.int16), torch.tensor(wh, dtype=torch.int16))

        assert out.is_contiguous()
        output = out.view(N, wh, C).permute((0, 2, 1)).contiguous()


        return output

    @staticmethod
    def backward(ctx, grad_output):
        # x: N*D*(w*h)
        # c: D*C
        # lamda: D*C
        # output: N*C*(w*h)
        #grad_out = N*C*(w*h)

        grad_x = grad_c = grad_la = None
        x_squeeze, c, lamda, out, N, D, C, wh = ctx.saved_variables

        N = int(N)
        D = int(D)
        C = int(C)
        wh = int(wh)
        assert len(grad_output.shape) == 3 and grad_output.shape[0] == N \
            and grad_output.shape[1] == C and grad_output.shape[2] == wh

        grad_output_squeeze = grad_output.permute((0, 2, 1)).contiguous().view(N*wh, C)
        # exit()
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros((N*wh, D), dtype=torch.float).cuda()
            MemberShip_Input_Backward_Wrapper(grad_x, grad_output_squeeze, x_squeeze, c, lamda, out)
            assert grad_x.is_contiguous()
            grad_x = grad_x.view(N, wh, D).permute((0, 2, 1)).contiguous()

        if ctx.needs_input_grad[1]:
            grad_c = torch.zeros((D, C), dtype=torch.float).cuda()
            MemberShip_Center_Backward_Wrapper(grad_c, grad_output_squeeze, x_squeeze, c, lamda, out)
            assert grad_c.is_contiguous()

        if ctx.needs_input_grad[2]:
            grad_la = torch.zeros((D, C), dtype=torch.float).cuda()
            MemberShip_Lamda_Backward_Wrapper(grad_la, grad_output_squeeze, x_squeeze, c, lamda, out)
            assert grad_la.is_contiguous()


        # print(grad_x)
        return grad_x, grad_c, grad_la


class Membership_norm_cuda(nn.Module):
    def __init__(self, feature, class_num, init_c=None, init_lamda=None, c_grad=True, lamda_grad=True):
        super(Membership_norm_cuda, self).__init__()

        if init_c is None:
            self.c = nn.Parameter(data=0.0*torch.ones((feature, class_num), dtype=torch.float),
                                  requires_grad=c_grad)
            # self.c = nn.Parameter(data=torch.rand((feature, class_num), dtype=torch.float),
                                #   requires_grad=c_grad)
        else:
            assert len(init_c.shape) == 2 and \
                   init_c.shape[0] == feature and \
                   init_c.shape[1] == class_num
            self.c = nn.Parameter(data=init_c, requires_grad=c_grad)

        if init_lamda is None:
            self.lamda = nn.Parameter(data=1.0*torch.ones((feature, class_num), dtype=torch.float),
                                      requires_grad=lamda_grad)
            # self.lamda = nn.Parameter(data=torch.rand((feature, class_num), dtype=torch.float),
                                    #   requires_grad=lamda_grad)
        else:
            assert len(init_lamda.shape) == 2 and \
                   init_lamda.shape[0] == feature and \
                   init_lamda.shape[1] == class_num
            self.lamda = nn.Parameter(data=init_lamda, requires_grad=lamda_grad)

        self.fun = _menbership_function

    def forward(self, input):
        # input shape: N*D*(w*h)
        # c shape: D*C
        # lamda shape : D*C
        # output: N*C*(w*h)
        return self.fun.apply(input, self.c, self.lamda)
