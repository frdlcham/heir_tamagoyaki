import torch
from torch import nn
import torch_mlir
from torch_mlir.fx import OutputType

# SHAPES
T = 5
H = 28
W = 28

C_in_1 = 32 # channels input to first conv
C_out_1 = 184

C_in_2 = C_out_1
C_out_2 = C_in_2 # 2nd conv keeps channels the same

C_in_3 = C_out_2
C_out_3 = 56

STRIDE_conv_1 = 1
STRIDE_conv_2 = 2
STRIDE_conv_3 = 1

# Conv3d(in_channels, out_channels, kernel_size, 
# stride=1, padding=0, dilation=1, groups=1, bias=True, 
# padding_mode='zeros', device=None, dtype=None)

conv1 = nn.Conv3d(in_channels=C_in_1, out_channels=C_out_1, 
                  kernel_size=(1, 1, 1), stride=STRIDE_conv_1,
                  groups=1,
                  padding='same', bias=True)

conv2 = nn.Conv3d(in_channels=C_in_2, out_channels=C_out_2, 
                  kernel_size=(T, 3, 3), stride=STRIDE_conv_2,
                  groups=C_in_2, # this is what makes it depthwise
                  #padding='same', # ValueError: padding='same' is not supported for strided convolutions
                  bias=True)

conv3 = nn.Conv3d(in_channels=C_in_3, out_channels=C_out_3, 
                  kernel_size=(1, 1, 1), stride=STRIDE_conv_3,
                  groups=1,
                  padding='same', bias=True)

input = torch.randn((1, C_in_1, T, H, W))


# output = conv3(conv2(conv1(input)))
# print(output)

class DepthwiseAsGrouped2d(nn.Module):
    """Exact rewrite of a depthwise Conv3d whose kernel depth equals the input depth
    (no temporal padding), as a grouped Conv2d over (C*T) channels."""
    def __init__(self, conv3d: nn.Conv3d):
        super().__init__()
        C = conv3d.in_channels
        kT, kH, kW = conv3d.kernel_size
        self.conv2d = nn.Conv2d(C * kT, C, (kH, kW),
                                stride=conv3d.stride[1:],
                                groups=C,
                                bias=conv3d.bias is not None)
        with torch.no_grad():
            # (C, 1, kT, kH, kW) -> (C, kT, kH, kW)
            self.conv2d.weight.copy_(conv3d.weight.reshape(C, kT, kH, kW))
            if conv3d.bias is not None:
                self.conv2d.bias.copy_(conv3d.bias)

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = x.reshape(N, C * T, H, W)        # channel index = c*T + t
        return self.conv2d(x).unsqueeze(2)   # restore depth dim of size 1


model = nn.Sequential(conv1, DepthwiseAsGrouped2d(conv2), conv3).eval()
# model = nn.Sequential(conv1, conv2, conv3).eval() # issue: conv2 fails when trying to convert to mlir (because depthwise conv?)

mlir = torch_mlir.fx.export_and_import(
    model,
    input,
    output_type=OutputType.LINALG_ON_TENSORS)

print(mlir)
print(mlir.operation.get_asm(large_elements_limit=16, large_resource_limit=16)) # nicer way to print this (makes it actually readable)