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

model = nn.Sequential(conv1, conv2, conv3).eval() # issue: conv2 fails when trying to convert to mlir (because depthwise conv?)

mlir = torch_mlir.fx.export_and_import(
    model,
    input,
    output_type=OutputType.LINALG_ON_TENSORS)

print(mlir)