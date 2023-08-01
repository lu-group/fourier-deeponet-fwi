import deepxde as dde
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierDeepONet(dde.nn.pytorch.NN):
    """MIONet with two input functions for Cartesian product format."""

    def __init__(
            self,
            num_parameter,
            width=64,
            modes1=20,
            modes2=20,
            regularization=None,
            merge_operation="mul",
    ):
        super().__init__()
        self.num_parameter = num_parameter
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.branch = Branch(self.width)
        self.trunk = Trunk(self.width, self.num_parameter)
        self.merger = decoder(self.modes1, self.modes2, self.width)
        self.b = nn.Parameter(torch.tensor(0.0))
        self.regularizer = regularization
        self.merge_operation = merge_operation

    def forward(self, inputs):
        x1 = self.branch(inputs[0])
        x2 = self.trunk(inputs[1])
        if self.merge_operation == "add":
            x = x1 + x2
        elif self.merge_operation == "mul":
            x = torch.mul(x1, x2)
        else:
            raise NotImplementedError(
                f"{self.merge_operation} operation to be implimented"
            )
        x = x + self.b
        x = self.merger(x)
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        # print("out_ft shape:",out_ft.shape)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(input_channels * 2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class decoder(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(decoder, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.unet1 = U_net(self.width, self.width, 3, 0)
        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 70)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet1(x)
        x = x1 + x2 + x3
        x = self.linear1(x)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 512)
        x3 = self.unet2(x)
        x = x1 + x2 + x3
        x = self.linear2(x)
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 256)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = self.linear3(x)
        x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.view(batchsize, 1, size_x, 70)[:, :, 1:-1, :]

        return x


class Branch(nn.Module):
    def __init__(self, width):
        super(Branch, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(5, self.width)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), "replicate")
        x = x.permute(0, 3, 2, 1)  # -1, 72, 1000, 5
        x = self.fc0(x)  # -1, 72, 1000, 64
        x = x.permute(0, 3, 1, 2)  # -1, 64, 72, 1000

        return x


class Trunk(nn.Module):
    def __init__(self, width, num_parameter):
        super(Trunk, self).__init__()
        self.width = width
        self.num_parameter = num_parameter
        self.fc0 = nn.Linear(self.num_parameter, self.width)

    def forward(self, x):
        x = self.fc0(x)

        return x[:, :, None, None]