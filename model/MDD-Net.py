# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.real_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.imag_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.real_conv1d(x[..., 0]) - self.imag_conv1d(x[..., 1])
        imag = self.real_conv1d(x[..., 1]) + self.imag_conv1d(x[..., 0])
        output = torch.stack((real, imag), dim=-1)
        return output

class ComplexBatchNormal1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True, complex_axis=1):
        super(ComplexBatchNormal1d, self).__init__()
        self.num_features = num_features//2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats 
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)
        
        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)
        
        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        # 均值
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
                
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        # 方差
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        tau   = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real_Linear = nn.Linear(in_features, out_features, bias)
        self.imag_Linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        real = self.real_Linear(x[..., 0]) - self.imag_Linear(x[..., 1])
        imag = self.real_Linear(x[..., 1]) + self.imag_Linear(x[..., 0])
        output = torch.stack((real, imag), dim=-1)
        return output

class fre_net(nn.Module):
    def __init__(self, input_length):
        super(fre_net, self).__init__()
        self.input_length = input_length
        self.input_size1 = 256
        self.input_size2 = 128
        self.features = 32
        self.RIConv00 = ComplexConv1d(1, self.features, 1)

        self.RITConv01 = ComplexConv1d(self.input_length, self.input_size1, kernel_size=3, padding=1)
        self.RITConv02 = ComplexConv1d(self.input_length, self.input_size1, kernel_size=5, padding=2)
        self.RITConv03 = ComplexConv1d(self.input_length, self.input_size1, kernel_size=7, padding=3)
        self.RITConv04 = ComplexConv1d(self.input_length, self.input_size1, kernel_size=1)

        self.RIConv01 = ComplexConv1d(self.input_size1, self.input_size2, kernel_size=3, padding=1)
        self.RIConv04 = ComplexConv1d(self.input_size1, self.input_size2, kernel_size=1)

        self.RI_fc1 = ComplexConv1d(self.input_size2, self.input_size2//4, kernel_size=1, bias=False)
        self.RI_fc2 = ComplexConv1d(self.input_size2//4, self.input_size2, kernel_size=1, bias=False)

        self.RIConv000 = ComplexConv1d(1, self.features, 1)
        self.RITConv11 = ComplexConv1d(self.input_size2, self.input_size1, kernel_size=3, padding=1)
        self.RITConv12 = ComplexConv1d(self.input_size2, self.input_size1, kernel_size=5, padding=2)
        self.RITConv13 = ComplexConv1d(self.input_size2, self.input_size1, kernel_size=7, padding=3)
        self.RITConv14 = ComplexConv1d(self.input_size2, self.input_size1, kernel_size=1)

        self.RIConv11 = ComplexConv1d(self.input_size1, self.input_length, kernel_size=3, padding=1)
        self.RIConv14 = ComplexConv1d(self.input_size1, self.input_length, kernel_size=1)

        self.RI_fc3 = ComplexConv1d(self.input_length, self.input_length//4, kernel_size=1, bias=False)
        self.RI_fc4 = ComplexConv1d(self.input_length//4, self.input_length, kernel_size=1, bias=False)
    
        self.RIOut = ComplexLinear(self.features*8, self.features*4, bias=True)
        self.RIOut2 = ComplexLinear(self.features*4, 1, bias=True)
        self.RIOut3 = ComplexLinear(self.features*8, self.features*4, bias=True)
        self.RIOut4 = ComplexLinear(self.features*4, 1, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.Prelu = nn.PReLU()
        self.RIbn = ComplexBatchNormal1d(self.input_length)
        self.RIbn2 = ComplexBatchNormal1d(self.input_size2)

    def forward(self, x):
        x = torch.fft.fft(x, dim=1)
        real, imag = x.real, x.imag
        ri = torch.stack((real, imag), -1)

        ri0 = self.RIConv00(ri.transpose(2, 1)).transpose(2, 1)
        ri0 = self.RIbn(ri0)

        ri1 = self.RITConv01(ri0)
        temp = self.RITConv02(ri0)
        ri1 = torch.cat((ri1, temp), 2)
        temp = self.RITConv03(ri0)
        ri1 = torch.cat((ri1, temp), 2)
        temp = self.RITConv04(ri0)
        ri1 = torch.cat((ri1, temp), 2)

        ri2 = self.RIConv01(ri1)
        temp = self.RIConv04(ri1)
        ri2 = torch.cat((ri2, temp), 2)
        
        avg_out = self.RI_fc2(self.Prelu(self.RI_fc1(torch.stack((self.avg_pool(ri2[:, :, :, 0]), self.avg_pool(ri2[:, :, :, 1])), -1))))
        max_out = self.RI_fc2(self.Prelu(self.RI_fc1(torch.stack((self.max_pool(ri2[:, :, :, 0]), self.max_pool(ri2[:, :, :, 1])), -1))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        ri2 = out * ri2

        riOut = self.RIOut(ri2)
        riOut = self.Prelu(riOut)
        riOut = self.RIOut2(riOut)

        ri0 = self.RIConv000(riOut.transpose(2, 1)).transpose(2, 1)
        ri0 = self.RIbn2(ri0)

        ri1 = self.RITConv11(ri0)
        temp = self.RITConv12(ri0)
        ri1 = torch.cat((ri1, temp), 2)
        temp = self.RITConv13(ri0)
        ri1 = torch.cat((ri1, temp), 2)
        temp = self.RITConv14(ri0)
        ri1 = torch.cat((ri1, temp), 2)

        ri2 = self.RIConv11(ri1)
        temp = self.RIConv14(ri1)
        ri2 = torch.cat((ri2, temp), 2)
        
        avg_out = self.RI_fc4(self.Prelu(self.RI_fc3(torch.stack((self.avg_pool(ri2[:, :, :, 0]), self.avg_pool(ri2[:, :, :, 1])), -1))))
        max_out = self.RI_fc4(self.Prelu(self.RI_fc3(torch.stack((self.max_pool(ri2[:, :, :, 0]), self.max_pool(ri2[:, :, :, 1])), -1))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        ri2 = out * ri2

        riOut = self.RIOut3(ri2)
        riOut = self.Prelu(riOut)
        riOut = self.RIOut4(riOut)

        riOut = torch.complex(riOut[...,0], riOut[..., 1])
        riOut = torch.fft.ifft(riOut, dim=1)
        riOut = riOut.real
        return riOut
      
class time_net(nn.Module):
    def __init__(self, input_length):
        super(time_net, self).__init__()
        self.input_length = input_length # 100
        self.input_size1 = 256
        self.input_size2 = 128
        self.features = 32

        self.conv00 = nn.Conv1d(1, self.features, 1)
        self.bn = nn.BatchNorm1d(self.input_length)
        # Expand block
        self.conv01 = nn.Conv1d(self.input_length, self.input_size1, kernel_size=3, padding=1)
        self.conv02 = nn.Conv1d(self.input_length, self.input_size1, kernel_size=5, padding=2)
        self.conv03 = nn.Conv1d(self.input_length, self.input_size1, kernel_size=7, padding=3)
        self.conv04 = nn.Conv1d(self.input_length, self.input_size1, kernel_size=1)

        self.conv11 = nn.Conv1d(self.input_size1, self.input_size2, kernel_size=3, padding=1)
        self.conv14 = nn.Conv1d(self.input_size1, self.input_size2, kernel_size=1)

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(self.input_size2, self.input_size2 // 4, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(self.input_size2 // 4, self.input_size2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.Prelu = nn.PReLU()

        self.conv15 = nn.Linear(self.features*8, self.features*4, bias=True)
        self.conv16 = nn.Linear(self.features*4, 1, bias=True)

        # Expand block
        self.conv000 = nn.Conv1d(1, self.features, 1)
        self.bn2 = nn.BatchNorm1d(self.input_size2)
        self.conv21 = nn.Conv1d(self.input_size2, self.input_size1, kernel_size=3, padding=1)
        self.conv22 = nn.Conv1d(self.input_size2, self.input_size1, kernel_size=5, padding=2)
        self.conv23 = nn.Conv1d(self.input_size2, self.input_size1, kernel_size=7, padding=3)
        self.conv24 = nn.Conv1d(self.input_size2, self.input_size1, kernel_size=1)

        self.conv31 = nn.Conv1d(self.input_size1, self.input_length, kernel_size=3, padding=1)
        self.conv34 = nn.Conv1d(self.input_size1, self.input_length, kernel_size=1)

        self.fc3 = nn.Conv1d(self.input_length, self.input_length // 4, kernel_size=1, bias=False)
        self.fc4 = nn.Conv1d(self.input_length // 4, self.input_length, kernel_size=1, bias=False)

        self.conv35 = nn.Linear(self.features*8, self.features*4, bias=True)
        self.conv36 = nn.Linear(self.features*4, 1, bias=True)

    def forward(self, x):

        x1 = self.conv00(x.transpose(2, 1)).transpose(2, 1)
        x1 = self.bn(x1)

        inputx = self.conv01(x1)# top layer
        temp = self.conv02(x1)
        inputx = torch.cat((inputx, temp), 2)
        temp = self.conv03(x1)
        inputx = torch.cat((inputx, temp), 2)
        temp = self.conv04(x1)
        inputx = torch.cat((inputx, temp), 2)

        inputx2 = self.conv11(inputx)# top layer
        temp = self.conv14(inputx)
        inputx2 = torch.cat((inputx2, temp), 2)

        avg_out = self.fc2(self.Prelu(self.fc1(self.avg_pool(inputx2))))
        max_out = self.fc2(self.Prelu(self.fc1(self.max_pool(inputx2))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        inputx2 = out * inputx2
    
        inputx3 = self.conv15(inputx2)
        inputx3 = self.Prelu(inputx3)
        inputx3 = self.conv16(inputx3)

        out1 = self.conv000(inputx3.transpose(2, 1)).transpose(2, 1)
        out1 = self.bn2(out1)

        inputx4 = self.conv21(out1)# top layer
        temp = self.conv22(out1)
        inputx4 = torch.cat((inputx4, temp), 2)
        temp = self.conv23(out1)
        inputx4 = torch.cat((inputx4, temp), 2)
        temp = self.conv24(out1)
        inputx4 = torch.cat((inputx4, temp), 2)

        inputx5 = self.conv31(inputx4)# top layer
        temp = self.conv34(inputx4)
        inputx5 = torch.cat((inputx5, temp), 2)

        avg_out = self.fc4(self.Prelu(self.fc3(self.avg_pool(inputx5))))
        max_out = self.fc4(self.Prelu(self.fc3(self.max_pool(inputx5))))

        out = avg_out + max_out
        out = self.sigmoid(out)
        inputx5 = out * inputx5

        inputx6 = self.conv35(inputx5)
        inputx6 = self.Prelu(inputx6)
        inputx6 = self.conv36(inputx6)

        return inputx6

class net(nn.Module):
    def __init__(self, input_length):
        super(net, self).__init__()
        self.input_length = input_length
        self.fre_net = fre_net(input_length)
        self.time_net = time_net(input_length)

    def forward(self, signal):
        fre_result = self.fre_net(signal)
        time_result = self.time_net(fre_result)
        return fre_result, time_result

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

if __name__ == "__main__":
    model = net(200)
    for name, parameters in model.state_dict().items():
        if "weight" in name:
            print(name, '.', parameters.detach().shape)
