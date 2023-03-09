import torch
from torch import nn
import torch.nn.functional as F


class Upscale1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor=None, linear_interpolation=True):
        super(Upscale1dLayer, self).__init__()

        if linear_interpolation:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        else:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = kernel_size // 2, padding_mode='reflect') 

    def forward(self, x):
        return self.conv1d(self.upsample_layer(x))


class Upscale1dLayer_multi_input(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor=None, linear_interpolation=True):
        super(Upscale1dLayer_multi_input, self).__init__()

        if linear_interpolation:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
        else:
            self.upsample_layer = torch.nn.Upsample(scale_factor=scale_factor)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = kernel_size // 2, padding_mode='reflect')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.conv1d(self.upsample_layer(x))
   

class Pulse2pulseGenerator(nn.Module):
    def __init__(self, model_size=16, kernel_size=25, num_input_channels=1, num_output_channels=7):
        super(Pulse2pulseGenerator, self).__init__()

        padding_size = kernel_size // 2
        
        self.conv_1 = nn.Conv1d(num_input_channels, model_size, kernel_size, stride=2, padding = padding_size, padding_mode='reflect')
        self.conv_2 = nn.Conv1d(model_size, model_size * 2, kernel_size, stride=2, padding = padding_size, padding_mode='reflect')
        self.conv_3 = nn.Conv1d(model_size * 2, model_size * 4, kernel_size, stride=2, padding = padding_size, padding_mode='reflect') 
        self.conv_4 = nn.Conv1d(model_size * 4, model_size * 8, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        self.conv_5 = nn.Conv1d(model_size * 8, model_size * 16, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        self.conv_6 = nn.Conv1d(model_size * 16, model_size * 32, kernel_size, stride=5, padding = padding_size, padding_mode='reflect')
        
        self.dropout1 = nn.Dropout(0.1)
        self.deconv_1 = Upscale1dLayer(32 * model_size, 16 * model_size, kernel_size, stride=1, scale_factor=5)
        self.dropout2 = nn.Dropout(0.1)
        self.deconv_2 = Upscale1dLayer_multi_input(32 * model_size, 8 * model_size, kernel_size, stride=1, scale_factor=5)
        self.dropout3 = nn.Dropout(0.1)
        self.deconv_3 = Upscale1dLayer_multi_input(16 * model_size, 4 * model_size, kernel_size, stride=1, scale_factor=5)
        self.deconv_4 = Upscale1dLayer_multi_input(8 * model_size, 2 * model_size, kernel_size, stride=1, scale_factor=2)
        self.deconv_5 = Upscale1dLayer_multi_input(4 * model_size, model_size, kernel_size, stride=1, scale_factor=2)
        self.deconv_6 = Upscale1dLayer(model_size, num_output_channels, kernel_size, stride=1, scale_factor=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        conv_1_out = F.leaky_relu(self.conv_1(x))
        conv_2_out = F.leaky_relu(self.conv_2(conv_1_out))
        conv_3_out = F.leaky_relu(self.conv_3(conv_2_out))
        conv_4_out = F.leaky_relu(self.conv_4(conv_3_out))
        conv_5_out = F.leaky_relu(self.conv_5(conv_4_out))
        conv_6_out = F.leaky_relu(self.dropout1(self.conv_6(conv_5_out)))
  
        deconv_1_out = F.relu(self.dropout2(self.deconv_1(conv_6_out)))
        deconv_2_out = F.relu(self.dropout3(self.deconv_2(deconv_1_out, conv_5_out)))
        deconv_3_out = F.relu(self.deconv_3(deconv_2_out, conv_4_out))
        deconv_4_out = F.relu(self.deconv_4(deconv_3_out, conv_3_out))
        deconv_5_out = F.relu(self.deconv_5(deconv_4_out, conv_2_out))
        deconv_6_out = self.deconv_6(deconv_5_out)

        output = torch.tanh(deconv_6_out)
        return output