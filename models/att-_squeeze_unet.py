from numpy import float32
import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, input_channel, squeeze, expand):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=squeeze, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=squeeze),
        )
        self.left = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=3, padding= 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        # print("left", left.shape)
        right = self.right(x)
        # print("right", right.shape)
        x = torch.concat([left, right], axis= 1)
        return x

class AttFireModule(nn.Module):
    def __init__(self, input_channel, squeeze, expand):
        super(AttFireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=squeeze, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=squeeze),
        )
        self.left = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        # print("left", left.shape)
        right = self.right(x)
        # print("right", right.shape)
        x = torch.concat([left, right], axis= 1)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels,filters):
        super(AttentionBlock, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=filters)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=filters)
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)

        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        out = x*psi

        return out
    
class UpsamplingBlock(nn.Module):
    def __init__(self,input_channels, filters, squeeze, expand, strides, deconv_ksize, att_filters):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, filters, kernel_size=deconv_ksize, stride=strides)
        self.fire = FireModule(filters, squeeze, expand)
        self.attention = AttentionBlock(filters, att_filters)
    
    def forward(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        d = torch.concat([x, d], axis=1)
        x = self.fire(d)

        return x

class AttSqueezeUNet(nn.Module):
    def __init__(self, in_channels, n_class=1, dropout=False):
        super(AttSqueezeUNet, self).__init__()
        self.__dropout = dropout
        self.channel_axis = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2)
        )

        self.fire1 = FireModule(64, 16, 64)
        self.fire2 = FireModule(64, 16, 64)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        self.fire3 = FireModule(64,32,128)
        self.fire4 = FireModule(128,32,128)
        self.maxpooling_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        self.fire5 = FireModule(128, 48, 192)
        self.fire6 = FireModule(192, 48, 192)
        self.fire7 = FireModule(192, 48, 192)
        self.fire8 = FireModule(192, 48, 192)

        self.upsampling1 = UpsamplingBlock(192, 192, squeeze=48, expand=192, strides=(1,1), deconv_ksize=(3), att_filters=96)
        self.upsampling2
        self.upsampling3
        self.upsampling4
        self.upsampling5



if __name__ == "__main__":
    model = AttFireModule(input_channel=3, squeeze=8, expand=16)
    print(model)
    input_ = torch.randn([1,3,224,224], dtype=torch.float32)
    print(input_.shape)
    output = model(input_)
    print(output.shape)
