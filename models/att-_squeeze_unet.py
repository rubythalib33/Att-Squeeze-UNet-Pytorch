from numpy import float32
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
    def __init__(self, in_channels, in_channels2 ,filters):
        super(AttentionBlock, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=filters)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels2, filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=filters)
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        print("--g", g.shape, self.w_g)
        print("--x", x.shape, self.w_x)
        g1 = self.w_g(g)
        print("--g1", g1.shape)
        x1 = self.w_x(x)
        print("--x1", x1.shape)

        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        out = x*psi

        return out
    
class UpsamplingBlock(nn.Module):
    def __init__(self,input_channels, input_channels2, filters, squeeze, expand, strides, deconv_ksize, att_filters):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_channels, filters, kernel_size=deconv_ksize, stride=strides)
        self.fire = FireModule(filters+input_channels2, squeeze, expand)
        self.attention = AttentionBlock(filters, input_channels2, att_filters)
    
    def forward(self, x, g):
        d = self.upconv(x)
        d = TF.resize(d, x.shape[2:])
        print("-d", d.shape)
        x = self.attention(d, g)
        print("-x", x.shape)
        d = torch.concat([x, d], axis=1)
        print("-d", d.shape)
        x = self.fire(d)
        print("-x", x.shape)

        return x

class AttSqueezeUNet(nn.Module):
    def __init__(self, in_channels, n_class=1, dropout=False):
        super(AttSqueezeUNet, self).__init__()
        self.__dropout = dropout
        self.channel_axis = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.maxpooling_1 = nn.MaxPool2d(kernel_size=(3,3), stride=2)

        self.dropout = nn.Dropout(0.2) if self.__dropout else None

        self.fire1 = FireModule(64, 16, 64)
        self.fire2 = FireModule(128, 16, 64)
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        self.fire3 = FireModule(128,32,128)
        self.fire4 = FireModule(256,32,128)
        self.maxpooling_3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))

        self.fire5 = FireModule(256, 48, 192)
        self.fire6 = FireModule(384, 48, 192)
        self.fire7 = FireModule(384, 48, 256)
        self.fire8 = FireModule(512, 48, 256)

        self.upsampling1 = UpsamplingBlock(512,384, 192, squeeze=48, expand=192, strides=(1,1), deconv_ksize=(3), att_filters=96)
        self.upsampling2 = UpsamplingBlock(384, 256, 128, squeeze=32, expand=128, strides=(1,1), deconv_ksize=(3), att_filters=64)
        self.upsampling3 = UpsamplingBlock(256, 128, 64, squeeze=16, expand=64, strides=(1,1), deconv_ksize=(3), att_filters=16)
        self.upsampling4 = UpsamplingBlock(64, 64, 32, squeeze=16, expand=32, strides=(1,1), deconv_ksize=(3), att_filters=4)
        self.upsampling5 = nn.Upsample(size=(2,2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upsampling6 = nn.Upsample(size = (2,2))

        self.conv3  = nn.Sequential(
            nn.Conv2d(64, n_class, 1),
            nn.LogSoftmax() if n_class > 1 else nn.Sigmoid()
        )
    
    def forward(self, x):
        x0 = self.conv1(x)
        print("x0",x0.shape)
        x1 = self.maxpooling_1(x0)
        print("x1",x1.shape)

        x2 = self.fire1(x1)
        x2 = self.fire2(x2)
        x2 = self.maxpooling_2(x2)
        print("x2",x2.shape)

        x3 = self.fire3(x2)
        x3 = self.fire4(x3)
        x3 = self.maxpooling_3(x3)
        print("x3", x3.shape)

        x4 = self.fire5(x3)
        x4 = self.fire6(x4)
        print("x4",x4.shape)

        x5 = self.fire7(x4)
        x5 = self.fire8(x5)
        print("x5",x5.shape)

        if self.__dropout:
            x5 = self.dropout(x5)
        
        d5 = self.upsampling1(x5,x4)
        print("d5", d5.shape)
        d4 = self.upsampling2(d5, x3)
        print("d4", d4.shape)
        d3 = self.upsampling3(d4, x2)
        print("d3", d3.shape)
        d2 = self.upsampling4(d3, x1)
        print("d2", d2.shape)
        d1 = self.upsampling5(d2)
        print("d1", d1.shape)

        d0 = torch.concat([d1,x0], dim=1)
        d0 = self.conv2(d0)
        d0 = self.upsampling6(d0)

        d = self.conv3(d0)

        return d





if __name__ == "__main__":
    model = AttSqueezeUNet(in_channels=3, n_class=1)
    print(model)
    input_ = torch.randn([1,3,224,224], dtype=torch.float32)
    print(input_.shape)
    output = model(input_)
    print(output.shape)
