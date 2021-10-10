from UtilityManagement.pytorch_util import *
import UtilityManagement.config as cf
import math
import os
import warnings

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(DecoderBlock, self).__init__()
        # Block 3 : Deconv -> BN -> LeakyRelu -> Dropout
        self.deconv = set_deconv(in_channel, out_channel, kernel=3, strides=2, bias=False)
        self.bn = set_batch_normalization(out_channel)

        # Block 4 : Skip Connection -> Dropoout

        # Block 5 : Conv -> BN -> LeakyRelu ->
        #           Conv -> BN -> LeakyRelu ->
        #           Conv -> BN -> LeakyRelu -> Dropout

        self.conv1 = set_conv(out_channel, out_channel, kernel=3, strides=1, padding=1, bias=False)
        self.bn1 = set_batch_normalization(out_channel)

        self.conv2 = set_conv(out_channel, out_channel, kernel=3, strides=1, padding=1, bias=False)
        self.bn2 = set_batch_normalization(out_channel)

        self.conv3 = set_conv(out_channel, out_channel, kernel=3, strides=1, padding=1, bias=False)
        self.bn3 = set_batch_normalization(out_channel)

        self.leakyrelu = set_leakyrelu(True)
        self.dropout = set_dropout(dropout)

    def forward(self, x, skip_layer):
        out = self.deconv(x)
        out = self.leakyrelu(out)
        out = self.bn(out)
        out = self.dropout(out)

        out = out + skip_layer
        out = self.dropout(out)

        out = self.conv1(out)
        out = self.leakyrelu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.bn2(out)


        out = self.conv3(out)
        out = self.leakyrelu(out)
        out = self.bn3(out)


        out = self.dropout(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, pooling=True):
        super(EncoderBlock, self).__init__()
        self.is_pooling = pooling
        # Shortcut
        self.conv0 = set_conv(in_channel, out_channel, kernel=1, strides=1, padding=0, bias=False)
        self.bn0 = set_batch_normalization(out_channel)

        self.conv1 = set_conv(in_channel, out_channel, kernel=3, strides=1, padding=1, bias=False)
        self.bn1 = set_batch_normalization(out_channel)

        self.conv2 = set_conv(out_channel, out_channel, kernel=3, strides=1, padding=1, bias=False)
        self.bn2 = set_batch_normalization(out_channel)

        self.leakyrelu = set_leakyrelu(True)

        self.dropout = set_dropout(dropout)
        self.pooling = set_max_pool(kernel=2, strides=2)

    def forward(self, x):
        shortcut = self.conv0(x)
        shortcut = self.leakyrelu(shortcut)
        shortcut = self.bn0(shortcut)


        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.bn1(out)


        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.bn2(out)


        out = out + shortcut
        before_pooling = self.dropout(out)
        if self.is_pooling == True:
            after_pooling = self.pooling(before_pooling)
        else:
            after_pooling = None

        return before_pooling, after_pooling


class SalsaNet(nn.Module):

    def __init__(self, classes):
        super(SalsaNet, self).__init__()

        self.model_name = 'SalsaNet'

        # SalsaNet 기본 구성
        # Encoder
        # Block1 -> Block2 -> Block3 -> Block4 -> Block5
        # Block : Residual Block ( Shortcut -> 3x3 Conv -> 3x3 Conv -> Add )
        # Decoder
        # Block1 -> Block2 -> Block3 -> Block4 -> Block5
        # Block : Residual Block ( Shortcut -> Deconv -> Add -> 3x3 Conv -> 3x3 Conv -> 3x3 Conv
        # Channel : 32, 64, 128, 256, 256
        # Input : 256x64x4
        # Conv 뒤 Batch Normalization 뒤 LeakyRelu
        # Block 마지막에 Dropout -> Pooling

        channels = (32, 64, 128, 256, 256)

        in_channel = 4
        self.inplanes = 64

        # Encoder
        self.layer1 = EncoderBlock(in_channel, channels[0], 0.5, True)
        self.layer2 = EncoderBlock(channels[0], channels[1], 0.5, True)
        self.layer3 = EncoderBlock(channels[1], channels[2], 0.5, True)
        self.layer4 = EncoderBlock(channels[2], channels[3], 0.5, True)
        self.layer5 = EncoderBlock(channels[3], channels[4], 0.5, False)

        # Decoder
        self.layer6 = DecoderBlock(channels[4], channels[3], 0.5)
        self.layer7 = DecoderBlock(channels[3], channels[2], 0.5)
        self.layer8 = DecoderBlock(channels[2], channels[1], 0.5)
        self.layer9 = DecoderBlock(channels[1], channels[0], 0.5)

        self.conv = set_conv(channels[0], classes, kernel=1, strides=1, padding=0, bias=False)

    def forward(self, x):
        encoder1_before, encoder1_after = self.layer1(x)
        encoder2_before, encoder2_after = self.layer2(encoder1_after)
        encoder3_before, encoder3_after = self.layer3(encoder2_after)
        encoder4_before, encoder4_after = self.layer4(encoder3_after)
        encoder5_before, encoder5_after = self.layer5(encoder4_after)

        decoder4 = self.layer6(encoder5_before, encoder4_before)
        decoder3 = self.layer7(decoder4, encoder3_before)
        decoder2 = self.layer8(decoder3, encoder2_before)
        decoder1 = self.layer9(decoder2, encoder1_before)

        out = self.conv(decoder1)
        return out

    def get_name(self):
        return self.model_name

    def initialize_weights(self, init_weights):
        if init_weights is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


def salsanet(classes):
    pretrained_path = cf.paths['pretrained_path']
    model = SalsaNet(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model!')
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model


# devices = torch.device("cuda") if is_gpu_avaliable() else torch.device("cpu")
# model = salsanet(3)
# get_summary(model, devices)