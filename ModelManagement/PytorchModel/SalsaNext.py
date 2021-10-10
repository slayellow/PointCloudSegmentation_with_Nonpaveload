from UtilityManagement.pytorch_util import *
import UtilityManagement.config as cf
import math
import os
import warnings

class ContextualModule(nn.Module):
    # Contextual Module
    # 1x1 Conv ( Dilation : 1 ) -> LeakyReLU ( Shortcut ) -> 3x3 Conv ( Dilation : 1 ) -> LeakyReLU -> BN ->
    # 3x3 Conv ( Dilation : 2 ) -> LeakyReLU -> BN -> Skip Connection ( + Shortcut )

    def __init__(self, in_channel, out_channel):
        super(ContextualModule, self).__init__()

        self.conv1 = set_conv(in_channel, out_channel, kernel=1, strides=1, dilation=1, padding=0, bias=False)

        self.conv2 = set_conv(out_channel, out_channel, kernel=3, strides=1, dilation=1, padding=1, bias=False)
        self.bn2 = set_batch_normalization(out_channel)

        self.conv3 = set_conv(out_channel, out_channel, kernel=3, strides=1, dilation=2, padding=2, bias=False)
        self.bn3 = set_batch_normalization(out_channel)

        self.leakyrelu = set_leakyrelu(True)
    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.leakyrelu(shortcut)

        out = self.conv2(shortcut)
        out = self.leakyrelu(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.leakyrelu(out)
        out = self.bn3(out)

        out = shortcut + out
        return out



class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate, dropout=True):
        super(DecoderBlock, self).__init__()

        self.bDropout = dropout

        self.pixelshuffle = set_pixel_shuffle(scale=2)
        self.dropout = set_dropout(rate=dropout_rate)
        self.leakyrelu = set_leakyrelu(True)

        self.conv1 = set_conv(in_channel // 4 + 2 * out_channel, out_channel, kernel=3, strides=1, dilation=1, padding=1, bias=False)
        self.bn1 = set_batch_normalization(out_channel)

        self.conv2 = set_conv(out_channel, out_channel, kernel=3, strides=1, dilation=2, padding=2, bias=False)
        self.bn2 = set_batch_normalization(out_channel)

        self.conv3 = set_conv(out_channel, out_channel, kernel=2, strides=1, dilation=2,  padding=1, bias=False)
        self.bn3 = set_batch_normalization(out_channel)

        self.conv4 = set_conv(out_channel * 3, out_channel, kernel=1, strides=1, dilation=1, padding=0, bias=False)
        self.bn4 = set_batch_normalization(out_channel)

    def forward(self, x, skip_layer):
        out = self.pixelshuffle(x)
        if self.bDropout:
            out = self.dropout(out)

        out = set_concat((out, skip_layer), axis=1)
        if self.bDropout:
            out = self.dropout(out)

        conv1 = self.conv1(out)
        conv1 = self.leakyrelu(conv1)
        conv1 = self.bn1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.leakyrelu(conv2)
        conv2 = self.bn2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.leakyrelu(conv3)
        conv3 = self.bn3(conv3)

        out = set_concat((conv1, conv2, conv3), axis=1)

        out = self.conv4(out)
        out = self.leakyrelu(out)
        out = self.bn4(out)


        if self.bDropout:
            out = self.dropout(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate, dropout=True, pooling=True):
        super(EncoderBlock, self).__init__()

        self.bPooling = pooling
        self.bDropout = dropout

        self.conv0 = set_conv(in_channel, out_channel, kernel=1, strides=1, dilation=1, padding=0, bias=False)

        self.conv1 = set_conv(in_channel, out_channel, kernel=3, strides=1, dilation=1, padding=1, bias=False)
        self.bn1 = set_batch_normalization(out_channel)

        self.conv2 = set_conv(out_channel, out_channel, kernel=3, strides=1, dilation=2, padding=2, bias=False)
        self.bn2 = set_batch_normalization(out_channel)

        self.conv3 = set_conv(out_channel, out_channel, kernel=2, strides=1, dilation=2, padding=1, bias=False)
        self.bn3 = set_batch_normalization(out_channel)

        self.conv4 = set_conv(out_channel * 3, out_channel, kernel=1, strides=1, dilation=1, padding=0, bias=False)
        self.bn4 = set_batch_normalization(out_channel)

        self.dropout = set_dropout(dropout_rate)
        self.pooling = set_avg_pool(kernel=3, strides=2, padding=1)
        self.leakyrelu = set_leakyrelu(True)


    def forward(self, x):
        shortcut = self.conv0(x)
        shortcut = self.leakyrelu(shortcut)

        conv1 = self.conv1(x)
        conv1 = self.leakyrelu(conv1)
        conv1 = self.bn1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.leakyrelu(conv2)
        conv2 = self.bn2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.leakyrelu(conv3)
        conv3 = self.bn3(conv3)

        concat = set_concat((conv1, conv2, conv3), axis=1)
        concat = self.conv4(concat)
        concat = self.leakyrelu(concat)
        concat = self.bn4(concat)


        before_pooling = shortcut + concat

        if self.bPooling:
            if self.bDropout:
                after_pooling = self.dropout(before_pooling)
            else:
                after_pooling = before_pooling
            after_pooling = self.pooling(after_pooling)

            return before_pooling, after_pooling
        else:
            if self.bDropout:
                after_pooling = self.dropout(before_pooling)
            else:
                after_pooling = before_pooling
            return before_pooling, after_pooling




class SalsaNext(nn.Module):

    def __init__(self, classes):
        super(SalsaNext, self).__init__()

        self.model_name = 'SalsaNext'

        # SalsaNext 기본 구성
        channels = (32, 64, 128, 256, 256)

        in_channel = 5

        # Contextual Module
        self.context1 = ContextualModule(in_channel, channels[0])
        self.context2 = ContextualModule(channels[0], channels[0])
        self.context3 = ContextualModule(channels[0], channels[0])

        # Encoder
        self.layer1 = EncoderBlock(channels[0], channels[1], 0.2, dropout=False, pooling=True)
        self.layer2 = EncoderBlock(channels[1], channels[2], 0.2, dropout=True, pooling=True)
        self.layer3 = EncoderBlock(channels[2], channels[3], 0.2, dropout=True, pooling=True)
        self.layer4 = EncoderBlock(channels[3], channels[4], 0.2, dropout=True, pooling=True)
        self.layer5 = EncoderBlock(channels[4], channels[4], 0.2, dropout=True, pooling=False)

        # Decoder
        self.layer6 = DecoderBlock(channels[3], channels[2], 0.2, dropout=True)
        self.layer7 = DecoderBlock(channels[2], channels[2], 0.2, dropout=True)
        self.layer8 = DecoderBlock(channels[2], channels[1], 0.2, dropout=True)
        self.layer9 = DecoderBlock(channels[1], channels[0], 0.2, dropout=False)

        self.conv = set_conv(channels[0], classes, kernel=1, strides=1, padding=0, bias=False)

    def forward(self, x):
        context = self.context1(x)
        context = self.context2(context)
        context = self.context3(context)

        encoder1_before, encoder1_after = self.layer1(context)
        encoder2_before, encoder2_after = self.layer2(encoder1_after)
        encoder3_before, encoder3_after = self.layer3(encoder2_after)
        encoder4_before, encoder4_after = self.layer4(encoder3_after)
        encoder5_before, encoder5_after = self.layer5(encoder4_after)

        decoder4 = self.layer6(encoder5_before, encoder4_before)
        decoder3 = self.layer7(decoder4, encoder3_before)
        decoder2 = self.layer8(decoder3, encoder2_before)
        decoder1 = self.layer9(decoder2, encoder1_before)

        out = self.conv(decoder1)       # Logit

        out = set_softmax(out, axis=1)  # Softmax Activation
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


def salsanext(classes):
    pretrained_path = cf.paths['pretrained_path']
    model = SalsaNext(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model! --> {}'.format(os.path.join(pretrained_path, model.get_name()+'.pth')))
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        print("No Pretrained Model! --> {}".format(model.get_name()))
        model.initialize_weights(init_weights=True)

    return model


# devices = torch.device("cuda") if is_gpu_avaliable() else torch.device("cpu")
# model = salsanext(20)
# get_summary(model, devices)