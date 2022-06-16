import torch.nn as nn
import torch.nn.functional as F
import torch

'''The Refining Network of DMR'''
class Deformable_Skip_Learner(nn.Module):
    def __init__(self, inch):
        super(Deformable_Skip_Learner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                pad = ksz // 2
                building_block_layers.append(nn.Conv3d(inch, outch, ksz, stride, pad))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])
        self.encoder_layer1 = make_building_block(inch[3], [outch1, outch2, outch3], [5, 5, 5], [1, 1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3 + 32 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3 + 32 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer2to1 = make_building_block(outch3 + 16 * 2, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv3d(outch3, outch3, (3, 3, 3), padding=(1, 1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv3d(outch3, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv3d(outch2, outch2, (3, 3, 3), padding=(1, 1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv3d(outch2, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
                                      nn.ReLU())

        self.decoder3 = nn.Sequential(nn.Conv3d(outch1, outch1, (3, 3, 3), padding=(1, 1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv3d(outch1, 3, (3, 3, 3), padding=(1, 1, 1), bias=True))

    def interpolate_dims(self, hypercorr, spatial_size=None):
        bsz, ch, d, w, h = hypercorr.size()
        hypercorr = F.interpolate(hypercorr, (2 * d, 2 * w, 2 * h), mode='trilinear', align_corners=True)
        return hypercorr

    def forward(self, hypercorr_pyramid, moving_feat, fixed_feat):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        hypercorr_sqz1 = self.encoder_layer1(hypercorr_pyramid[3])

        # Propagate encoded 3D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-6:-3])
        hypercorr_mix43 = 2 * hypercorr_sqz4 + hypercorr_sqz3  #add
        hypercorr_mix43 = torch.cat([hypercorr_mix43, moving_feat[-2], fixed_feat[-2]], dim=1) #skip connection
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_dims(hypercorr_mix43, hypercorr_sqz2.size()[-6:-3])
        hypercorr_mix432 = 2 * hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = torch.cat([hypercorr_mix432, moving_feat[-3], fixed_feat[-3]], dim=1)
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        hypercorr_mix432 = self.interpolate_dims(hypercorr_mix432, hypercorr_sqz1.size()[-6:-3])
        hypercorr_mix4321 = 2 * hypercorr_mix432 + hypercorr_sqz1
        hypercorr_mix4321 = torch.cat([hypercorr_mix4321, moving_feat[-4], fixed_feat[-4]], dim=1)
        hypercorr_mix4321 = self.encoder_layer2to1(hypercorr_mix4321)

        # Decode the encoded 3D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_mix4321)
        upsample_size = (hypercorr_decoded.size(-3) * 2, hypercorr_decoded.size(-2) * 2, hypercorr_decoded.size(-1) * 2)
        hypercorr_decoded = 2 * F.interpolate(hypercorr_decoded, upsample_size, mode='trilinear', align_corners=True)
        hypercorr_decoded = self.decoder2(hypercorr_decoded)
        logit_mask = self.decoder3(hypercorr_decoded)

        return logit_mask


if __name__ == "__main__":
    import torch

    corr = []
    corr.append(torch.rand(2, 3, 14, 12, 10))
    corr.append(torch.rand(2, 3, 28, 24, 20))
    corr.append(torch.rand(2, 3, 56, 48, 40))
    corr.append(torch.rand(2, 3, 112, 96, 80))
    hpn_learner = Deformable_Skip_Learner([3, 3, 3, 3])
    moving_feat = []
    moving_feat.append(torch.rand(2, 16, 112, 96, 80))
    moving_feat.append(torch.rand(2, 32, 56, 48, 40))
    moving_feat.append(torch.rand(2, 32, 28, 24, 20))
    moving_feat.append(torch.rand(2, 64, 14, 12, 10))

    fixed_feat = []
    fixed_feat.append(torch.rand(2, 16, 112, 96, 80))
    fixed_feat.append(torch.rand(2, 32, 56, 48, 40))
    fixed_feat.append(torch.rand(2, 32, 28, 24, 20))
    fixed_feat.append(torch.rand(2, 64, 14, 12, 10))

    y = hpn_learner(corr, moving_feat, fixed_feat)
    print(y.shape)
