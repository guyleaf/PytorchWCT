import random
from enum import Enum

import torch
import torch.nn as nn
import torchfile

from modelsNIPS import (
    Decoder1,
    Decoder2,
    Decoder3,
    Decoder4,
    Decoder5,
    Encoder1,
    Encoder2,
    Encoder3,
    Encoder4,
    Encoder5,
)


class TransferMode(str, Enum):
    P2P = "p2p"
    RANDOM = "random"

    def __str__(self):
        return self.value


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WCT(nn.Module):
    def __init__(self, args):
        super().__init__()
        # load pre-trained network
        vgg1 = torchfile.load(args.vgg1)
        decoder1_torch = torchfile.load(args.decoder1)
        vgg2 = torchfile.load(args.vgg2)
        decoder2_torch = torchfile.load(args.decoder2)
        vgg3 = torchfile.load(args.vgg3)
        decoder3_torch = torchfile.load(args.decoder3)
        vgg4 = torchfile.load(args.vgg4)
        decoder4_torch = torchfile.load(args.decoder4)
        vgg5 = torchfile.load(args.vgg5)
        decoder5_torch = torchfile.load(args.decoder5)

        self.e1 = Encoder1(vgg1)
        self.d1 = Decoder1(decoder1_torch)
        self.e2 = Encoder2(vgg2)
        self.d2 = Decoder2(decoder2_torch)
        self.e3 = Encoder3(vgg3)
        self.d3 = Decoder3(decoder3_torch)
        self.e4 = Encoder4(vgg4)
        self.d4 = Decoder4(decoder4_torch)
        self.e5 = Encoder5(vgg5)
        self.d5 = Decoder5(decoder5_torch)

    def whiten_and_color(self, cF: torch.Tensor, sF: torch.Tensor):
        # -- content feature whitening
        cF_size = cF.size()
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        content_conv = (
            torch.mm(cF, cF.t()).div(cF_size[1] - 1)
            + torch.eye(cF_size[0], device=cF.device).double()
        )
        c_u, c_e, c_v = torch.svd(content_conv, some=False)

        k_c = cF_size[0]
        for i in range(cF_size[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        # -- style feature whitening
        sF_size = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        style_conv = torch.mm(sF, sF.t()).div(sF_size[1] - 1)
        s_u, s_e, s_v = torch.svd(style_conv, some=False)

        k_s = sF_size[0]
        for i in range(sF_size[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        target_feature = torch.mm(
            torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())),
            whiten_cF,
        )
        target_feature = target_feature + s_mean.unsqueeze(1).expand_as(target_feature)
        return target_feature

    def transform(self, cF: torch.Tensor, sF: torch.Tensor, alpha: float):
        cF = cF.double()
        sF = sF.double()
        C, W, H = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cF_view = cF.view(C, -1)
        sF_view = sF.view(C, -1)

        target_feature = self.whiten_and_color(cF_view, sF_view)
        target_feature = target_feature.view_as(cF)
        csF = alpha * target_feature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        # csF.data.resize_(ccsF.size()).copy_(ccsF)
        return csF
