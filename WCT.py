import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from Loader import Dataset
from utils import WCT, TransferMode, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="WCT Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--content-path", default="images/content", help="path of content folder"
    )
    parser.add_argument(
        "--style-path", default="images/style", help="path of style folder"
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--vgg1",
        default="models/vgg_normalised_conv1_1.t7",
        help="Path to the VGG conv1_1",
    )
    parser.add_argument(
        "--vgg2",
        default="models/vgg_normalised_conv2_1.t7",
        help="Path to the VGG conv2_1",
    )
    parser.add_argument(
        "--vgg3",
        default="models/vgg_normalised_conv3_1.t7",
        help="Path to the VGG conv3_1",
    )
    parser.add_argument(
        "--vgg4",
        default="models/vgg_normalised_conv4_1.t7",
        help="Path to the VGG conv4_1",
    )
    parser.add_argument(
        "--vgg5",
        default="models/vgg_normalised_conv5_1.t7",
        help="Path to the VGG conv5_1",
    )
    parser.add_argument(
        "--decoder5",
        default="models/feature_invertor_conv5_1.t7",
        help="Path to the decoder5",
    )
    parser.add_argument(
        "--decoder4",
        default="models/feature_invertor_conv4_1.t7",
        help="Path to the decoder4",
    )
    parser.add_argument(
        "--decoder3",
        default="models/feature_invertor_conv3_1.t7",
        help="Path to the decoder3",
    )
    parser.add_argument(
        "--decoder2",
        default="models/feature_invertor_conv2_1.t7",
        help="Path to the decoder2",
    )
    parser.add_argument(
        "--decoder1",
        default="models/feature_invertor_conv1_1.t7",
        help="Path to the decoder1",
    )
    # parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--inference-size",
        type=int,
        default=512,
        help="resize image to inference_size x inference_size, leave it to 0 if not resize",
    )
    parser.add_argument("--out-dir", default="outputs/", help="folder to output images")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="hyperparameter to blend wct feature and content feature",
    )
    parser.add_argument(
        "--single-level",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Apply single-level stylization instead of multi-level",
    )
    parser.add_argument(
        "--transfer-mode",
        type=TransferMode,
        choices=list(TransferMode),
        default=TransferMode.P2P,
        help="mode for style transfer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="which gpu to run on.",
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="seed for random transfer mode."
    )

    args = parser.parse_args()
    return args


@torch.inference_mode()
def transfer_style(
    wct: WCT,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    alpha: float,
    single_level: bool = False,
) -> torch.Tensor:
    # --WCT on conv5_1
    # --------------------------------------------------------------------------
    # --Note that since conv5 feature is hard to invert,
    # --if you want to better preserve the content, you can start from WCT on
    # --conv4_1 first, i.e., on Line 283,
    # --local cF4 = vgg4:forward(content):clone()
    # --------------------------------------------------------------------------
    sF5 = wct.e5(style_img)
    cF5 = wct.e5(content_img)
    sF5 = sF5.squeeze(0)
    cF5 = cF5.squeeze(0)
    csF5 = wct.transform(cF5, sF5, alpha)
    im5 = wct.d5(csF5)

    if single_level:
        return im5

    # --WCT on conv4_1
    sF4 = wct.e4(style_img)
    cF4 = wct.e4(im5)
    sF4 = sF4.squeeze(0)
    cF4 = cF4.squeeze(0)
    csF4 = wct.transform(cF4, sF4, alpha)
    im4 = wct.d4(csF4)

    # --WCT on conv3_1
    sF3 = wct.e3(style_img)
    cF3 = wct.e3(im4)
    sF3 = sF3.squeeze(0)
    cF3 = cF3.squeeze(0)
    csF3 = wct.transform(cF3, sF3, alpha)
    im3 = wct.d3(csF3)

    # --WCT on conv2_1
    sF2 = wct.e2(style_img)
    cF2 = wct.e2(im3)
    sF2 = sF2.squeeze(0)
    cF2 = cF2.squeeze(0)
    csF2 = wct.transform(cF2, sF2, alpha)
    im2 = wct.d2(csF2)

    # --WCT on conv1_1
    sF1 = wct.e1(style_img)
    cF1 = wct.e1(im2)
    sF1 = sF1.squeeze(0)
    cF1 = cF1.squeeze(0)
    csF1 = wct.transform(cF1, sF1, alpha)
    im1 = wct.d1(csF1)
    return im1


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # Data loading code
    dataset = Dataset(
        args.content_path,
        args.style_path,
        args.inference_size,
        transfer_mode=args.transfer_mode,
    )
    loader = DataLoader(
        dataset, pin_memory=True, num_workers=args.workers, shuffle=False
    )

    wct = WCT(args)
    wct.to(args.device)

    avgTime = 0
    # cImg = torch.Tensor()
    # sImg = torch.Tensor()
    # csF = torch.Tensor()
    # csF = torch.tensor(csF)
    # if args.cuda:
    #     cImg = cImg.to(args.device)
    #     sImg = sImg.cuda(args.gpu)
    #     csF = csF.cuda(args.gpu)
    #     wct.cuda(args.gpu)
    for i, (content_img, style_img, imname) in enumerate(loader):
        imname = Path(imname[0])
        print("Transferring", imname)
        content_img = content_img.to(args.device)
        style_img = style_img.to(args.device)

        start_time = time.time()

        # WCT Style Transfer
        img = transfer_style(
            wct, content_img, style_img, args.alpha, single_level=args.single_level
        )

        end_time = time.time()
        print("Elapsed time is: %f" % (end_time - start_time))

        # save_image has this wired design to pad images with 4 pixels at default.
        out_file: Path = args.out_dir / imname.with_suffix(".png")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        save_image(img, out_file)

        avgTime += end_time - start_time

    print("Processed %d images. Averaged time is %f" % ((i + 1), avgTime / (i + 1)))
