import mimetypes
import random
from pathlib import Path
from typing import Union

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image

from utils import TransferMode


def collect_images(path: Union[str, Path]) -> list[Path]:
    mime_checker = mimetypes.MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and mime_type.startswith("image")

    path = Path(path)
    if path.is_dir():
        return sorted(filter(validate_file_type, path.rglob("*.*")))
    else:
        return [path]


def collect_images_from_images(
    images: list[Union[str, Path]],
    root_path: Union[str, Path],
    target_path: Union[str, Path],
) -> list[Path]:
    target_path = Path(target_path)
    if target_path.is_file():
        return [target_path]

    paths = []
    exts = [
        ext
        for ext, mime_type in mimetypes.types_map.items()
        if mime_type.startswith("image")
    ]
    exts += [ext.upper() for ext in exts]
    for image in images:
        image = Path(image)
        rel_path = image.relative_to(root_path)

        for ext in exts:
            path = target_path / rel_path.with_suffix(ext)
            if path.exists():
                break
        else:
            assert (
                False
            ), f"The corresponding background image is not found, {rel_path}."

        paths.append(path)
    return paths


def default_loader(path: Union[str, Path]):
    return Image.open(path).convert("RGB")


class Dataset(data.Dataset):
    def __init__(
        self,
        content_path: str,
        style_path: str,
        inference_size: int,
        transfer_mode: TransferMode = TransferMode.P2P,
    ):
        self.contentPath = content_path
        self.stylePath = style_path

        match transfer_mode:
            case TransferMode.P2P:
                self.content_paths = collect_images(content_path)
                self.style_paths = collect_images_from_images(
                    self.content_paths, content_path, style_path
                )
            case TransferMode.RANDOM:
                self.content_paths = collect_images(content_path)
                self.style_paths = random.choices(
                    collect_images(style_path), k=len(self.content_paths)
                )
            case _:
                raise NotImplementedError("Unsupported transfer mode.")

        self.inference_size = inference_size
        # self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        # normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        # self.prep = transforms.Compose(
        #     [
        #         transforms.Scale(fineSize),
        #         transforms.ToTensor(),
        #         # transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
        #     ]
        # )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        content_path = self.content_paths[index]
        style_path = self.style_paths[index]

        content_img = default_loader(content_path)
        style_img = default_loader(style_path)

        # resize
        if self.inference_size != 0:
            w, h = content_img.size
            if w > h:
                if w != self.inference_size:
                    neww = self.inference_size
                    newh = int(h * neww / w)
                    content_img = content_img.resize((neww, newh))
                    style_img = style_img.resize((neww, newh))
            else:
                if h != self.inference_size:
                    newh = self.inference_size
                    neww = int(w * newh / h)
                    content_img = content_img.resize((neww, newh))
                    style_img = style_img.resize((neww, newh))

        # Preprocess Images
        content_img = F.to_tensor(content_img)
        style_img = F.to_tensor(style_img)
        return (
            content_img,
            style_img,
            content_path.relative_to(self.contentPath).as_posix(),
        )

    def __len__(self):
        return len(self.content_paths)
