# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from functools import partial

DATA_BACKEND_CHOICES = ["pytorch", "synthetic"]
#try:
#    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
#    from nvidia.dali.pipeline import Pipeline
#    import nvidia.dali.ops as ops
#    import nvidia.dali.types as types
#
#    DATA_BACKEND_CHOICES.append("dali-gpu")
#    DATA_BACKEND_CHOICES.append("dali-cpu")
#except ImportError:
#    print(
#        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
#    )

try:
    from cambricon.dali.plugin.pytorch import DALIClassificationIterator
    from cambricon.dali.pipeline import Pipeline
    import cambricon.dali.ops as ops
    import cambricon.dali.types as types

    DATA_BACKEND_CHOICES.append("dali-mlu")
    DATA_BACKEND_CHOICES.append("dali-cpu")
except ImportError:
    print(
        "Please install cambricon DALI to run this example."
        )

def load_jpeg_from_file(path, cuda=True):
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


class HybridTrainPipe(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        interpolation,
        crop,
        dali_cpu=False,
    ):
        super(HybridTrainPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=True,
            pad_last_batch=True,
        )

        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.ImageDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.ImageDecoder(
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=211025920,
                host_memory_padding=140544512,
            )

        self.res = ops.RandomResizedCrop(
            device=dali_device,
            size=[crop, crop],
            interp_type=interpolation,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
        )

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, data_dir, interpolation, crop, size
    ):
        super(HybridValPipe, self).__init__(
            batch_size, num_threads, device_id, seed=12 + device_id
        )
        interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=False,
            pad_last_batch=True,
        )

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(
            device="gpu",
            resize_shorter=size,
            interp_type=interpolation,
        )
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class DALIWrapper(object):
    def gen_wrapper(dalipipeline, num_classes, one_hot, memory_format, device):
        for data in dalipipeline:
            input = data[0]["data"].contiguous(memory_format=memory_format)
            if device == 'gpu':
                target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            elif device == 'mlu':
                target = torch.reshape(data[0]["label"], [-1]).mlu().long()
            if one_hot:
                target = expand(num_classes, torch.float, target)
            yield input, target
        dalipipeline.reset()

    def __init__(self, dalipipeline, num_classes, one_hot, memory_format, device):
        self.dalipipeline = dalipipeline
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format
        self.device = device

    def __iter__(self):
        return DALIWrapper.gen_wrapper(
            self.dalipipeline, self.num_classes, self.one_hot, self.memory_format, self.device
        )


def get_dali_train_loader(dali_cpu=False):
    def gdtl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        augmentation=None,
        start_epoch=0,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        device='mlu',
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        traindir = os.path.join(data_path, "train")
        if augmentation is not None:
            raise NotImplementedError(
                f"Augmentation {augmentation} for dali loader is not supported"
            )

        if device == 'gpu':
            device_id = rank % torch.cuda.device_count()
        elif device == 'mlu':
            device_id = rank % torch.mlu.device_count()

        pipe = HybridTrainPipe(
            batch_size=batch_size,
            num_threads=workers,
            device_id=device_id,
            data_dir=traindir,
            interpolation=interpolation,
            crop=image_size,
            dali_cpu=dali_cpu,
        )

        pipe.build()
        train_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", fill_last_batch=False
        )

        return (
            DALIWrapper(train_loader, num_classes, one_hot, memory_format, device),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdtl


def get_dali_val_loader():
    def gdvl(
        data_path,
        image_size,
        batch_size,
        num_classes,
        one_hot,
        interpolation="bilinear",
        crop_padding=32,
        workers=5,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format,
        **kwargs,
    ):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        valdir = os.path.join(data_path, "val")

        pipe = HybridValPipe(
            batch_size=batch_size,
            num_threads=workers,
            device_id=rank % torch.cuda.device_count(),
            data_dir=valdir,
            interpolation=interpolation,
            crop=image_size,
            size=image_size + crop_padding,
        )

        pipe.build()
        val_loader = DALIClassificationIterator(
            pipe, reader_name="Reader", fill_last_batch=False
        )

        return (
            DALIWrapper(val_loader, num_classes, one_hot, memory_format),
            int(pipe.epoch_size("Reader") / (world_size * batch_size)),
        )

    return gdvl
