import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmocr.models.textrecog.layers import BasicBlock
from mmocr.registry import MODELS


class conv_adapter(nn.Module):
    """
    This is the implementation of the adapter module, namely conv1x1.
    """

    def __init__(self, in_planes, out_planes, stride=1):
        super(conv_adapter, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_task(nn.Module):
    def __init__(
        self, in_planes, out_planes, num_tasks, stride=1, padding=1, kernel_size=3
    ):
        """
        The convolutional layer with task adapters. Relu is not included here.
        Set stride=2 for downsampling, set stride=1 for no downsampling, set stride=(2, 1) for downsampling in height only.
        """
        super(conv_task, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=0,
        )
        self.num_tasks = num_tasks
        self.bn = nn.BatchNorm2d(out_planes)
        self.parallel_adapters = nn.ModuleList(
            [
                conv_adapter(in_planes, out_planes, stride=stride)
                for _ in range(num_tasks)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_planes) for i in range(num_tasks)])

    def forward(self, x, task_id):
        y = self.conv(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.parallel_adapters[task_id](x)
        y = y + x
        y = self.bns[task_id](y)

        return y


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_tasks, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_task(
            in_planes, out_planes, num_tasks, stride=stride, padding=1
        )
        self.conv2 = conv_task(out_planes, out_planes, num_tasks, stride=1, padding=1)
        self.num_tasks = num_tasks
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, task_id):
        residual = x
        out = self.conv1(x, task_id)
        out = F.relu(out)
        out = self.conv2(out, task_id)
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out, task_id


@MODELS.register_module()
class AdaptersResnet(BaseModule):
    """
    This is the implementation of the ResNet with adapters.
    """

    def __init__(
        self, num_tasks: int, num_blocks: list, num_classes: list, in_planes=32
    ):
        super().__init__()
        self.in_planes = in_planes
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        block = BasicBlock
        blocks = [block for _ in range(len(num_blocks))]
        self.conv1 = conv_task(
            3, self.in_planes, num_tasks, stride=1, padding=1, kernel_size=3
        )
        self.layers = nn.ModuleList()

        for i in range(len(num_blocks)):
            if i == 0:
                self.layers.append(
                    self._make_layer(blocks[i], 64 * (2**i), num_blocks[i], stride=2)
                )
            elif i == 1:
                self.layers.append(
                    self._make_layer(blocks[i], 64 * (2**i), num_blocks[i], stride=2)
                )
            else:
                self.layers.append(
                    self._make_layer(
                        blocks[i], 64 * (2**i), num_blocks[i], stride=(2, 1)
                    )
                )

        self.end_bns = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(32 * 2 ** len(blocks)), nn.ReLU())
                for _ in range(num_tasks)
            ]
        )

        self.end_conv = conv_task(
            32 * 2 ** len(blocks),
            32 * 2 ** len(blocks),
            num_tasks,
            stride=(2, 1),
            padding=(0, 0),
            kernel_size=(2, 1),
        )


    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, self.num_tasks, stride=stride)]
        self.in_planes = planes
        layers.extend(
            block(self.in_planes, planes, self.num_tasks, stride=1)
            for _ in range(1, num_blocks)
        )
        return nn.ModuleList(layers)

    def forward(self, x, task_id):
        x = self.conv1(x, task_id)
        x = F.relu(x)
        for layer in self.layers:
            for basic_block in layer:
                x, task_id = basic_block(x, task_id)
        x = self.end_bns[task_id](x)
        x = self.end_conv(x, task_id)
        return x
