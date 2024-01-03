import math
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import Sequential
from models.textrecog.layers import BidirectionalGRU

from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.layers import BidirectionalLSTM
from mmocr.registry import MODELS
from mmocr.structures import TextRecogDataSample

from .base import BaseDecoder


def scaled_dot_product(query, key, value, mask=None):
    """
    Inputs:
        query - Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key - Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value - Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        mask - Tensor of shape (batch_size, seq_len)
    """
    # Calculate the dot product of the queries and keys
    energy = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.FloatTensor([key.shape[-1]])
    )

    # Apply the mask
    if mask is not None:
        energy = energy.masked_fill(mask == 0, -1e10)

    # Calculate the attention weights
    attention = torch.softmax(energy, dim=-1)

    # Calculate the dot product of the attention and values
    x = torch.matmul(attention, value)

    return x, attention


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def padding_mask(input):
    """
    The padding mask is used to mask out the padding tokens in the sequence.

    Inputs:
        input - Tensor of shape (batch_size, seq_len)

    Returns:
        mask - Tensor of shape (batch_size, 1, 1, seq_len)
    """
    mask = input.eq(0)
    return mask


def subsequent_mask(input):
    """
    The subsequent mask is used to mask out the subsequent positions in the sequence.

    Inputs:
        input - Tensor of shape (batch_size, seq_len)

    Returns:
        mask - Tensor of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = input.shape
    mask = torch.triu(
        torch.ones((seq_len, seq_len), device=input.device), diagonal=1
    ).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            embed_dim - Dimensionality of the embedding
            num_heads - Number of heads to use in the attention block
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by the number of heads"

        # Linear layers to project the queries, keys and values
        self.qkv_proj = nn.Linear(input_dim, embed_dim * 3)

        # Linear layer to project the output of the attention block
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        self.fc_out.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len, input_dim)
            mask - Tensor of shape (batch_size, seq_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Project the queries, keys and values
        qkv = self.qkv_proj(x)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Split the queries, keys and values into multiple heads
        queries = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        values = values.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # Calculate the dot product of the queries and keys
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / math.sqrt(
            int(keys.shape[-1])
        )

        # Apply the mask
        if mask is not None:
            _mask = torch.ones_like(energy)
            for i in range(batch_size):
                # Add padding mask
                _mask[i, :, : mask[i].sum().int(), : mask[i].sum().int()] = 0
                _mask = _mask.bool()
                # Add look-forward mask
                _one_mask = nn.Transformer.generate_square_subsequent_mask(
                    energy.size(-1)
                )
                _one_mask = _one_mask.unsqueeze(0)
                _head_mask = torch.cat(
                    [_one_mask for _ in range(self.num_heads)], dim=0
                ).to(_mask.device)
                _mask[i] = _head_mask + _mask[i]
            _MASKING_VALUE = -1e30 if energy.dtype == torch.float32 else -1e4
            energy = energy.masked_fill(_mask == 0, _MASKING_VALUE)

        # Calculate the attention weights
        attention = torch.softmax(energy, dim=-1)

        # Calculate the dot product of the attention and values
        x = torch.matmul(attention, values)

        # Concatenate the heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.embed_dim)

        # Project the output of the attention block
        x = self.fc_out(x)

        # Apply dropout
        x = self.dropout(x)

        if return_attention:
            return x, attention
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_seq_len=5000):
        """
        Inputs:
            input_dim - Dimensionality of the input
            max_seq_len - Maximum sequence length to use
        """
        super().__init__()

        # Calculate the positional encodings once in log space
        max_seq_len = int(max_seq_len)
        pe = torch.zeros(max_seq_len, input_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_dim, 2) * -(math.log(10000.0) / input_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register the positional encodings as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attension layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len, input_dim)
            mask - Tensor of shape (batch_size, seq_len)
        """
        # Apply the attention block
        x2 = self.self_attn(x, mask=mask, return_attention=False)

        # Apply residual connection and layer normalization
        x = x + x2
        x = self.norm1(x)

        # Apply the MLP
        x2 = self.mlp(x)

        # Apply residual connection and layer normalization
        x = x + x2
        x = self.norm2(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """
        Inputs:
            num_layers - Number of encoder blocks to use
            block_args - Arguments to pass to the EncoderBlock
        """
        super().__init__()

        # Create the encoder blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len, input_dim)
            mask - Tensor of shape (batch_size, seq_len)
        """
        # Apply each encoder block
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x

    def get_attention(self, x, mask=None):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len, input_dim)
            mask - Tensor of shape (batch_size, seq_len)
        """
        # Apply each encoder block
        attention = []
        for layer in self.layers:
            x, attn = layer.self_attn(x, x, x, mask=mask)
            attention.append(attn)

        return attention


class AdapterBlock(nn.Module):
    def __init__(self, input_dim, h_dim):
        """
        Bottleneck block for the adapter layers

        Inputs:
            input_dim - Dimensionality of the input
            h_dim - Dimensionality of the hidden layer
        """
        super().__init__()

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, input_dim)
        )

    def forward(self, x):
        x = self.bottleneck(x)
        return x


class AdapterEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        dim_feedforward,
        adapter_dim,
        num_classes: list,
        dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            adapter_dim - Dimensionality of the adapter layers
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        self.model_dim = input_dim
        self.num_tasks = len(num_classes)

        self.positional_encoding = PositionalEncoding(input_dim)

        # Attension layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
            nn.Dropout(dropout),
        )

        # Adapter layers
        self.transormer_adapters_list = nn.ModuleList(
            [AdapterBlock(input_dim, adapter_dim) for _ in range(self.num_tasks)]
        )

    def forward(
        self, x, task_id: int, mask=None, add_positional_encoding=True, img_metas=None
    ):  ##TODO: img_metas might be deleted
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len, input_dim)
            mask - Tensor of shape (batch_size, seq_len)
        """
        # Apply the attention block
        if add_positional_encoding:
            x = self.positional_encoding(x)

        attn_out = self.self_attn(x, mask=mask, return_attention=False)

        adapter_out = self.transormer_adapters_list[task_id](attn_out)
        x = x + adapter_out
        x = self.norm1(x)

        # Apply the MLP
        linear_out = self.mlp(x)
        x = x + linear_out
        x = self.norm2(x)

        return x


@MODELS.register_module()
class AdapterTransformerDecoder(BaseDecoder):
    def __init__(
        self,
        dictionary: Union[Dictionary, Dict],
        num_layers,
        num_classes: list,
        init_cfg=dict(type="Xavier", layer="Conv2d"),
        block_args: dict = dict(),
        module_loss: Dict = None,
        postprocessor: Dict = None,
        task_id: int = None,
        **kwargs,
    ):
        """
        Inputs:
            num_layers - Number of encoder blocks to use
            block_args - Arguments to pass to the EncoderBlock
        """
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
        )

        # Create the encoder blocks
        self.task_id = task_id
        self.layers = nn.ModuleList(
            [
                AdapterEncoderBlock(num_classes=num_classes, **block_args)
                for _ in range(num_layers)
            ]
        )
        self.num_tasks = len(num_classes)
        self.positional_encoding = PositionalEncoding
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(block_args["input_dim"], num_classes[i])
                for i in range(self.num_tasks)
            ]
        )

    def padding_mask(self, x):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len)
        """
        return padding_mask(x)

    def subsequent_mask(self, x):
        """
        Inputs:
            x - Tensor of shape (batch_size, seq_len)
        """
        return subsequent_mask(x)

    # def forward_train(self, feat, out_enc, targets_dict, img_metas, task_id: int, add_mask=True, **kwargs):
    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,  # TODO: debug this: remove default task_id=0
        add_mask: bool = True,
        img_metas=None,
    ) -> torch.Tensor:
        """
        Inputs:
            feat - Tensor of shape (batch_size, seq_len, input_dim)
            out_enc - Tensor of shape (batch_size, seq_len, input_dim)
            targets_dict - Dictionary of targets
            img_metas - List of image metadatas
            task_id - Task id
        """
        # Apply each encoder block
        # assert(task_id < self.num_tasks), "Task id must be less than the number of tasks"
        # assert task_id is not None, "Task id must be specified"
        if task_id is None:
            task_id = self.task_id
        assert feat.size(2) == 1, "feature height must be 1"
        feat.squeeze_(2)
        feat = feat.permute(0, 2, 1)
        if add_mask:
            # mask = self.padding_mask(feat)
            mask = torch.zeros(feat.size(0), feat.size(1)).to(feat.device)
            if img_metas is not None:
                for i, img_meta in enumerate(img_metas):
                    valid_shape = img_meta["resize_shape"]
                    mask[i, : math.floor(valid_shape[1] / 4)] = 1
        else:
            mask = None
        for layer in self.layers:
            feat = layer(
                feat,
                task_id=task_id,
                mask=mask,
                add_positional_encoding=True,
                img_metas=img_metas,
            )

        feat.permute(0, 2, 1).contiguous()
        feat = self.classifiers[task_id](feat)
        return feat

    # def forward_test(self, feat, out_enc, img_metas, task_id):
    #     """
    #     Args:
    #         feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

    #     Returns:
    #         Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
    #         :math:`C` is ``num_classes``.
    #     """
    #     if task_id is None:
    #         task_id = self.task_id
    #     return self.forward_train(feat, out_enc, None, img_metas, task_id)

    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,  # TODO: debug this: remove default task_id=0
    ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.
        """
        # assert task_id is not None, "Task id must be specified"
        if task_id is None:
            task_id = self.task_id
        return self.forward_train(feat, out_enc, data_samples, task_id)

    def predict(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,
    ) -> Sequence[TextRecogDataSample]:
        """Perform forward propagation of the decoder and postprocessor.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder. Defaults
                to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        if task_id is None:
            task_id = self.task_id
        out_dec = self(feat, out_enc, data_samples, task_id=task_id)
        return self.postprocessor(out_dec, data_samples)

    def forward(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,
    ) -> torch.Tensor:
        """Decoder forward.

         Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            Tensor: Features from ``decoder`` forward.
        """
        if task_id is None:
            task_id = self.task_id
        if self.training:
            if getattr(self, "module_loss") is not None:
                data_samples = self.module_loss.get_targets(data_samples)
            return self.forward_train(feat, out_enc, data_samples, task_id=task_id)
        else:
            return self.forward_test(feat, out_enc, data_samples, task_id=task_id)


class BlockRNN(nn.Module):
    def __init__(self, rnn_type, in_channels, num_classes):
        """
        :构建基础RNN-BLOCK代码
        :param rnn_type: 确定为GRU还是LSTM
        :param in_channels: 输入的channel通道
        :param num_classes: 输出每个step的分类类别
        """
        super().__init__()
        self.rnn_type = rnn_type.lower()
        assert self.rnn_type in ("gru", "lstm")
        self.num_classes = num_classes

        self.rnn = None
        if self.rnn_type == "gru":
            self.rnn = self.rnn = Sequential(
                BidirectionalGRU(in_channels, 256, 256),
                BidirectionalGRU(256, 256, num_classes),
            )

        elif self.rnn_type == "lstm":
            self.rnn = Sequential(
                BidirectionalLSTM(in_channels * 2, 256, 256),
                BidirectionalLSTM(256, 256, num_classes),
            )

    def forward(self, x):
        output = self.rnn(x)
        return output


@MODELS.register_module()
class AdapterCRNNDecoder(BaseDecoder):
    """Decoder for CRNN.
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        rnn_flag (bool): Use RNN or CNN as the decoder.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: int,
        dictionary: Union[Dictionary, Dict],
        num_tasks: int = None,
        num_classes: list = None,
        rnn_flag=True,
        rnn_type="lstm",
        module_loss: Dict = None,
        postprocessor: Dict = None,
        init_cfg=dict(type="Xavier", layer="Conv2d"),
        task_id: int = None,
        **kwargs,
    ):
        super().__init__(
            init_cfg=init_cfg,
            dictionary=dictionary,
            module_loss=module_loss,
            postprocessor=postprocessor,
        )
        self.task_id = task_id
        self.num_classes = num_classes
        self.rnn_flag = rnn_flag
        self.rnn_type = rnn_type

        if rnn_flag:
            self.decoder = nn.ModuleList(
                [
                    BlockRNN(rnn_type, in_channels, num_classes[i])
                    for i in range(num_tasks)
                ]
            )
        else:
            self.decoder = nn.ModuleList(
                [
                    nn.Conv2d(in_channels, num_classes[i], kernel_size=1, stride=1)
                    for i in range(num_tasks)
                ]
            )

    def forward_train(
        self,
        feat: torch.Tensor,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,  # TODO: debug this: remove default task_id=0
    ) -> torch.Tensor:
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        if task_id is None:
            task_id = self.task_id

        assert (
            feat.size(2) == 1
        ), (
            f"feature height must be 1, the current feature shape is {feat.shape}"
        )  # [32, 512, 2, 64]
        if self.rnn_flag:
            x = feat.squeeze(2)  # [N, C, W]
            x = x.permute(2, 0, 1)  # [W, N, C]
            for i, rnn in enumerate(self.decoder):
                if i == task_id:
                    x = rnn(x)
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            for i, rnn in enumerate(self.decoder):
                if i == task_id:
                    x = rnn(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = x.size()
            outputs = x.view(n, w, c * h)
        return outputs

    # def forward_test(self, feat, out_enc, img_metas, task_id=0): # TODO: debug this: remove default task_id=0
    #     """
    #     Args:
    #         feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

    #     Returns:
    #         Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
    #         :math:`C` is ``num_classes``.
    #     """
    def forward_test(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,  # TODO: debug this: remove default task_id=0
    ) -> torch.Tensor:
        """Forward for testing.

        Args:
            feat (torch.Tensor, optional): The feature map from backbone of
                shape :math:`(N, E, H, W)`. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output. Defaults to None.
            data_samples (Sequence[TextRecogDataSample]): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.
        """
        # assert task_id is not None, "Task id must be specified"
        if task_id is None:
            task_id = self.task_id
        return self.forward_train(feat, out_enc, data_samples, task_id)

    def predict(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,
    ) -> Sequence[TextRecogDataSample]:
        """Perform forward propagation of the decoder and postprocessor.

        Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder. Defaults
                to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            list[TextRecogDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_text``.
        """
        if task_id is None:
            task_id = self.task_id
        out_dec = self(feat, out_enc, data_samples, task_id=task_id)
        return self.postprocessor(out_dec, data_samples)

    def forward(
        self,
        feat: Optional[torch.Tensor] = None,
        out_enc: Optional[torch.Tensor] = None,
        data_samples: Optional[Sequence[TextRecogDataSample]] = None,
        task_id: int = None,
    ) -> torch.Tensor:
        """Decoder forward.

         Args:
            feat (Tensor, optional): Features from the backbone. Defaults
                to None.
            out_enc (Tensor, optional): Features from the encoder.
                Defaults to None.
            data_samples (list[TextRecogDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images. Defaults to None.

        Returns:
            Tensor: Features from ``decoder`` forward.
        """
        if task_id is None:
            task_id = self.task_id
        if self.training:
            if getattr(self, "module_loss") is not None:
                data_samples = self.module_loss.get_targets(data_samples)
            return self.forward_train(feat, out_enc, data_samples, task_id=task_id)
        else:
            return self.forward_test(feat, out_enc, data_samples, task_id=task_id)


if __name__ == "__main__":
    gru_inst = BlockRNN(rnn_type="gru", in_channels=256, num_classes=7046, dropout=0)

    lstm_inst = BlockRNN(rnn_type="lstm", in_channels=256, num_classes=7046, dropout=0)

    x = torch.rand((122, 32, 256))
    gru_inst.forward(x)

    lstm_inst.forward(x)
