"""PyTorch definition of the geographic-zone classifier.

地理区域分类神经网络定义。
A residual multilayer perceptron is used: the tabular feature vector is small
(28 dimensions) but the decision boundaries between COASTAL_SEA / NEAR_COAST
are strongly non-linear, so depth plus batch normalisation and skip
connections give a markedly better convergence than a plain MLP.

Architecture (default sizes)
----------------------------
::

    x (batch, 28)
      -> BatchNorm1d(28)                       input normalisation
      -> ResidualBlock(28  -> 256)             Linear, BN, GELU, Dropout + skip
      -> ResidualBlock(256 -> 128)
      -> ResidualBlock(128 -> 64)
      -> Linear(64 -> 4)                       logits, one per zone class

The network outputs raw logits; the softmax is applied by the loss function
during training (:class:`torch.nn.CrossEntropyLoss`) and explicitly at
inference time (:meth:`ZoneClassifier.predict_proba`,
:mod:`maritime_route.model.inference`). With the default sizes the model has
98 364 trainable parameters.

Main objects: :class:`ResidualBlock`, :class:`ZoneClassifier`.

结构：输入 BatchNorm -> 三个残差块 (256/128/64) -> 线性输出层（4 类 logits）。
网络输出未归一化的 logits，softmax 在损失函数或推理阶段计算。
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Fully connected block with batch-norm, GELU, dropout and a skip path.

    残差块：BatchNorm + GELU + Dropout，并带有线性投影的跳跃连接。

    Computes ``y = Dropout(GELU(BN(W x + b))) + P(x)`` where ``P`` is the
    identity if the input and output widths are equal and a learned linear
    projection otherwise (the widths must match for the addition).

    * **Linear** ``W x + b`` - affine map from ``in_features`` to
      ``out_features``.
    * **BatchNorm1d** - normalises every output unit to zero mean and unit
      variance over the mini-batch, then applies a learned scale and shift.
      It keeps the activations in a well-conditioned range, which allows a
      higher learning rate and faster convergence. In ``eval()`` mode running
      averages collected during training are used instead of batch
      statistics.
    * **GELU** - Gaussian Error Linear Unit, ``x * Phi(x)`` with ``Phi`` the
      standard normal CDF. A smooth alternative to ReLU that lets small
      negative values through, giving non-zero gradients for them.
    * **Dropout** - during training zeroes each activation with probability
      ``p`` and scales the survivors by ``1 / (1 - p)``; a regulariser against
      co-adaptation of neurons. Disabled in ``eval()`` mode.
    * **Skip connection** ``+ P(x)`` - the block only has to learn a
      correction to its input; gradients flow directly through the addition,
      which eases the training of deeper stacks.

    Parameters
    ----------
    in_features:
        Width of the input vector.
    out_features:
        Width of the output vector.
    dropout:
        Dropout probability ``p`` in [0, 1).
    """

    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        """Create the layers of the block (see the class docstring). 构建残差块各层。"""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Skip path: identity when widths match, otherwise a learned projection
        # so that ``h + projection(x)`` has matching shapes.
        # 跳跃连接：宽度相同用恒等映射，否则用线性投影以对齐维度。
        self.projection = (
            nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Apply the block.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, out_features)``.

        前向计算：主路径 Linear->BN->GELU->Dropout，加上跳跃连接。
        """
        # Main path: (batch, in) -> (batch, out). 主路径。
        h = self.dropout(self.activation(self.norm(self.linear(x))))
        # Residual addition. 残差相加。
        return h + self.projection(x)


class ZoneClassifier(nn.Module):
    """Four-way classifier over the engineered geographic features.

    输入: (batch, n_features) -> 输出: (batch, n_classes) 的 logits。

    The input features arrive already standardised by the scikit-learn
    scaler; the extra ``input_norm`` BatchNorm layer makes the network robust
    to small differences in scale that remain (e.g. between the binary flag
    features and the log-transformed ones).

    Parameters
    ----------
    n_features:
        Width of the input vector (28 for the current feature set).
    n_classes:
        Number of output classes (4 zone classes).
    hidden_sizes:
        Width of each residual block, from input to output.
    dropout:
        Dropout probability used in every block.

    Attributes
    ----------
    n_features, n_classes, hidden_sizes, dropout_p:
        Copies of the constructor arguments, stored so that
        :meth:`config_dict` can save them with the weights and the inference
        service can rebuild an identical network.
    input_norm:
        BatchNorm1d over the input features.
    blocks:
        :class:`torch.nn.Sequential` of :class:`ResidualBlock`.
    head:
        Final linear layer producing the logits.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 4,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        """Build the layers and initialise the weights. 构建网络层并初始化权重。"""
        super().__init__()
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout_p = float(dropout)

        self.input_norm = nn.BatchNorm1d(self.n_features)
        # Chain the blocks: the output width of one is the input width of the next.
        # 依次连接残差块：上一块的输出宽度即下一块的输入宽度。
        blocks = []
        prev = self.n_features
        for size in self.hidden_sizes:
            blocks.append(ResidualBlock(prev, size, dropout))
            prev = size
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(prev, self.n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise every linear layer with Kaiming-uniform weights and zero bias.

        Kaiming (He) initialisation draws weights from
        ``U(-b, b)`` with ``b = sqrt(6 / fan_in)`` (gain for ReLU), which keeps
        the variance of the activations roughly constant from layer to layer
        for ReLU-like non-linearities such as GELU. Zero biases start every
        unit centred. BatchNorm layers keep PyTorch's default (scale 1,
        shift 0).

        Kaiming 均匀初始化线性层权重（适用于 ReLU/GELU 类激活），偏置置零。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        """Compute class logits.

        Parameters
        ----------
        x:
            Standardised features, shape ``(batch, n_features)``, float32.
            In training mode the batch must contain at least 2 samples
            (BatchNorm needs a variance).

        Returns
        -------
        torch.Tensor
            Logits, shape ``(batch, n_classes)``; not normalised.

        前向计算，返回未归一化的 logits。
        """
        return self.head(self.blocks(self.input_norm(x)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities. 推理时返回各类别概率。

        Switches the model to evaluation mode (BatchNorm uses its running
        statistics, Dropout is disabled) and applies the softmax
        ``p_c = exp(z_c) / sum_j exp(z_j)`` over the class axis, so each row
        sums to 1. Gradient tracking is disabled by the decorator.

        Parameters
        ----------
        x:
            Standardised features, shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Probabilities, shape ``(batch, n_classes)``.
        """
        self.eval()
        return torch.softmax(self.forward(x), dim=1)

    def config_dict(self) -> dict:
        """Return the constructor arguments needed to rebuild the network.

        Saved inside the checkpoint and the metadata JSON; the inference
        service reads it to create a network of identical shape before loading
        the weights.

        Returns
        -------
        dict
            ``n_features``, ``n_classes``, ``hidden_sizes``, ``dropout`` and an
            ``architecture`` label.

        返回重建网络所需的结构参数（随权重一同保存）。
        """
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout_p,
            "architecture": "ResidualMLP",
        }

    def count_parameters(self) -> int:
        """Return the number of trainable parameters (weights and biases).

        BatchNorm running statistics are buffers, not parameters, and are not
        counted.

        Returns
        -------
        int
            Sum of the element counts of all parameters with
            ``requires_grad=True``.

        返回可训练参数总数（不含 BatchNorm 的滑动统计量）。
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
