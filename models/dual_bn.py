import torch.nn as nn


class DualBN2d(nn.Module):
    """
    Element wise Dual BN, efficient implementation.
    NOTE: Boolean tensor indexing is very slow!
    """

    def __init__(self, num_features):
        """
        Args:
            num_features: int. Number of channels
        """

        super(DualBN2d, self).__init__()
        self.BN_none = nn.BatchNorm2d(num_features)
        self.BN_task = nn.BatchNorm2d(num_features)

    def forward(self, _input, _lambda):
        """
        Args:
            _input: Tensor.
            _lambda: Tensor.

        Returns:
            _output: Tensor. size=(N,C,H,W)
        """

        if _lambda is None:
            _output = self.BN_none(_input)
        else:
            _output = self.BN_task(_input)

        return _output

