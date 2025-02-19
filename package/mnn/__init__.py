# Jiale Chen, Dingling Yao, Adeel Pervez, Dan Alistarh, and Francesco Locatello. Scalable mechanistic neural networks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Oazgf8A24z.
# https://github.com/IST-DASLab/ScalableMNN


import torch

import mnn_cpp


_block_diag_2_placeholder: torch.Tensor = torch.empty((), device=torch.device('meta'))


def cholesky_inplace(
        block_diag_0: torch.Tensor,
        block_diag_1: torch.Tensor,
        block_diag_2: torch.Tensor,
        tmp_info: torch.Tensor,
        enable_ldl: bool = True,
) -> None:
    enable_block_diag_2: bool = block_diag_2 is not None
    mnn_cpp.cholesky_inplace(
        block_diag_0,
        block_diag_1,
        block_diag_2 if enable_block_diag_2 else _block_diag_2_placeholder,
        tmp_info,
        enable_ldl,
        enable_block_diag_2,
    )


def substitution_inplace(
        block_diag_0: torch.Tensor,
        block_diag_1: torch.Tensor,
        block_diag_2: torch.Tensor,
        rhs: torch.Tensor,
        enable_ldl: bool = True,
) -> None:
    enable_block_diag_2: bool = block_diag_2 is not None
    mnn_cpp.substitution_inplace(
        block_diag_0,
        block_diag_1,
        block_diag_2 if enable_block_diag_2 else _block_diag_2_placeholder,
        rhs,
        enable_ldl,
        enable_block_diag_2,
    )
