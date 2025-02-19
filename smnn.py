# Jiale Chen, Dingling Yao, Adeel Pervez, Dan Alistarh, and Francesco Locatello. Scalable mechanistic neural networks. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Oazgf8A24z.
# https://github.com/IST-DASLab/ScalableMNN


import itertools
import math

import torch

try:
    import mnn
except ImportError:
    mnn = None


def ode_forward(
        coefficients: torch.Tensor,
        rhs_equation: torch.Tensor,
        init_vars: torch.Tensor,
        steps: torch.Tensor,
        n_steps: int = None,
        n_init_var_steps: int = None,
        is_step_dim_first: bool = False,
        weight_equation: float = 1.,
        weight_init_var: float = 1.,
        weight_smooth: float = 1.,
        enable_central_smoothness: bool = False,
        prefer_cuda_graph: bool = True,
        frozen_lhs_cache_version: int = 0,
) -> torch.Tensor:
    """
    Implementation of the ODE forward pass. The backward pass is optimized.
    This implementation has linear time & space complexity.
    ([b]: broadcastable dimensions, [e]: contiguous or expanded dimensions)

    Args:
        coefficients: (..., n_steps[b], n_equations, n_dims, n_orders)
        rhs_equation: (..., n_steps[b], n_equations[e])
        init_vars: (..., n_init_var_steps[b], n_dims[e], n_init_var_orders[e])
        steps: (..., n_steps-1[b])
        n_steps: int, optional, please specify if it cannot be inferred from the tensor shapes
        n_init_var_steps: int, optional, please specify if it cannot be inferred from the tensor shapes
        is_step_dim_first: bool, if True, the step dimension is already the first dimension
        weight_equation: float, optional weight for the least squares system
        weight_init_var: float, optional weight for the least squares system
        weight_smooth: float, optional weight for the least squares system
        enable_central_smoothness: bool, if True, the central smoothness constraint is used
        prefer_cuda_graph: bool, if True, use cuda graph if cuda device is used; automatically disabled for cpu device
        frozen_lhs_cache_version: int,
            if > 0, the left-hand side matrix is frozen and cached, please specify a unique version id for each lhs

    Returns:
        out: (..., n_steps, n_dims, n_orders)
    """

    if not is_step_dim_first:
        # move the step dimension to the first dimension
        coefficients: torch.Tensor = move_step_dim_first(coefficients, i=4, revert=False)
        rhs_equation: torch.Tensor = move_step_dim_first(rhs_equation, i=2, revert=False)
        init_vars: torch.Tensor = move_step_dim_first(init_vars, i=3, revert=False)
        steps: torch.Tensor = move_step_dim_first(steps, i=1, revert=False)

    dtype: torch.dtype = coefficients.dtype
    device: torch.device = coefficients.device

    # infer the numbers from the shapes and check if the shapes are compatible
    n_steps: int = steps.size(0) + 1 if n_steps is None else n_steps
    assert n_steps >= 2
    n_init_var_steps: int = init_vars.size(0) if n_init_var_steps is None else n_init_var_steps

    n_steps_coefficients, *batch_coefficients, n_equations, n_dims, n_orders = coefficients.shape
    assert n_steps_coefficients in [n_steps, 1]
    n_steps_rhs_equation, *batch_rhs_equation, n_equations_rhs_equation = rhs_equation.shape
    assert n_steps_rhs_equation in [n_steps, 1] and n_equations_rhs_equation == n_equations
    n_init_var_steps_rhs_init, *batch_rhs_init, n_dims_rhs_init, n_init_var_orders = init_vars.shape
    assert n_init_var_steps_rhs_init in [n_init_var_steps, 1] and n_dims_rhs_init == n_dims
    n_steps_steps, *batch_steps = steps.shape
    assert n_steps_steps in [n_steps - 1, 1]
    batch_lhs: torch.Size = torch.broadcast_shapes(batch_coefficients, batch_steps)
    batch: torch.Size = torch.broadcast_shapes(batch_lhs, batch_rhs_equation, batch_rhs_init)

    solver_options: dict = {
        'enable_central_smoothness': enable_central_smoothness,
        'frozen_lhs_cache_version': frozen_lhs_cache_version,
        'enable_cuda_graph': prefer_cuda_graph and device.type == 'cuda',
        'enable_ldl': True,
    }

    # compute the solution
    if frozen_lhs_cache_version not in LinearSolver.frozen_lhs_caches:
        block_diag_0, block_diag_1, block_diag_2, beta = compute_ata_atb(
            coefficients,
            rhs_equation,
            init_vars,
            steps,
            n_steps,
            n_init_var_steps,
            weight_equation,
            weight_init_var,
            weight_smooth,
            enable_central_smoothness,
            dtype,
            device,
            n_dims,
            n_orders,
            n_init_var_orders,
            batch,
            batch_lhs,
        )
        x: torch.Tensor = LinearSolver.apply(
            block_diag_0,
            block_diag_1,
            block_diag_2,
            beta,
            solver_options,
        )  # (n_steps, ..., n_dims * n_orders, 1)
    else:
        # use the cached left-hand side matrix
        beta: torch.Tensor = compute_atb(
            coefficients,
            rhs_equation,
            init_vars,
            n_steps,
            n_init_var_steps,
            weight_equation,
            weight_init_var,
            dtype,
            device,
            n_orders,
            n_init_var_orders,
            batch,
        )
        x: torch.Tensor = LinearSolver.apply(None, None, None, beta, solver_options)
        # (n_steps, ..., n_dims * n_orders, 1)

    x: torch.Tensor = x.reshape(n_steps, *batch, n_dims, n_orders)  # (n_steps, ..., n_dims, n_orders)
    if not is_step_dim_first:
        # move the step dimension back
        x: torch.Tensor = move_step_dim_first(x, i=3, revert=True)
    return x


def move_step_dim_first(x: torch.Tensor, i: int, revert: bool) -> torch.Tensor:
    """
    Move the step (-i-th) dimension to the first (revert=False) or last (revert=True) dimension.

    Args:
        x: (..., n_steps[b], ...)
        i: int
        revert: bool

    Returns:
        out: (n_steps[b], ...) or (..., n_steps[b])
    """

    n_tensor_dims: int = x.dim()
    if not revert:
        dim_order: list[int] = [
            n_tensor_dims - i,
            *range(n_tensor_dims - i),
            *range(n_tensor_dims - i + 1, n_tensor_dims),
        ]
    else:
        dim_order: list[int] = [*range(1, n_tensor_dims - i + 1), 0, *range(n_tensor_dims - i + 1, n_tensor_dims)]
    return x.permute(dim_order)


def compute_ata_atb(
        coefficients: torch.Tensor,
        rhs_equation: torch.Tensor,
        init_vars: torch.Tensor,
        steps: torch.Tensor,
        n_steps: int,
        n_init_var_steps: int,
        weight_equation: float,
        weight_init_var: float,
        weight_smooth: float,
        enable_central_smoothness: bool,
        dtype: torch.dtype,
        device: torch.device,
        n_dims: int,
        n_orders: int,
        n_init_var_orders: int,
        batch: torch.Size,
        batch_lhs: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the matrix A^T A and the vector A^T b for the linear solver. The backward pass is by autograd.
    ([b]: broadcastable dimensions, [e]: contiguous or expanded dimensions)

    Args:
        coefficients: (n_steps[b], ..., n_equations, n_dims, n_orders)
        rhs_equation: (n_steps[b], ..., n_equations[e])
        init_vars: (n_init_var_steps[b], ..., n_dims[e], n_init_var_orders[e])
        steps: (n_steps-1[b], ...)
        n_steps: int
        n_init_var_steps: int
        weight_equation: float
        weight_init_var: float
        weight_smooth: float
        enable_central_smoothness: bool
        dtype: torch.dtype
        device: torch.device
        n_dims: int
        n_orders: int
        n_init_var_orders: int
        batch: torch.Size
        batch_lhs: torch.Size

    Returns:
        block_diag_0: (n_steps, ..., n_dims * n_orders, n_dims * n_orders),
        block_diag_1: (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders),
        block_diag_2: (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders),
        beta: (n_steps, ..., n_dims * n_orders, 1),
    """

    # ode equation constraints
    c: torch.Tensor = coefficients.flatten(start_dim=-2)  # (n_steps[b], ..., n_equations, n_dims * n_orders)
    ct: torch.Tensor = c.transpose(-2, -1) * weight_equation ** 2  # (n_steps[b], ..., n_dims * n_orders, n_equations)
    block_diag_0: torch.Tensor = ct @ c  # (n_steps[b], ..., n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = ct @ rhs_equation[..., None]  # (n_steps[b], ..., n_dims * n_orders, 1)

    block_diag_0: torch.Tensor = block_diag_0.repeat(
        n_steps // block_diag_0.size(0),
        *[ss // s for ss, s in zip(batch_lhs, block_diag_0.shape[1:-2])],
        1,
        1,
    )  # (n_steps, ..., n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = beta.repeat(
        n_steps // beta.size(0),
        *[ss // s for ss, s in zip(batch, beta.shape[1:-2])],
        1,
        1,
    )  # (n_steps, ..., n_dims * n_orders, 1)

    # initial-value constraints
    weight2_init_var: float = weight_init_var ** 2
    init_idx: torch.Tensor = torch.arange(n_init_var_orders, device=device).repeat(n_dims) \
                             + (n_orders * torch.arange(n_dims, device=device)).repeat_interleave(n_init_var_orders)
    # (n_dims * n_init_var_orders)
    block_diag_0[:n_init_var_steps, ..., init_idx, init_idx] += weight2_init_var
    beta[:n_init_var_steps, ..., :, 0] += torch.cat([
        init_vars * weight2_init_var,
        torch.zeros(*init_vars.shape[:-1], n_orders - n_init_var_orders, dtype=dtype, device=device),
    ], dim=-1).flatten(start_dim=-2)

    # smoothness constraints (forward & backward)
    order_idx: torch.Tensor = torch.arange(n_orders, device=device)  # (n_orders)
    sign_vec: torch.Tensor = order_idx % 2 * (-2) + 1  # (n_orders)
    sign_map: torch.Tensor = sign_vec * sign_vec[:, None]  # (n_orders, n_orders)

    expansions: torch.Tensor = steps[..., None] ** order_idx * weight_smooth  # (n_steps-1[b], ..., n_orders)
    et_e_diag: torch.Tensor = expansions ** 2  # (n_steps-1[b], ..., n_orders)
    e_outer: torch.Tensor = expansions[..., None] * expansions[..., None, :]  # (n_steps-1[b], ..., n_orders, n_orders)
    factorials: torch.Tensor = (-(order_idx - order_idx[:, None] + 1).triu().to(dtype=dtype).lgamma()).exp()
    # (n_orders, n_orders)
    if enable_central_smoothness:
        et_e_diag[..., -1] = 0.
        factorials[-1, -1] = 0.
    et_ft_f_e: torch.Tensor = e_outer * (factorials.t() @ factorials)  # (n_steps-1[b], ..., n_orders, n_orders)

    smooth_block_diag_1: torch.Tensor = e_outer * -(factorials + factorials.transpose(-2, -1) * sign_map)
    # (n_steps-1[b], ..., n_orders, n_orders)
    smooth_block_diag_0: torch.Tensor = torch.zeros(n_steps, *batch_lhs, n_orders, n_orders, dtype=dtype, device=device)
    # (n_steps, ..., n_orders, n_orders)
    smooth_block_diag_0[:-1] += et_ft_f_e
    smooth_block_diag_0[1:] += et_ft_f_e * sign_map
    smooth_block_diag_0[:-1, ..., order_idx, order_idx] += et_e_diag
    smooth_block_diag_0[1:, ..., order_idx, order_idx] += et_e_diag

    smooth_block_diag_1: torch.Tensor = smooth_block_diag_1.repeat(
        (n_steps - 1) // smooth_block_diag_1.size(0),
        *([1] * len(batch_lhs)),
        1,
        1,
    )  # (n_steps-1, ..., n_orders, n_orders)
    block_diag_1: torch.Tensor = torch.zeros(
        n_steps - 1, *batch_lhs, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device,
    )  # (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders)

    if enable_central_smoothness:
        steps: torch.Tensor = steps.repeat((n_steps - 1) // steps.size(0), *([1] * len(batch_lhs)))  # (n_steps-1, ...)

        # smoothness constraints (central)
        steps2: torch.Tensor = steps[:-1] + steps[1:]  # (n_steps-2, ...)
        weight2_smooth: float = weight_smooth ** 2
        steps26: torch.Tensor = steps2 ** (n_orders * 2 - 6) * weight2_smooth  # (n_steps-2, ...)
        steps25: torch.Tensor = steps2 ** (n_orders * 2 - 5) * weight2_smooth  # (n_steps-2, ...)
        steps24: torch.Tensor = steps2 ** (n_orders * 2 - 4) * weight2_smooth  # (n_steps-2, ...)

        smooth_block_diag_0[:-2, ..., n_orders - 2, n_orders - 2] += steps26
        smooth_block_diag_0[2:, ..., n_orders - 2, n_orders - 2] += steps26
        smooth_block_diag_0[1:-1, ..., n_orders - 1, n_orders - 1] += steps24
        smooth_block_diag_1[:-1, ..., n_orders - 1, n_orders - 2] += steps25
        smooth_block_diag_1[1:, ..., n_orders - 2, n_orders - 1] -= steps25
        smooth_block_diag_2: torch.Tensor | None = torch.zeros(
            n_steps - 2, *batch_lhs, n_orders, n_orders, dtype=dtype, device=device,
        )  # (n_steps-2, ..., n_orders, n_orders)
        smooth_block_diag_2[..., n_orders - 2, n_orders - 2] = -steps26

        block_diag_2: torch.Tensor | None = torch.zeros(
            n_steps - 2, *batch_lhs, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device,
        )  # (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders
    else:
        smooth_block_diag_2 = None
        block_diag_2 = None

    # copy to n_dims
    for dim in range(n_dims):
        i1: int = dim * n_orders
        i2: int = (dim + 1) * n_orders
        block_diag_0[..., i1:i2, i1:i2] += smooth_block_diag_0
        block_diag_1[..., i1:i2, i1:i2] = smooth_block_diag_1
        if block_diag_2 is not None:
            block_diag_2[..., i1:i2, i1:i2] = smooth_block_diag_2

    return block_diag_0, block_diag_1, block_diag_2, beta


def compute_atb(
        coefficients: torch.Tensor,
        rhs_equation: torch.Tensor,
        init_vars: torch.Tensor,
        n_steps: int,
        n_init_var_steps: int,
        weight_equation: float,
        weight_init_var: float,
        dtype: torch.dtype,
        device: torch.device,
        n_orders: int,
        n_init_var_orders: int,
        batch: torch.Size,
) -> torch.Tensor:
    """
    Compute the vector A^T b for the linear solver. The backward pass is by autograd.
    ([b]: broadcastable dimensions, [e]: contiguous or expanded dimensions)

    Args:
        coefficients: (n_steps[b], ..., n_equations, n_dims, n_orders)
        rhs_equation: (n_steps[b], ..., n_equations[e])
        init_vars: (n_init_var_steps[b], ..., n_dims[e], n_init_var_orders[e])
        n_steps: int
        n_init_var_steps: int
        weight_equation: float
        weight_init_var: float
        dtype: torch.dtype
        device: torch.device
        n_orders: int
        n_init_var_orders: int
        batch: torch.Size

    Returns:
        beta: (n_steps, ..., n_dims * n_orders, 1)
    """

    # ode equation constraints
    beta: torch.Tensor = coefficients.flatten(start_dim=-2).transpose(-2, -1) @ (
        rhs_equation[..., None] * weight_equation ** 2
    )  # (n_steps[b], ..., n_dims * n_orders, 1)
    beta: torch.Tensor = beta.repeat(
        n_steps // beta.size(0),
        *[ss // s for ss, s in zip(batch, beta.shape[1:-2])],
        1,
        1,
    )  # (n_steps, ..., n_dims * n_orders, 1)
    # initial-value constraints
    beta[:n_init_var_steps, ..., :, 0] += torch.cat([
        init_vars * weight_init_var ** 2,
        torch.zeros(*init_vars.shape[:-1], n_orders - n_init_var_orders, dtype=dtype, device=device),
    ], dim=-1).flatten(start_dim=-2)
    return beta


class LinearSolver(torch.autograd.Function):
    frozen_lhs_caches: dict[int, tuple] = {}
    cuda_graph_info: dict[tuple, dict] = {}

    @staticmethod
    def forward(
            ctx,
            block_diag_0: torch.Tensor,
            block_diag_1: torch.Tensor,
            block_diag_2: torch.Tensor | None,
            rhs: torch.Tensor,
            solver_options: dict,
    ) -> torch.Tensor:
        """
        Solver forward pass.

        Args:
            ctx: torch.autograd.function.BackwardCFunction
            block_diag_0: (n_steps, ..., n_dims * n_orders, n_dims * n_orders)
            block_diag_1: (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders)
            block_diag_2: (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders) or None
            rhs: (n_steps, ..., n_dims * n_orders, 1)
            solver_options: dict

        Returns:
            out: (n_steps, ..., n_dims * n_orders, 1)
        """

        solver = mnn if mnn is not None else LinearSolver
        dtype: torch.dtype = rhs.dtype
        device: torch.device = rhs.device
        n_steps, *batch, n_rows, _ = rhs.shape

        ctx.solver_options = solver_options
        enable_central_smoothness: bool = solver_options['enable_central_smoothness']
        frozen_lhs_cache_version: int = solver_options['frozen_lhs_cache_version']
        enable_cuda_graph: bool = solver_options['enable_cuda_graph']
        enable_ldl: bool = solver_options['enable_ldl']

        if enable_cuda_graph:
            torch.cuda.set_device(device)
            graph_key: tuple = rhs.shape, dtype, device, enable_central_smoothness, frozen_lhs_cache_version, enable_ldl
            if graph_key not in LinearSolver.cuda_graph_info:
                # build the CUDA graphs
                graph_forward: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
                graph_substitution: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
                graph_tensors: dict[str, torch.Tensor] = {
                    'block_diag_0': torch.empty(n_steps, *batch, n_rows, n_rows, dtype=dtype, device=device),
                    'block_diag_1': torch.empty(n_steps - 1, *batch, n_rows, n_rows, dtype=dtype, device=device),
                    'block_diag_2': torch.empty(n_steps - 2, *batch, n_rows, n_rows, dtype=dtype, device=device)
                        if enable_central_smoothness else None,
                    'rhs': torch.empty(n_steps, *batch, n_rows, 1, dtype=dtype, device=device),
                    'tmp_info': torch.empty(batch, dtype=torch.int32, device=device),
                }
                n_cuda_graph_warmups: int = 10
                s: torch.cuda.Stream = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(n_cuda_graph_warmups):
                        solver.cholesky_inplace(
                            graph_tensors['block_diag_0'],
                            graph_tensors['block_diag_1'],
                            graph_tensors['block_diag_2'],
                            graph_tensors['tmp_info'],
                            enable_ldl=enable_ldl,
                        )
                        solver.substitution_inplace(
                            graph_tensors['block_diag_0'],
                            graph_tensors['block_diag_1'],
                            graph_tensors['block_diag_2'],
                            graph_tensors['rhs'],
                            enable_ldl=enable_ldl,
                        )
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(graph_forward):
                    solver.cholesky_inplace(
                        graph_tensors['block_diag_0'],
                        graph_tensors['block_diag_1'],
                        graph_tensors['block_diag_2'],
                        graph_tensors['tmp_info'],
                        enable_ldl=enable_ldl,
                    )
                    solver.substitution_inplace(
                        graph_tensors['block_diag_0'],
                        graph_tensors['block_diag_1'],
                        graph_tensors['block_diag_2'],
                        graph_tensors['rhs'],
                        enable_ldl=enable_ldl,
                    )
                s: torch.cuda.Stream = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(n_cuda_graph_warmups):
                        solver.substitution_inplace(
                            graph_tensors['block_diag_0'],
                            graph_tensors['block_diag_1'],
                            graph_tensors['block_diag_2'],
                            graph_tensors['rhs'],
                            enable_ldl=enable_ldl,
                        )
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(graph_substitution):
                    solver.substitution_inplace(
                        graph_tensors['block_diag_0'],
                        graph_tensors['block_diag_1'],
                        graph_tensors['block_diag_2'],
                        graph_tensors['rhs'],
                        enable_ldl=enable_ldl,
                    )
                torch.cuda.synchronize(device)
                LinearSolver.cuda_graph_info[graph_key] = {
                    'graph_forward': graph_forward,
                    'graph_substitution': graph_substitution,
                    'graph_tensors': graph_tensors,
                }

            # replay the CUDA graphs
            graph_info: dict = LinearSolver.cuda_graph_info[graph_key]
            graph_tensors: dict[str, torch.Tensor] = graph_info['graph_tensors']
            if frozen_lhs_cache_version not in LinearSolver.frozen_lhs_caches:
                graph_tensors['block_diag_0'].copy_(block_diag_0)
                graph_tensors['block_diag_1'].copy_(block_diag_1)
                if enable_central_smoothness:
                   graph_tensors['block_diag_2'].copy_(block_diag_2)
                graph_tensors['rhs'].copy_(rhs)
                graph_forward: torch.cuda.CUDAGraph = graph_info['graph_forward']
                graph_forward.replay()
                if frozen_lhs_cache_version != 0:
                    LinearSolver.frozen_lhs_caches[frozen_lhs_cache_version] = (
                        graph_tensors['block_diag_0'],
                        graph_tensors['block_diag_1'],
                        graph_tensors['block_diag_2'],
                    )
            else:
                graph_tensors['rhs'].copy_(rhs)
                graph_substitution: torch.cuda.CUDAGraph = graph_info['graph_substitution']
                graph_substitution.replay()
            rhs: torch.Tensor = graph_tensors['rhs']  # (n_steps, ..., n_dims * n_orders, 1)

        else:  # not enable_cuda_graph
            if not frozen_lhs_cache_version in LinearSolver.frozen_lhs_caches:
                # compute the LDL/Cholesky decomposition
                tmp_info: torch.Tensor = torch.empty(batch, dtype=torch.int32, device=device)  # (...)
                solver.cholesky_inplace(
                    block_diag_0,
                    block_diag_1,
                    block_diag_2,
                    tmp_info,
                    enable_ldl=enable_ldl,
                )
                if frozen_lhs_cache_version != 0:
                    LinearSolver.frozen_lhs_caches[frozen_lhs_cache_version] = block_diag_0, block_diag_1, block_diag_2
            else:
                # use the cached left-hand side matrix
                block_diag_0, block_diag_1, block_diag_2 = LinearSolver.frozen_lhs_caches[frozen_lhs_cache_version]
            # solve for the results
            solver.substitution_inplace(
                block_diag_0,
                block_diag_1,
                block_diag_2,
                rhs,
                enable_ldl=enable_ldl,
            )
            # store the variables for the backward pass
            ctx.block_diag_0, ctx.block_diag_1, ctx.block_diag_2, ctx.x = block_diag_0, block_diag_1, block_diag_2, rhs

        return rhs.clone()

    @staticmethod
    def backward(
            ctx,
            rhs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor, None]:
        """
        Solver backward pass.

        Args:
            ctx: torch.autograd.function.BackwardCFunction
            rhs: (n_steps, ..., n_dims * n_orders, 1)

        Returns:
            da0: (n_steps, ..., n_dims * n_orders, n_dims * n_orders)
            da1: (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders)
            da2: (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders) or None
            db: (n_steps, ..., n_dims * n_orders, 1)
            None
        """

        dtype: torch.dtype = rhs.dtype
        device: torch.device = rhs.device

        solver_options: dict = ctx.solver_options
        enable_central_smoothness: bool = solver_options['enable_central_smoothness']
        frozen_lhs_cache_version: int = solver_options['frozen_lhs_cache_version']
        enable_cuda_graph: bool = solver_options['enable_cuda_graph']
        enable_ldl: bool = solver_options['enable_ldl']

        if enable_cuda_graph:
            # replay the CUDA graphs
            graph_key: tuple = rhs.shape, dtype, device, enable_central_smoothness, frozen_lhs_cache_version, enable_ldl
            graph_info: dict = LinearSolver.cuda_graph_info[graph_key]
            graph_substitution: torch.cuda.CUDAGraph = graph_info['graph_substitution']
            graph_tensors: dict[str, torch.Tensor] = graph_info['graph_tensors']
            x: torch.Tensor = -graph_tensors['rhs']  # (n_steps, ..., n_dims * n_orders, 1)
            graph_tensors['rhs'].copy_(rhs)
            graph_substitution.replay()
            rhs: torch.Tensor = graph_tensors['rhs']
        else:  # not enable_cuda_graph
            x: torch.Tensor = -ctx.x  # (n_steps, ..., n_dims * n_orders, 1)
            rhs: torch.Tensor = rhs.clone()
            solver = mnn if mnn is not None else LinearSolver
            solver.substitution_inplace(
                ctx.block_diag_0, ctx.block_diag_1, ctx.block_diag_2, rhs, enable_ldl=enable_ldl,
            )

        if frozen_lhs_cache_version == 0:
            # compute the gradients
            da0: torch.Tensor | None = rhs * x[..., None, :, 0]
            da1: torch.Tensor | None = rhs[1:] * x[:-1, ..., None, :, 0] + x[1:] * rhs[:-1, ..., None, :, 0]
            if enable_central_smoothness:
                da2: torch.Tensor | None = rhs[2:] * x[:-2, ..., None, :, 0] + x[2:] * rhs[:-2, ..., None, :, 0]
            else:
                da2 = None
        else:
            da0 = da1 = da2 = None

        return da0, da1, da2, rhs, None

    @staticmethod
    def cholesky_inplace(
        block_diag_0: torch.Tensor | list[torch.Tensor],
        block_diag_1: torch.Tensor | list[torch.Tensor],
        block_diag_2: torch.Tensor | list[torch.Tensor] | None,
        tmp_info: torch.Tensor,
        enable_ldl: bool = True,
    ) -> None:
        """
        LDL/Cholesky decomposition of the block diagonal matrix (inplace).

        Args:
            block_diag_0: (n_steps, ..., n_dims * n_orders, n_dims * n_orders), inplace
            block_diag_1: (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders), inplace
            block_diag_2: (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders), inplace
            tmp_info: (...), inplace
            enable_ldl: bool, if True, the LDL decomposition is used instead of Cholesky decomposition

        Returns:
            None
        """

        enable_block_diag_2: bool = block_diag_2 is not None
        n_steps: int = len(block_diag_0)
        for step in range(n_steps):
            if enable_block_diag_2 and step >= 2:
                torch.linalg.solve_triangular(
                    block_diag_0[step - 2].transpose(-2, -1),
                    block_diag_2[step - 2],
                    upper=True,
                    left=False,
                    unitriangular=False,
                    out=block_diag_2[step - 2],
                )  # block_diag_2[step - 2] @= block_diag_0[step - 2].t().inv()
                LinearSolver.bsubbmm(
                    block_diag_1[step - 1],
                    block_diag_2[step - 2],
                    block_diag_1[step - 2].transpose(-2, -1),
                )  # block_diag_1[step - 1] -= block_diag_2[step - 2] @ block_diag_1[step - 2].t()
            if step >= 1:
                torch.linalg.solve_triangular(
                    block_diag_0[step - 1].transpose(-2, -1),
                    block_diag_1[step - 1],
                    upper=True,
                    left=False,
                    unitriangular=False,
                    out=block_diag_1[step - 1],
                )  # block_diag_1[step - 1] @= block_diag_0[step - 1].t().inv()
                if enable_block_diag_2 and step >= 2:
                    LinearSolver.bsubbmm(
                        block_diag_0[step],
                        block_diag_2[step - 2],
                        block_diag_2[step - 2].transpose(-2, -1),
                    )  # block_diag_0[step] -= block_diag_2[step - 2] @ block_diag_2[step - 2].t()
                LinearSolver.bsubbmm(
                    block_diag_0[step],
                    block_diag_1[step - 1],
                    block_diag_1[step - 1].transpose(-2, -1),
                )  # block_diag_0[step] -= block_diag_1[step - 1] @ block_diag_1[step - 1].t()
            torch.linalg.cholesky_ex(
                block_diag_0[step],
                upper=False,
                check_errors=False,
                out=(block_diag_0[step], tmp_info),
            )

        if enable_ldl:
            # LDL decomposition https://en.wikipedia.org/wiki/Cholesky_decomposition#Block_variant
            # block_diag_0: Cholesky decomposition of D
            # block_diag_1: L blocks
            # block_diag_2: L blocks
            torch.linalg.solve_triangular(
                block_diag_0[:-1],
                block_diag_1,
                upper=False,
                left=False,
                unitriangular=False,
                out=block_diag_1,
            )  # block_diag_1 @= block_diag_0[:-1].inv()
            if enable_block_diag_2:
                torch.linalg.solve_triangular(
                    block_diag_0[:-2],
                    block_diag_2,
                    upper=False,
                    left=False,
                    unitriangular=False,
                    out=block_diag_2,
                )  # block_diag_2 @= block_diag_0[:-2].inv()

    @staticmethod
    def substitution_inplace(
            block_diag_0: torch.Tensor | list[torch.Tensor],
            block_diag_1: torch.Tensor | list[torch.Tensor],
            block_diag_2: torch.Tensor | list[torch.Tensor] | None,
            rhs: torch.Tensor | list[torch.Tensor],
            enable_ldl: bool = True,
    ) -> None:
        """
        Solve for the results using the substitution algorithm (inplace).

        Args:
            block_diag_0: (n_steps, ..., n_dims * n_orders, n_dims * n_orders)
            block_diag_1: (n_steps-1, ..., n_dims * n_orders, n_dims * n_orders)
            block_diag_2: (n_steps-2, ..., n_dims * n_orders, n_dims * n_orders)
            rhs: (n_steps, ..., n_dims * n_orders, 1), inplace
            enable_ldl: bool, if True, the LDL decomposition is used instead of Cholesky decomposition

        Returns:
            None
        """

        enable_block_diag_2: bool = block_diag_2 is not None
        n_steps: int = len(block_diag_0)

        # A X = B => L (D (Lt X)) = B
        for step in range(n_steps):
            # solve L Z = B, block forward substitution
            if enable_block_diag_2 and step >= 2:
                LinearSolver.bsubbmm(
                    rhs[step],
                    block_diag_2[step - 2],
                    rhs[step - 2],
                )  # rhs[step] -= block_diag_2[step - 2] @ rhs[step - 2]
            if step >= 1:
                LinearSolver.bsubbmm(
                    rhs[step],
                    block_diag_1[step - 1],
                    rhs[step - 1],
                )  # rhs[step] -= block_diag_1[step - 1] @ rhs[step - 1]
            if not enable_ldl:
                torch.linalg.solve_triangular(
                    block_diag_0[step],
                    rhs[step],
                    upper=False,
                    left=True,
                    unitriangular=False,
                    out=rhs[step],
                )  # rhs[step] = block_diag_0[step].inv() @ rhs[step]
        if enable_ldl:
            # solve D Y = Z, block forward substitution
            # torch.cholesky_solve(
            #     rhs,
            #     block_diag_0,
            #     upper=False,
            #     out=rhs,
            # )  # rhs = (block_diag_0 @ block_diag_0.t()).inv() @ rhs
            # the above is slow so we use the following instead
            torch.linalg.solve_triangular(
                block_diag_0,
                rhs,
                upper=False,
                left=True,
                unitriangular=False,
                out=rhs,
            )  # rhs = block_diag_0.inv() @ rhs
            torch.linalg.solve_triangular(
                block_diag_0.transpose(-2, -1),
                rhs,
                upper=True,
                left=True,
                unitriangular=False,
                out=rhs,
            )  # rhs = block_diag_0.t().inv() @ rhs
        for step in range(n_steps - 1, -1, -1):
            # solve Lt X = Y, block backward substitution
            if enable_block_diag_2 and step < n_steps - 2:
                LinearSolver.bsubbmm(
                    rhs[step],
                    block_diag_2[step].transpose(-2, -1),
                    rhs[step + 2],
                )  # rhs[step] -= block_diag_2[step].t() @ rhs[step + 2]
            if step < n_steps - 1:
                LinearSolver.bsubbmm(
                    rhs[step],
                    block_diag_1[step].transpose(-2, -1),
                    rhs[step + 1],
                )  # rhs[step] -= block_diag_1[step].t() @ rhs[step + 1]
            if not enable_ldl:
                torch.linalg.solve_triangular(
                    block_diag_0[step].transpose(-2, -1),
                    rhs[step],
                    upper=True,
                    left=True,
                    unitriangular=False,
                    out=rhs[step],
                )  # rhs[step] = block_diag_0[step].t().inv() @ rhs[step]

    @staticmethod
    def bsubbmm(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
        """
        Compute c -= a @ b

        Args:
            c: (..., n_dims * n_orders, 1)
            a: (..., n_dims * n_orders, n_dims * n_orders)
            b: (..., n_dims * n_orders, 1)

        Returns:
            None
        """

        c -= a @ b  # (..., n_dims * n_orders, 1)
        # not yet supporting multiple batch dims
        # a, b, c = a.flatten(end_dim=-3), b.flatten(end_dim=-3), c.flatten(end_dim=-3)
        # torch.baddbmm(c, a, b, beta=1, alpha=-1, out=c)


def ode_forward_basic(
        coefficients: torch.Tensor,
        rhs_equation: torch.Tensor,
        rhs_init: torch.Tensor,
        steps: torch.Tensor,
        n_steps: int = None,
        n_init_var_steps: int = None,
) -> torch.Tensor:
    """
    Basic (simplest) implementation of the ODE forward pass. The backward pass is fully by autograd.
    This implementation has linear time & space complexity.
    ([b]: broadcastable dimensions, [e]: contiguous or expanded dimensions)

    Args:
        coefficients: (..., n_steps[b], n_equations, n_dims, n_orders)
        rhs_equation: (..., n_steps[b], n_equations[e])
        rhs_init: (..., n_init_var_steps[b], n_dims[e], n_init_var_orders[e])
        steps: (..., n_steps-1[b])
        n_steps: int, optional, please specify if it cannot be inferred from the tensor shapes
        n_init_var_steps: int, optional, please specify if it cannot be inferred from the tensor shapes

    Returns:
        out: (..., n_steps, n_dims, n_orders)
    """

    dtype: torch.dtype = coefficients.dtype
    device: torch.device = coefficients.device

    n_steps: int = steps.size(-1) + 1 if n_steps is None else n_steps
    assert n_steps >= 2
    n_init_var_steps: int = rhs_init.size(-3) if n_init_var_steps is None else n_init_var_steps

    *batch_coefficients, n_steps_coefficients, n_equations, n_dims, n_orders = coefficients.shape
    assert n_steps_coefficients in [n_steps, 1]
    *batch_rhs_equation, n_steps_rhs_equation, n_equations_rhs_equation = rhs_equation.shape
    assert n_steps_rhs_equation in [n_steps, 1] and n_equations_rhs_equation == n_equations
    *batch_rhs_init, n_init_var_steps_rhs_init, n_dims_rhs_init, n_init_var_orders = rhs_init.shape
    assert n_init_var_steps_rhs_init in [n_init_var_steps, 1] and n_dims_rhs_init == n_dims
    *batch_steps, n_steps_steps = steps.shape
    assert n_steps_steps in [n_steps - 1, 1]
    batch_lhs: torch.Size = torch.broadcast_shapes(batch_coefficients, batch_steps)
    batch: torch.Size = torch.broadcast_shapes(batch_lhs, batch_rhs_equation, batch_rhs_init)

    # ode equation constraints
    c: torch.Tensor = coefficients.flatten(start_dim=-2)  # (..., n_steps[b], n_equations, n_dims * n_orders)
    ct: torch.Tensor = c.transpose(-2, -1)  # (..., n_steps[b], n_dims * n_orders, n_equations)
    block_diag_0: torch.Tensor = ct @ c  # (..., n_steps[b], n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = ct @ rhs_equation[..., None]  # (..., n_steps[b], n_dims * n_orders, 1)

    block_diag_0: torch.Tensor = block_diag_0.repeat(
        *[ss // s for ss, s in zip(batch_lhs, block_diag_0.shape[:-3])],
        n_steps // block_diag_0.size(-3),
        1,
        1,
    )  # (..., n_steps, n_dims * n_orders, n_dims * n_orders)
    beta: torch.Tensor = beta.repeat(
        *[ss // s for ss, s in zip(batch, beta.shape[:-3])],
        n_steps // beta.size(-3),
        1,
        1,
    )  # (..., n_steps, n_dims * n_orders, 1)

    # initial-value constraints
    init_idx: torch.Tensor = torch.arange(n_init_var_orders, device=device).repeat(n_dims) \
                             + (n_orders * torch.arange(n_dims, device=device)).repeat_interleave(n_init_var_orders)
    # (n_dims * n_init_var_orders)
    block_diag_0[..., :n_init_var_steps, init_idx, init_idx] += 1.
    beta[..., :n_init_var_steps, :, 0] += torch.cat([
        rhs_init,
        torch.zeros(*rhs_init.shape[:-1], n_orders - n_init_var_orders, dtype=dtype, device=device),
    ], dim=-1).flatten(start_dim=-2)

    # smoothness constraints (forward & backward)
    order_idx: torch.Tensor = torch.arange(n_orders, device=device)  # (n_orders)
    sign_vec: torch.Tensor = order_idx % 2 * (-2) + 1  # (n_orders)
    sign_map: torch.Tensor = sign_vec * sign_vec[:, None]  # (n_orders, n_orders)

    expansions: torch.Tensor = steps[..., None] ** order_idx  # (..., n_steps-1[b], n_orders)
    et_e_diag: torch.Tensor = expansions ** 2  # (..., n_steps-1[b], n_orders)
    et_e_diag[..., -1] = 0.
    factorials: torch.Tensor = (-(order_idx - order_idx[:, None] + 1).triu().to(dtype=dtype).lgamma()).exp()
    # (n_orders, n_orders)
    factorials[-1, -1] = 0.
    e_outer: torch.Tensor = expansions[..., None] * expansions[..., None, :]  # (..., n_steps-1[b], n_orders, n_orders)
    et_ft_f_e: torch.Tensor = e_outer * (factorials.t() @ factorials)  # (..., n_steps-1[b], n_orders, n_orders)

    smooth_block_diag_1: torch.Tensor = e_outer * -(factorials + factorials.transpose(-2, -1) * sign_map)
    # (..., n_steps-1[b], n_orders, n_orders)
    smooth_block_diag_0: torch.Tensor = torch.zeros(*batch_lhs, n_steps, n_orders, n_orders, dtype=dtype, device=device)
    # (..., n_steps, n_orders, n_orders)
    smooth_block_diag_0[..., :-1, :, :] += et_ft_f_e
    smooth_block_diag_0[..., 1:, :, :] += et_ft_f_e * sign_map
    smooth_block_diag_0[..., :-1, order_idx, order_idx] += et_e_diag
    smooth_block_diag_0[..., 1:, order_idx, order_idx] += et_e_diag

    smooth_block_diag_1: torch.Tensor = smooth_block_diag_1.repeat(
        *([1] * len(batch_lhs)),
        (n_steps - 1) // smooth_block_diag_1.size(-3),
        1,
        1,
    )  # (..., n_steps-1, n_orders, n_orders)
    steps: torch.Tensor = steps.repeat(*([1] * len(batch_lhs)), (n_steps - 1) // steps.size(-1))  # (..., n_steps-1)

    # smoothness constraints (central)
    steps2: torch.Tensor = steps[..., :-1] + steps[..., 1:]  # (..., n_steps-2)
    steps26: torch.Tensor = steps2 ** (n_orders * 2 - 6)  # (..., n_steps-2)
    steps25: torch.Tensor = steps2 ** (n_orders * 2 - 5)  # (..., n_steps-2)
    steps24: torch.Tensor = steps2 ** (n_orders * 2 - 4)  # (..., n_steps-2)

    smooth_block_diag_0[..., :-2, n_orders - 2, n_orders - 2] += steps26
    smooth_block_diag_0[..., 2:, n_orders - 2, n_orders - 2] += steps26
    smooth_block_diag_0[..., 1:-1, n_orders - 1, n_orders - 1] += steps24
    smooth_block_diag_1[..., :-1, n_orders - 1, n_orders - 2] += steps25
    smooth_block_diag_1[..., 1:, n_orders - 2, n_orders - 1] -= steps25
    smooth_block_diag_2: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 2, n_orders, n_orders, dtype=dtype, device=device,
    )  # (..., n_steps-2, n_orders, n_orders)
    smooth_block_diag_2[..., n_orders - 2, n_orders - 2] = -steps26

    # copy to n_dims
    block_diag_1: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 1, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device,
    )  # (..., n_steps-1, n_dims * n_orders, n_dims * n_orders)
    block_diag_2: torch.Tensor = torch.zeros(
        *batch_lhs, n_steps - 2, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device,
    )  # (..., n_steps-2, n_dims * n_orders, n_dims * n_orders)
    for dim in range(n_dims):
        i1: int = dim * n_orders
        i2: int = (dim + 1) * n_orders
        block_diag_0[..., i1:i2, i1:i2] += smooth_block_diag_0
        block_diag_1[..., i1:i2, i1:i2] = smooth_block_diag_1
        block_diag_2[..., i1:i2, i1:i2] = smooth_block_diag_2

    # blocked cholesky decomposition
    block_diag_0_list: list[torch.Tensor] = list(block_diag_0.unbind(dim=-3))
    block_diag_1_list: list[torch.Tensor] = list(block_diag_1.unbind(dim=-3))
    block_diag_2_list: list[torch.Tensor] = list(block_diag_2.unbind(dim=-3))
    for step in range(n_steps):
        if step >= 2:
            block_diag_2_list[step - 2] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 2].transpose(-2, -1),
                block_diag_2_list[step - 2],
                upper=True,
                left=False,
            )
            block_diag_1_list[step - 1] = block_diag_1_list[step - 1] \
                                          - block_diag_2_list[step - 2] @ block_diag_1_list[step - 2].transpose(-2, -1)
        if step >= 1:
            block_diag_1_list[step - 1] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 1].transpose(-2, -1),
                block_diag_1_list[step - 1],
                upper=True,
                left=False,
            )
            if step >= 2:
                block_diag_0_list[step] = block_diag_0_list[step] \
                                          - block_diag_2_list[step - 2] @ block_diag_2_list[step - 2].transpose(-2, -1)
            block_diag_0_list[step] = block_diag_0_list[step] \
                                      - block_diag_1_list[step - 1] @ block_diag_1_list[step - 1].transpose(-2, -1)
        block_diag_0_list[step], _ = torch.linalg.cholesky_ex(
            block_diag_0_list[step],
            upper=False,
            check_errors=False,
        )

    # A X = B => L (Lt X) = B
    # solve L Y = B, block forward substitution
    b_list: list[torch.Tensor] = list(beta.unbind(dim=-3))
    y_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps):
        b_step: torch.Tensor = b_list[step]
        if step >= 2:
            b_step = b_step - block_diag_2_list[step - 2] @ y_list[step - 2]
        if step >= 1:
            b_step = b_step - block_diag_1_list[step - 1] @ y_list[step - 1]
        y_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step],
            b_step,
            upper=False,
            left=True,
        )

    # solve Lt X = Y, block backward substitution
    x_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps - 1, -1, -1):
        y_step: torch.Tensor = y_list[step]
        if step < n_steps - 2:
            y_step = y_step - block_diag_2_list[step].transpose(-2, -1) @ x_list[step + 2]
        if step < n_steps - 1:
            y_step = y_step - block_diag_1_list[step].transpose(-2, -1) @ x_list[step + 1]
        x_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step].transpose(-2, -1),
            y_step,
            upper=True,
            left=True,
        )

    u: torch.Tensor = torch.stack(x_list, dim=-3).reshape(*batch, n_steps, n_dims, n_orders)
    # (..., n_steps, n_dims, n_orders)
    return u


def ode_forward_reference(
        coefficients: torch.Tensor,
        rhs_equation: torch.Tensor,
        rhs_init: torch.Tensor,
        steps: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of the ODE forward pass.
    This implementation constructs the dense matrix A and has cubic time complexity and quadratic space complexity.

    Args:
        coefficients: (..., n_steps, n_equations, n_dims, n_orders)
        rhs_equation: (..., n_steps, n_equations)
        rhs_init: (..., n_init_var_steps, n_dims, n_init_var_orders)
        steps: (..., n_steps-1)

    Returns:
        out: (..., n_steps, n_dims, n_orders)
    """

    dtype: torch.dtype = coefficients.dtype
    device: torch.device = coefficients.device

    *batches, n_steps, n_equations, n_dims, n_orders = coefficients.shape
    *_, n_init_var_steps, _, n_init_var_orders = rhs_init.shape

    # ode equation constraints
    A_eq: torch.Tensor = torch.zeros(
        *batches, n_steps * n_equations, n_steps * n_dims * n_orders,
        dtype=dtype, device=device,
    )  # (..., n_steps * n_equations, n_steps * n_dims * n_orders)
    for i, (step, equation) in enumerate(itertools.product(range(n_steps), range(n_equations))):
        A_eq[..., i, step * n_dims * n_orders: (step + 1) * n_dims * n_orders] \
            = coefficients[..., step, equation, :, :].flatten(start_dim=-2)
    beta_eq: torch.Tensor = rhs_equation.flatten(start_dim=-2)  # (..., n_steps * n_equations)

    # initial-value constraints
    A_in: torch.Tensor = torch.zeros(
        *batches, n_init_var_steps * n_dims * n_init_var_orders, n_steps * n_dims * n_orders,
        dtype=dtype, device=device,
    )  # (..., n_init_var_steps * n_dims * n_init_var_orders, n_steps * n_dims * n_orders)
    for i, (step, dim, order) in enumerate(itertools.product(
            range(n_init_var_steps), range(n_dims), range(n_init_var_orders),
    )):
        A_in[..., i, (step * n_dims + dim) * n_orders + order] = 1.
    beta_in: torch.Tensor = rhs_init.flatten(start_dim=-3)  # (..., n_init_var_steps * n_dims * n_init_var_orders)

    # smoothness constraints (forward)
    A_sf: torch.Tensor = torch.zeros(
        *batches, (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders,
        dtype=dtype, device=device,
    )  # (..., (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders)
    for i, (step, dim, order) in enumerate(itertools.product(range(n_steps - 1), range(n_dims), range(n_orders - 1))):
        for o in range(order, n_orders):
            A_sf[..., i, (step * n_dims + dim) * n_orders + o] = steps[..., step] ** o / math.factorial(o - order)
        A_sf[..., i, ((step + 1) * n_dims + dim) * n_orders + order] = - steps[..., step] ** order

    # smoothness constraints (backward)
    A_sb: torch.Tensor = torch.zeros(
        *batches, (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders,
        dtype=dtype, device=device,
    )  # (..., (n_steps - 1) * n_dims * (n_orders - 1), n_steps * n_dims * n_orders)
    for i, (step, dim, order) in enumerate(itertools.product(range(n_steps - 1), range(n_dims), range(n_orders - 1))):
        for o in range(order, n_orders):
            A_sb[..., i, ((step + 1) * n_dims + dim) * n_orders + o] \
                = (- steps[..., step]) ** o / math.factorial(o - order)
        A_sb[..., i, (step * n_dims + dim) * n_orders + order] = - (- steps[..., step]) ** order

    # smoothness constraints (central)
    A_sc: torch.Tensor = torch.zeros(
        *batches, (n_steps - 2) * n_dims, n_steps * n_dims * n_orders,
        dtype=dtype, device=device,
    )  # (..., (n_steps - 2) * n_dims, n_steps * n_dims * n_orders)
    for i, (step, dim) in enumerate(itertools.product(range(n_steps - 2), range(n_dims))):
        A_sc[..., i, (step * n_dims + dim) * n_orders + (n_orders - 2)] \
            = (steps[..., step] + steps[..., step + 1]) ** (n_orders - 3)
        A_sc[..., i, ((step + 1) * n_dims + dim) * n_orders + (n_orders - 1)] \
            = (steps[..., step] + steps[..., step + 1]) ** (n_orders - 2)
        A_sc[..., i, ((step + 2) * n_dims + dim) * n_orders + (n_orders - 2)] \
            = - (steps[..., step] + steps[..., step + 1]) ** (n_orders - 3)

    A: torch.Tensor = torch.cat([A_eq, A_in, A_sb, A_sc, A_sf], dim=-2) # (..., m, n_steps * n_dims * n_orders)
    beta: torch.Tensor = torch.cat([
        beta_eq,
        beta_in,
        torch.zeros_like(A_sf[..., 0]),
        torch.zeros_like(A_sc[..., 0]),
        torch.zeros_like(A_sb[..., 0]),
    ], dim=-1)  # (..., m)

    AtA: torch.Tensor = A.transpose(-2, -1) @ A  # (..., n_steps * n_dims * n_orders, n_steps * n_dims * n_orders)
    Atb: torch.Tensor = A.transpose(-2, -1) @ beta[..., None]  # (..., n_steps * n_dims * n_orders, 1)

    L, _ = torch.linalg.cholesky_ex(AtA, upper=False, check_errors=False)
    # (..., n_steps * n_dims * n_orders, n_steps * n_dims * n_orders)
    u: torch.Tensor = Atb.cholesky_solve(L, upper=False)  # (..., n_steps * n_dims * n_orders, 1)
    u: torch.Tensor = u.reshape(*batches, n_steps, n_dims, n_orders)  # (..., n_steps, n_dims, n_orders)
    return u  # (..., n_steps, n_dims, n_orders)


def _unit_test() -> None:
    """
    Unit test for the ODE forward pass.
    """

    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device('cuda:0')
    batches: tuple = (11,)
    # batches: tuple = ()
    n_steps, n_equations, n_dims, n_orders = 7, 2, 3, 5
    n_init_var_steps, n_init_var_orders = 3, 4
    coefficients: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_steps, n_equations, n_dims, n_orders,
        dtype=dtype, device=device,
    ))
    rhs_equation: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_steps, n_equations,
        dtype=dtype, device=device,
    ))
    rhs_init: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_init_var_steps, n_dims, n_init_var_orders,
        dtype=dtype, device=device,
    ))
    steps: torch.Tensor = torch.nn.Parameter(torch.rand(*batches, n_steps - 1, dtype=dtype, device=device))
    _ = ode_forward(coefficients, rhs_equation, rhs_init, steps, enable_central_smoothness=True)

    coefficients: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_steps, n_equations, n_dims, n_orders,
        dtype=dtype, device=device,
    ))
    rhs_equation: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_steps, n_equations,
        dtype=dtype, device=device,
    ))
    rhs_init: torch.Tensor = torch.nn.Parameter(torch.randn(
        *batches, n_init_var_steps, n_dims, n_init_var_orders,
        dtype=dtype, device=device,
    ))
    steps: torch.Tensor = torch.nn.Parameter(torch.rand(*batches, n_steps - 1, dtype=dtype, device=device))
    u: torch.Tensor = ode_forward(coefficients, rhs_equation, rhs_init, steps, enable_central_smoothness=True)

    u.sum().backward()
    u0: torch.Tensor = ode_forward_reference(coefficients, rhs_equation, rhs_init, steps)
    diff: torch.Tensor = u - u0
    print(diff.abs().max().item())

    var_list: list[torch.Tensor] = [coefficients, rhs_equation, rhs_init, steps]
    grads: list[torch.Tensor] = [var.grad for var in var_list]
    for var in var_list:
        var.grad = None
    u0.sum().backward()
    grads0: list[torch.Tensor] = [var.grad for var in var_list]
    grads_diff: list[torch.Tensor] = [g - g0 for g, g0 in zip(grads, grads0)]
    print([grad_diff.abs().max().item() for grad_diff in grads_diff])

    print('Unit test done!')  # Random inputs may be unstable. Numerical errors are likely to appear.


if __name__ == '__main__':
    _unit_test()
