#include <pybind11/pybind11.h>
#include <torch/torch.h>


inline void bsubbmm(
    torch::Tensor& c,
    const torch::Tensor& a,
    const torch::Tensor& b
) {
    c -= a.matmul(b);
    // torch::baddbmm_out(
    //     /*out=*/c,
    //     /*input=*/c,
    //     /*batch1=*/a,
    //     /*batch2=*/b,
    //     /*beta=*/1,
    //     /*alpha=*/-1
    // );
}


void cholesky_inplace(
    torch::Tensor& block_diag_0,
    torch::Tensor& block_diag_1,
    torch::Tensor& block_diag_2,
    torch::Tensor& tmp_info,
    const bool enable_ldl,
    const bool enable_block_diag_2
) {
    torch::Tensor out;  // error: cannot bind non-const lvalue reference of type 'at::Tensor&' to an rvalue of type 'at::Tensor'
    auto n_steps = block_diag_0.size(0);
    for (auto step = 0; step < n_steps; ++step) {
        if (enable_block_diag_2 && step >= 2) {
            out = block_diag_2[step - 2];
            torch::linalg_solve_triangular_out(
                /*out=*/out,
                /*A=*/block_diag_0[step - 2].transpose(-2, -1),
                /*B=*/out,
                /*upper=*/true,
                /*left=*/false,
                /*unitriangular=*/false
            );
            out = block_diag_1[step - 1];
            bsubbmm(
                out,
                block_diag_2[step - 2],
                block_diag_1[step - 2].transpose(-2, -1)
            );
        }
        if (step >= 1) {
            out = block_diag_1[step - 1];
            torch::linalg_solve_triangular_out(
                /*out=*/out,
                /*A=*/block_diag_0[step - 1].transpose(-2, -1),
                /*B=*/out,
                /*upper=*/true,
                /*left=*/false,
                /*unitriangular=*/false
            );
            out = block_diag_0[step];
            if (enable_block_diag_2 && step >= 2) {
                bsubbmm(
                    out,
                    block_diag_2[step - 2],
                    block_diag_2[step - 2].transpose(-2, -1)
                );
            }
            bsubbmm(
                out,
                block_diag_1[step - 1],
                block_diag_1[step - 1].transpose(-2, -1)
            );
        }
        out = block_diag_0[step];
        torch::linalg_cholesky_ex_out(
            /*out_L=*/out,
            /*out_info=*/tmp_info,
            /*A=*/out,
            /*upper=*/false,
            /*check_errors=*/false
        );
    }
    if (enable_ldl) {
        torch::linalg_solve_triangular_out(
            /*out=*/block_diag_1,
            /*A=*/block_diag_0.slice(0, 0, -1, 1),
            /*B=*/block_diag_1,
            /*upper=*/false,
            /*left=*/false,
            /*unitriangular=*/false
        );
        if (enable_block_diag_2) {
            torch::linalg_solve_triangular_out(
                /*out=*/block_diag_2,
                /*A=*/block_diag_0.slice(0, 0, -2, 1),
                /*B=*/block_diag_2,
                /*upper=*/false,
                /*left=*/false,
                /*unitriangular=*/false
            );
        }
    }
}


void substitution_inplace(
    const torch::Tensor& block_diag_0,
    const torch::Tensor& block_diag_1,
    const torch::Tensor& block_diag_2,
    torch::Tensor& rhs,
    const bool enable_ldl,
    const bool enable_block_diag_2
) {
    torch::Tensor out;
    auto n_steps = block_diag_0.size(0);
    for (auto step = 0; step < n_steps; ++step) {
        out = rhs[step];
        if (enable_block_diag_2 && step >= 2) {
            bsubbmm(
                out,
                block_diag_2[step - 2],
                rhs[step - 2]
            );
        }
        if (step >= 1) {
            bsubbmm(
                out,
                block_diag_1[step - 1],
                rhs[step - 1]
            );
        }
        if (!enable_ldl) {
            torch::linalg_solve_triangular_out(
                /*out=*/out,
                /*A=*/block_diag_0[step],
                /*B=*/out,
                /*upper=*/false,
                /*left=*/true,
                /*unitriangular=*/false
            );
        }
    }
    if (enable_ldl) {
        // torch::cholesky_solve_out(
        //     /*out=*/rhs,
        //     /*B=*/rhs,
        //     /*L=*/block_diag_0,
        //     /*upper=*/false
        // );
        torch::linalg_solve_triangular_out(
            /*out=*/rhs,
            /*A=*/block_diag_0,
            /*B=*/rhs,
            /*upper=*/false,
            /*left=*/true,
            /*unitriangular=*/false
        );
        torch::linalg_solve_triangular_out(
            /*out=*/rhs,
            /*A=*/block_diag_0.transpose(-2, -1),
            /*B=*/rhs,
            /*upper=*/true,
            /*left=*/true,
            /*unitriangular=*/false
        );
    }
    for (auto step = n_steps - 1; step > -1; --step) {
        out = rhs[step];
        if (enable_block_diag_2 && step < n_steps - 2) {
            bsubbmm(
                out,
                block_diag_2[step].transpose(-2, -1),
                rhs[step + 2]
            );
        }
        if (step < n_steps - 1) {
            bsubbmm(
                out,
                block_diag_1[step].transpose(-2, -1),
                rhs[step + 1]
            );
        }
        if (!enable_ldl) {
            torch::linalg_solve_triangular_out(
                /*out=*/out,
                /*A=*/block_diag_0[step].transpose(-2, -1),
                /*B=*/out,
                /*upper=*/true,
                /*left=*/true,
                /*unitriangular=*/false
            );
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cholesky_inplace", &cholesky_inplace, "some description.");
  m.def("substitution_inplace", &substitution_inplace, "some description.");
}
