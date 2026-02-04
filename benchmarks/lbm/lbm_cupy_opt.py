import cupy as cp
import numpy as np
from core.configs import LBMConfig


# CUDA Kernel for fused Collision and Streaming
def get_lbm_kernel(dtype_str):
    return r"""
extern "C" __global__
void lbm_step(const {0}* __restrict__ fin, 
              {0}* __restrict__ fout, 
              {0}* __restrict__ rho_out,
              {0}* __restrict__ ux_out,
              {0}* __restrict__ uy_out,
              const {0}* __restrict__ w, 
              const int* __restrict__ cx, 
              const int* __restrict__ cy, 
              {0} omega, 
              int nx, 
              int ny) {{
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    // Strides
    // fin shape: (9, nx, ny) -> stride i is nx*ny
    int stride_dir = nx * ny;
    int idx_base = x * ny + y; // Index in 2D grid

    // --- Macroscopic Moments ---
    {0} rho = 0.0;
    {0} ux = 0.0;
    {0} uy = 0.0;
    
    // Registers for local distribution to avoid re-reading
    {0} f[9];

    for (int i = 0; i < 9; i++) {{
        f[i] = fin[i * stride_dir + idx_base];
        rho += f[i];
        ux += f[i] * cx[i];
        uy += f[i] * cy[i];
    }}

    ux /= rho;
    uy /= rho;
    
    // Write macroscopic to global memory
    rho_out[idx_base] = rho;
    ux_out[idx_base] = ux;
    uy_out[idx_base] = uy;
    
    {0} u2 = ux * ux + uy * uy;

    // --- Collision & Streaming ---
    for (int i = 0; i < 9; i++) {{
        {0} cu = cx[i] * ux + cy[i] * uy;
        {0} feq = rho * w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        
        // Collision
        {0} f_post = f[i] - omega * (f[i] - feq);
        
        // Streaming: Write to neighbor
        // Target coordinates with periodicity
        int next_x = (x + cx[i] + nx) % nx;
        int next_y = (y + cy[i] + ny) % ny;
        
        int target_idx = i * stride_dir + (next_x * ny + next_y);
        
        fout[target_idx] = f_post;
    }}
}}
""".format(
        dtype_str
    )


class LBMCuPyOpt:
    def __init__(self, config: LBMConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.omega = config.omega

        # Determine types
        if config.precision == "f32":
            self.dtype = np.float32
            self.c_type = "float"
        else:
            self.dtype = np.float64
            self.c_type = "double"

        # Compile Kernel
        kernel_code = get_lbm_kernel(self.c_type)
        self.kernel = cp.RawKernel(kernel_code, "lbm_step")

        # Constants
        self.w_np = np.array(
            [
                config.w_0,
                config.w_1,
                config.w_1,
                config.w_1,
                config.w_1,
                config.w_2,
                config.w_2,
                config.w_2,
                config.w_2,
            ],
            dtype=self.dtype,
        )
        self.cx_np = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
        self.cy_np = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

        self.w = cp.asarray(self.w_np)
        self.cx = cp.asarray(self.cx_np)
        self.cy = cp.asarray(self.cy_np)

        # Data
        self.fin = cp.zeros((9, self.nx, self.ny), dtype=self.dtype)
        self.fout = cp.zeros((9, self.nx, self.ny), dtype=self.dtype)

        # Init
        np.random.seed(42)
        # h_rho = 1.0 + 0.01 * np.random.randn(self.nx, self.ny).astype(self.dtype)
        # self.rho = cp.asarray(h_rho, dtype=self.dtype)
        self.rho = self.init_guassian_pulse()
        self.ux = cp.zeros((self.nx, self.ny), dtype=self.dtype)
        self.uy = cp.zeros((self.nx, self.ny), dtype=self.dtype)

        # Initialization Step
        u2 = self.ux**2 + self.uy**2
        for i in range(9):
            cu = self.cx[i] * self.ux + self.cy[i] * self.uy
            self.fin[i, :, :] = (
                self.rho * self.w[i] * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u2)
            )

        # Block/Grid dims
        self.block_dim = (16, 16)
        self.grid_dim = (
            (self.nx + self.block_dim[0] - 1) // self.block_dim[0],
            (self.ny + self.block_dim[1] - 1) // self.block_dim[1],
        )

    def step(self):
        self.kernel(
            self.grid_dim,
            self.block_dim,
            (
                self.fin,
                self.fout,
                self.rho,
                self.ux,
                self.uy,
                self.w,
                self.cx,
                self.cy,
                self.dtype(self.omega),
                np.int32(self.nx),
                np.int32(self.ny),
            ),
        )

        # Swap
        self.fin, self.fout = self.fout, self.fin

        # Update macroscopic (Standardized state: time t+1)
        self.rho = cp.sum(self.fin, axis=0)
        self.ux = (
            cp.sum(self.fin * self.cx[:, cp.newaxis, cp.newaxis], axis=0) / self.rho
        )
        self.uy = (
            cp.sum(self.fin * self.cy[:, cp.newaxis, cp.newaxis], axis=0) / self.rho
        )

    def run(self):
        cp.cuda.Stream.null.synchronize()
        for _ in range(self.cfg.iterations):
            self.step()
        cp.cuda.Stream.null.synchronize()

    def init_guassian_pulse(self, fluct_density=1e-3):
        mshx = cp.linspace(-1, 1, self.nx, dtype=self.dtype)
        mshy = cp.linspace(-1, 1, self.ny, dtype=self.dtype)
        x, y = cp.meshgrid(mshx, mshy)
        rho = 1 + fluct_density * cp.exp(-100 * (x * x + y * y))
        return rho
