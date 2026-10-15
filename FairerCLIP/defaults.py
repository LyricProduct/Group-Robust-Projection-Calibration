from yacs.config import CfgNode as CN

_C = CN()

_C.device = "cuda"
_C.nolabel = False

_C.kernel_x = "RFFGaussian"
_C.kernel_x_options = CN()
_C.kernel_x_options.rff_dim = 1000
_C.kernel_x_options.sigma_numel_max = 4000

_C.kernel_s = "RFFGaussian"
_C.kernel_s_options = CN()

_C.kernel_z = "RFFGaussian"
_C.kernel_z_options = CN()
_C.kernel_z_options.rff_dim = 1000
_C.kernel_z_options.sigma_numel_max = 4000

_C.kernel_y = "RFFGaussian"
_C.kernel_y_options = CN()

_C.dim_z = 768

_C.seed = 2

_C.tau_i = 0.0
_C.tau_z_i = 0.0
_C.tau_t = 0.0
_C.tau_z_t = 0.0

_C.sample_ratio1 = 1.0
_C.sample_ratio2 = 1.0

_C.gamma_i = 3e-6
_C.gamma_t = 3e-6

_C.load_base_model = "clip_ViTL14"
_C.dataset = "waterbirds"

_C.iters = 1
