## To benchmark

### FP32
TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python gpt_transformer.py --fp8

### Mixed precision
TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_enable_cublaslt=true" python gpt_transformer.py --mixed --fp8

