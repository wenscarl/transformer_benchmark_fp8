import tensorflow as tf
import argparse
import time

from keras import layers
from keras import Model
from typing import Optional

dropout_rate = 0.0
model_size_scale = 1
from tensorflow.python.framework import dtypes
import DenseFp8 as dense_fp8
tf.config.experimental.enable_tensor_float_32_execution(False)
parser = argparse.ArgumentParser(description='config')
parser.add_argument('--fp8', action='store_true', help='use_fp8')
parser.add_argument('--mixed', action='store_true', help='mixed_precision')
parser.add_argument('--scale', type=int, help='model_scale', default=1)
args = parser.parse_args()

use_fp8 = args.fp8
# bmm always no use fp8
use_mixed = args.mixed
print("DEBUG: use_fp8", use_fp8)
print("DEBUG: use_mixed", use_mixed)

if use_mixed:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

oldDense = layers.Dense
if use_fp8:
    layers.Dense = dense_fp8.DenseFp8

init_val_scalar = 0.8

if use_fp8:
  FAKE_E4M3 = dtypes.float8_e4m3fn
  FAKE_E5M2 = dtypes.float8_e5m2

E4M3_MAX = 448.0
E5M2_MAX = 57344.0
AMAX_HIS_LEN = 16



class DotProductAttention(tf.keras.Model):
    """Attention operation in Transformer layer"""

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
    ):
        super().__init__()
        self.projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = float(kv_channels)
        self.norm_factor = tf.math.sqrt(self.hidden_size_per_attention_head)
        self.dropout = layers.Dropout(attention_dropout)
        if self.dropout.dtype_policy.name == "mixed_float16":
            self.norm_factor = tf.cast(self.norm_factor, dtype=tf.float16)
        init_val = tf.keras.initializers.Constant(init_val_scalar)
        self.input_amax_history = self.add_weight(
            "input_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.input_scale = self.add_weight(
            "input_scale", shape=(), initializer=init_val, trainable=False
        )
        self.kernel_amax_history = self.add_weight(
            "kernel_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.kernel_scale = self.add_weight(
            "kernel_scale", shape=(), initializer=init_val, trainable=False
        )
        self.input_grad_amax_history = self.add_weight(
            "input_grad_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.input_grad_scale = self.add_weight(
            "input_grad_scale", shape=(), initializer=init_val, trainable=False
        )
        self.kernel_grad_scale = self.add_weight(
            "kernel_grad_scale", shape=(), initializer=init_val, trainable=False
        )
        self.kernel_grad_amax_history = self.add_weight(
            "kernel_grad_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.output_grad_amax_history = self.add_weight(
            "output_grad_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.output_grad_scale = self.add_weight(
            "output_grad_scale", shape=(), initializer=init_val, trainable=False
        )
        self.output_amax_history = self.add_weight(
            "output_amax_history",
            shape=(AMAX_HIS_LEN,),
            initializer=init_val,
            trainable=False,
        )
        self.output_scale = self.add_weight(
            "output_scale", shape=(), initializer=init_val, trainable=False
        )

    def masked_softmax(self, inp: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor:
        if mask is not None:
            inp = tf.where(mask, -10000.0, inp)
        return tf.nn.softmax(inp, axis=-1)

    @tf.custom_gradient
    def in_qdq(self, input):
        """Quantize-dequantize both the input and the input's gradient."""
        qin = qdq_and_update(
            input, FAKE_E4M3, self.input_scale, self.input_amax_history
        )

        def grad(in_grad):
            in_grad_ret = qdq_and_update(
                in_grad, FAKE_E5M2, self.input_grad_scale, self.input_grad_amax_history
            )
            return in_grad_ret

        return qin, grad

    @tf.custom_gradient
    def out_qdq(self, output):
        """Quantize-dequantize both the output and the output's gradient, only if the next layer(in fwd sense) doesn't support fp8."""
        output = qdq_and_update(
            output, FAKE_E4M3, self.output_scale, self.output_amax_history
        )

        def grad(out_grad):
            return qdq_and_update(
                out_grad,
                FAKE_E5M2,
                self.output_grad_scale,
                self.output_grad_amax_history,
            )

        return output, grad

    @tf.custom_gradient
    def kernel_qdq(self, kernel):
        """Quantize-dequantize the kernel but not its gradient."""

        qkernel = qdq_and_update(
            kernel, FAKE_E4M3, self.kernel_scale, self.kernel_amax_history
        )

        def grad(kernel_grad):
            kernel_grad_ret = qdq_and_update(
                kernel_grad,
                FAKE_E5M2,
                self.kernel_grad_scale,
                self.kernel_grad_amax_history,
            )
            return kernel_grad_ret

        return qkernel, grad

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        b = query.shape[0]
        np = query.shape[2]
        sq = query.shape[1]
        sk = key.shape[1]
        hn = value.shape[3]

        # [b, sq, np, bn] -> [b, np, sq, bn]
        query = tf.transpose(query, perm=(0, 2, 1, 3))
        key = tf.transpose(key, perm=(0, 2, 3, 1))
        query = tf.reshape(query, shape=(b * np, sq, hn))
        key = tf.reshape(key, shape=(b * np, hn, sk))

        bmm1 = tf.matmul(query, key) / self.norm_factor

        # change view to [b, np, sq, sk]
        attention_scores = tf.reshape(bmm1, shape=(b, np, sq, sk))

        attention_probs = self.masked_softmax(attention_scores, None)  # attention_mask)

        attention_probs = self.dropout(attention_probs)

        # change view [sk, b * np, hn]
        # value = tf.reshape(value, shape=(sk, b * np, hn))
        value = tf.reshape(
            tf.transpose(value, perm=(0, 2, 1, 3)), shape=(b * np, sk, hn)
        )

        # change view [b * np, sq, sk]
        attention_probs = tf.reshape(attention_probs, shape=(b * np, sq, sk))

        context = tf.matmul(attention_probs, value)

        # change view to [b*np, sq, hn] - >[b, sq, np * hn]
        context = tf.reshape(context, shape=(b, np, sq, hn))
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        context = tf.reshape(context, shape=(b, sq, np * hn))
        return context


class BasicMLP(tf.keras.Model):
    """Feed-forward network in Transformer layer"""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
    ):
        super().__init__()
        if use_fp8:
            self.linear1 = layers.Dense(ffn_hidden_size, use_bias=True, is_last=True)
            self.linear2 = layers.Dense(hidden_size, use_bias=True, is_last=True)
        else:
            self.linear1 = oldDense(ffn_hidden_size, use_bias=True)
            self.linear2 = oldDense(hidden_size, use_bias=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.linear1(x)
        x = tf.nn.gelu(x, approximate=True)
        x = self.linear2(x)
        return x


class BasicTransformer(tf.keras.Model):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: int = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = layers.LayerNormalization(epsilon=layernorm_eps)
        if use_fp8:
            self.qkv_projection = layers.Dense(
                3 * hidden_size, use_bias=True, is_last=True
            )
        else:
            self.qkv_projection = oldDense(3 * hidden_size, use_bias=True)
        self.attention = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        if use_fp8:
            self.projection = layers.Dense(hidden_size, use_bias=True, is_last=True)
        else:
            self.projection = oldDense(hidden_size, use_bias=True)
        self.dropout = layers.Dropout(hidden_dropout)
        self.ln2 = layers.LayerNormalization(epsilon=layernorm_eps)
        self.mlp = BasicMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
        )

    def call(
        self,
        x: tf.Tensor,
        #      attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv_shape = qkv.shape
        qkv = tf.reshape(
            qkv,
            shape=(
                qkv_shape[0],
                qkv_shape[1],
                self.num_attention_heads,
                3 * self.kv_channels,
            ),
        )
        q, k, v = tf.split(qkv, 3, axis=3)

        attention_mask = None
        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        return x + res


# Layer configuration
hidden_size = 1024 * model_size_scale
sequence_length = 512
batch_size = 4
ffn_hidden_size = 1024 * model_size_scale
num_attention_heads = 32
dtype = tf.float32

opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.0001)
def speedometer(
    model: tf.keras.Model,
    input: tf.Tensor,
    y: tf.Tensor,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 20,
    warmup_iters: int = 20,
) -> None:
    """Measure average run time for a TF model

    Performs forward and backward passes.
    """
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}

    # There is a bug preventing model.fit in mixed precision
    if not use_mixed:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            jit_compile=True,
        )
        history = model.fit(
            input, y, batch_size=1, epochs=20, validation_split=0.0, verbose=1
        )
    else:
        p = tf.constant(0.0)  # Create small tensor to force GPU resync
        for v in model.variables:
          print("xxx variable name:", v.name)
        for v in model.trainable_variables:
          print("xxx trainable variable name:", v.name)

        @tf.function(jit_compile=True)
        def run_training(xx=input):
          with tf.GradientTape(persistent=True) as tape:
            tape.watch(xx)
            output = model(xx, **forward_kwargs)
            loss = tf.reduce_sum(output)
          dx, dvars = tape.gradient(loss, [xx, model.trainable_variables])
          #opt.apply_gradients(zip(dvars, model.trainable_variables))
          #xx = xx - dx
          return output, dx, dvars
        # Warmup runs
        for _ in range(warmup_iters):
          out, dx, dvars = run_training()
      
        print((p + 1.)) # Sync the GPU
      
        # Timing runs
        start = time.time()
        for _ in range(timing_iters):
          out, dx, dvars = run_training()
        print((p + 1.)) # Sync the GPU
        end = time.time()
      
        elapsed_time = (end - start) / timing_iters * 1000
      
        print(f"Mean time: {elapsed_time} ms")
        print("xxx check dx:")
        print(dx[0, 0], out.shape, input.shape)

tf.random.set_seed(12)
tf.keras.utils.set_random_seed(1)

# Synthetic data
x_data = tf.random.normal(shape=(batch_size, sequence_length, hidden_size), dtype=dtype)
y_data = tf.random.normal(shape=(batch_size, sequence_length, hidden_size*3), dtype=dtype)
y_data = tf.random.normal(x_data.shape)

basic_transformer = BasicTransformer(
    hidden_size,
    ffn_hidden_size,
    num_attention_heads,
    attention_dropout=dropout_rate,
    hidden_dropout=dropout_rate,
)

# Run once to build the variables and make sure the model.variables doesn't
# return None.
test_y = basic_transformer(x_data)
print(test_y)

speedometer(
    basic_transformer,
    x_data,
    y_data,
    forward_kwargs={"training": True},
)

print(basic_transformer.summary())

