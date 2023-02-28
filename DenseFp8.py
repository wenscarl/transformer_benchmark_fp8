# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the Dense layer."""


import tensorflow.compat.v2 as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec

# isort: off
from tensorflow.python.util.tf_export import keras_export

from tensorflow.python.framework import dtypes
FAKE_E4M3 = dtypes.float8_e4m3fn
FAKE_E5M2 = dtypes.float8_e5m2

E4M3_MAX = 448.
E5M2_MAX = 57344.
AMAX_HIS_LEN = 16


def get_fp8_max(fake_dtype):
  if fake_dtype == FAKE_E4M3:
    return E4M3_MAX
  else:
    assert fake_dtype == FAKE_E5M2
    return E5M2_MAX


def quantize(x, quantized_dtype, scale):
  dtype_max = get_fp8_max(quantized_dtype)
  policy = tf.keras.mixed_precision.global_policy()
  is_mixed_policy = (
        policy is not None and policy.compute_dtype != policy.variable_dtype
  )

  if is_mixed_policy:
    scaled_x = tf.clip_by_value(x / tf.cast(scale, tf.float16), -dtype_max, dtype_max)
  else:
    scaled_x = tf.clip_by_value(x / scale, -dtype_max, dtype_max)
  return tf.cast(scaled_x, quantized_dtype)


def dequantize(x, wide_dtype, scale):
  return tf.cast(x, wide_dtype) * tf.cast(scale, wide_dtype)

def quantize_dequantize(x, quantized_dtype, scale):
  orig_dtype = x.dtype
  qx = quantize(x, quantized_dtype, scale)
  return dequantize(qx, orig_dtype, scale)

def update_scale(x, quantized_dtype, scale_var, amax_history):
  dtype_max = get_fp8_max(quantized_dtype)
  amax_current = tf.cast(tf.math.reduce_max(tf.math.abs(x)), scale_var.dtype)
  amax_his_tsr = tf.tensor_scatter_nd_update(tf.roll(amax_history.read_value(), 1, 0),[[0]],[amax_current])
  amax_history.assign(tf.cast(amax_his_tsr, amax_history.dtype))
  amax_temp = tf.reduce_max(amax_history, axis=0)
  amax = tf.maximum(amax_temp, 2 ** -10)
  scale_var.assign(tf.cast(1.1 * amax / dtype_max,scale_var.dtype ))

def qdq_and_update(x, dtype, scale_var, amax_history):
  qx = quantize_dequantize(x, dtype, scale_var)
  update_scale(x, dtype, scale_var, amax_history)
  return qx

@keras_export("keras.layers.Dense")
class DenseFp8(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). These are all attributes of
    `Dense`.

    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 0 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`, then we create
    a `kernel` with shape `(d1, units)`, and the `kernel` operates along axis 2
    of the `input`, on every sub-tensor of shape `(1, 1, d1)` (there are
    `batch_size * d0` such sub-tensors).  The output in this case will have
    shape `(batch_size, d0, units)`.

    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    When a popular kwarg `input_shape` is passed, then keras will create
    an input layer to insert before the current layer. This can be treated
    equivalent to explicitly defining an `InputLayer`.

    Example:

    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    @utils.allow_initializer_layout
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        is_last=False,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        policy = tf.keras.mixed_precision.global_policy()
        self.is_mixed_policy = (
          policy is not None and policy.compute_dtype != policy.variable_dtype
        )
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.is_last=is_last

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        init32 = tf.keras.initializers.Constant(0.9)
        self.input_amax_history = self.add_weight(
            "input_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init32, trainable=False)
        self.input_scale = self.add_weight("input_scale", shape=(),
                                           initializer=init32, trainable=False)
        self.kernel_amax_history = self.add_weight(
            "kernel_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init32, trainable=False)
        self.kernel_scale = self.add_weight("kernel_scale", shape=(),
                                            initializer=init32, trainable=False)
        self.input_grad_amax_history = self.add_weight(
            "input_grad_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init32, trainable=False)
        self.input_grad_scale = self.add_weight("input_grad_scale", shape=(),
                                                initializer=init32,
                                                trainable=False)
        self.output_grad_amax_history = self.add_weight(
            "output_grad_amax_history", shape=(AMAX_HIS_LEN,),
            initializer=init32, trainable=False)
        self.output_grad_scale = self.add_weight(
            "output_grad_scale", shape=(),
            initializer=init32, trainable=False)
        self.built = True

    @tf.custom_gradient
    def in_qdq(self, input):
      """Quantize-dequantize both the input and the input's gradient."""
      qin = qdq_and_update(input, FAKE_E4M3, self.input_scale, self.input_amax_history)

      def grad(in_grad):
        in_grad_ret = qdq_and_update(in_grad, FAKE_E5M2, self.input_grad_scale, 
                                     self.input_grad_amax_history)
        return in_grad_ret
  
      return qin, grad
  
    @tf.custom_gradient
    def out_qdq(self, output):
      """Quantize-dequantize both the output and the output's gradient, only if the next layer(in fwd sense) doesn't support fp8."""
#      output = qdq_and_update(output, FAKE_E4M3, self.output_scale, self.output_amax_history)
      def grad(out_grad):
        return qdq_and_update(
            out_grad, FAKE_E5M2, self.output_grad_scale, self.
            output_grad_amax_history)
      return output, grad
  
    @tf.custom_gradient
    def kernel_qdq(self, kernel):
      """Quantize-dequantize the kernel but not its gradient."""

      qkernel  = qdq_and_update(kernel, FAKE_E4M3, self.kernel_scale, 
                                self.kernel_amax_history)
  
      if self.is_mixed_policy:
          qkernel = tf.cast(qkernel, tf.float16)
      def grad(kernel_grad, variables=None):
        return kernel_grad
  
      return qkernel, grad

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            # In case we encounter a RaggedTensor with a fixed last dimension
            # (last dimension not ragged), we can flatten the input and restore
            # the ragged dimensions at the end.
            if tf.compat.dimension_value(inputs.shape[-1]) is None:
                raise ValueError(
                    "Dense layer only supports RaggedTensors when the "
                    "innermost dimension is non-ragged. Received: "
                    f"inputs.shape={inputs.shape}."
                )
            original_inputs = inputs
            if inputs.flat_values.shape.rank > 1:
                inputs = inputs.flat_values
            else:
                # Innermost partition is encoded using uniform_row_length.
                # (This is unusual, but we can handle it.)
                if inputs.shape.rank == 2:
                    inputs = inputs.to_tensor()
                    is_ragged = False
                else:
                    for _ in range(original_inputs.ragged_rank - 1):
                        inputs = inputs.values
                    inputs = inputs.to_tensor()
                    original_inputs = tf.RaggedTensor.from_nested_row_splits(
                        inputs, original_inputs.nested_row_splits[:-1]
                    )

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul
            # operation for large sparse input tensors. The op will result in a
            # sparse gradient, as opposed to
            # sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id
                # per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding
                # lookup as a matrix multiply. We split our input matrix into
                # separate ids and weights tensors. The values of the ids tensor
                # should be the column indices of our input matrix and the
                # values of the weights tensor can continue to the actual matrix
                # weights.  The column arrangement of ids and weights will be
                # summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed
                # explanation of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape,
                )
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    self.kernel, ids, weights, combiner="sum"
                )
            else:
                outputs = tf.matmul(a=self.in_qdq(inputs), b=self.kernel_qdq(self.kernel))
                if self.is_last:
                  outputs = self.out_qdq(outputs)
        # Broadcast kernel to inputs.
        else:
        #    outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            outputs = tf.matmul(a=self.in_qdq(inputs), b=self.kernel_qdq(self.kernel))
            if self.is_last:
              outputs = self.out_qdq(outputs)
# Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        if is_ragged:
            outputs = original_inputs.with_flat_values(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
