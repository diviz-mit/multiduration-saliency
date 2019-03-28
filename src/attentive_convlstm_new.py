# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
#
# from tensorflow.python.keras import activations
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras import constraints
# from tensorflow.python.keras import initializers
# from tensorflow.python.keras import regularizers
# from tensorflow.python.keras.engine.base_layer import InputSpec
# from tensorflow.python.keras.engine.base_layer import Layer
# from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask
# from tensorflow.python.keras.layers.recurrent import _standardize_args
# from tensorflow.python.keras.layers.recurrent import RNN
# from tensorflow.python.keras.utils import conv_utils
# from tensorflow.python.keras.utils import generic_utils
# from tensorflow.python.keras.utils import tf_utils
# from tensorflow.python.util.tf_export import tf_export
# from keras.layers import ConvRNN2D
#
#
#



# This should go right above the ConvLSTM2D from the original file (so below all this stuff)
# -*- coding: utf-8 -*-
"""Convolutional-recurrent layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.layers.recurrent import _standardize_args

import numpy as np
import warnings
from keras.engine.base_layer import InputSpec, Layer
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.legacy.layers import Recurrent, ConvRecurrent2D
from keras.layers.recurrent import RNN
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import transpose_shape
from keras.layers import ConvRNN2D










class AttentiveConvLSTM2DCell(Layer):

      def __init__(self,
                   filters,
                   attentive_filters,
                   kernel_size,
                   attentive_kernel_size,
                   strides=(1, 1),
                   padding='valid',
                   data_format=None,
                   dilation_rate=(1, 1),
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   attentive_activation='tanh',
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   attentive_initializer='zeros',
                   bias_initializer='zeros',
                   unit_forget_bias=True,
                   kernel_regularizer=None,
                   recurrent_regularizer=None,
                   attentive_regularizer=None,
                   bias_regularizer=None,
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   attentive_constraint=None,
                   bias_constraint=None,
                   dropout=0.,
                   recurrent_dropout=0.,
                   attentive_dropout=0.,
                   **kwargs):
        super(AttentiveConvLSTM2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.attentive_filters = attentive_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.attentive_kernel_size = conv_utils.normalize_tuple(attentive_kernel_size, 2, 'attentive_kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attentive_activation = activations.get(attentive_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attentive_initializer = initializers.get(attentive_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attentive_regularizer = regularizers.get(attentive_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attentive_constraint = constraints.get(attentive_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.attentive_dropout = min(1., max(0., attentive_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self._attentive_dropout_mask = None

        # print('INIT CALLED')

      def build(self, input_shape):
        print('BUILD CALLED')

        if self.data_format == 'channels_first':
          channel_axis = 1
        else:
          channel_axis = -1
        if input_shape[channel_axis] is None:
          raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)

        self.kernel_shape = kernel_shape
        print('kernel_shape', kernel_shape)
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)
        input_attentive_kernel_shape = self.attentive_kernel_size + (input_dim, self.attentive_filters)
        hidden_attentive_kernel_shape = self.attentive_kernel_size + (self.filters, self.attentive_filters)
        squeeze_attentive_kernel_shape =  self.attentive_kernel_size + (self.attentive_filters, 1)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.input_attentive_kernel = self.add_weight(
            shape=input_attentive_kernel_shape,
            initializer=self.attentive_initializer,
            name='input_attentive_kernel',
            regularizer=self.attentive_regularizer,
            constraint=self.attentive_constraint)

        self.hidden_attentive_kernel = self.add_weight(
            shape=hidden_attentive_kernel_shape,
            initializer=self.attentive_initializer,
            name='hidden_attentive_kernel',
            regularizer=self.attentive_regularizer,
            constraint=self.attentive_constraint)

        self.squeeze_attentive_kernel = self.add_weight(
            shape=squeeze_attentive_kernel_shape,
            initializer=self.attentive_initializer,
            name='squeeze_attentive_kernel',
            regularizer=self.attentive_regularizer,
            constraint=self.attentive_constraint)


        if self.use_bias:
          if self.unit_forget_bias:

            def bias_initializer(_, *args, **kwargs):
              return K.concatenate([
                  self.bias_initializer((self.filters,), *args, **kwargs),
                  initializers.Ones()((self.filters,), *args, **kwargs),
                  self.bias_initializer((self.filters * 2,), *args, **kwargs),
              ])
          else:
            bias_initializer = self.bias_initializer
          self.bias = self.add_weight(
              shape=(self.filters * 4,),
              name='bias',
              initializer=bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)

          self.attentive_bias = self.add_weight(
              shape=(self.attentive_filters * 2,),
              name='attentive_bias',
              initializer=bias_initializer,
              regularizer=self.bias_regularizer,
              constraint=self.bias_constraint)

        else:
          self.bias = None

        self.kernel_i = self.kernel[:, :, :, :self.filters]
        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
        self.kernel_f = self.kernel[:, :, :, self.filters: self.filters * 2]
        self.recurrent_kernel_f = (self.recurrent_kernel[:, :, :, self.filters:
                                                        self.filters * 2])
        self.kernel_c = self.kernel[:, :, :, self.filters * 2: self.filters * 3]
        self.recurrent_kernel_c = (self.recurrent_kernel[:, :, :, self.filters * 2:
                                                        self.filters * 3])
        self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

        if self.use_bias:
          self.bias_i = self.bias[:self.filters]
          self.bias_f = self.bias[self.filters: self.filters * 2]
          self.bias_c = self.bias[self.filters * 2: self.filters * 3]
          self.bias_o = self.bias[self.filters * 3:]
          self.bias_wa = self.attentive_bias[:self.attentive_filters ]
          self.bias_ua = self.attentive_bias[self.attentive_filters : self.attentive_filters * 2]
        else:
          self.bias_i = None
          self.bias_f = None
          self.bias_c = None
          self.bias_o = None

        self.built = True

      def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
          self._dropout_mask = _generate_dropout_mask(
              K.ones_like(inputs),
              self.dropout,
              training=training,
              count=4)
        if (0 < self.recurrent_dropout < 1 and
            self._recurrent_dropout_mask is None):
          self._recurrent_dropout_mask = _generate_dropout_mask(
              K.ones_like(states[1]),
              self.recurrent_dropout,
              training=training,
              count=4)
        # if (0 < self.attentive_dropout < 1 and self._attentive_dropout_mask is None):
        #   self._attentive_dropout_mask = _generate_dropout_mask(
        #       K.ones_like(inputs),
        #       self.attentive_dropout,
        #       training=training,
        #       count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask
        # dropout matrices for attentive units
        # att_dp_mask = self._attentive_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state


        ##### ATTENTION MECHANISM

        h_and_x = self.input_conv(h_tm1, self.hidden_attentive_kernel, self.bias_ua, padding='same') + self.input_conv(inputs, self.input_attentive_kernel, self.bias_wa, padding='same')

        # shape of h_and_x: (batch, row, col, attentive_filters)
        # print('h_and_x.shape',h_and_x.shape)

        e = self.recurrent_conv(self.attentive_activation(h_and_x), self.squeeze_attentive_kernel)

        # print('e.shape',e.shape)
        # print('K.shape(e)',K.shape(e))
        a = K.reshape(K.softmax(K.batch_flatten(e)), K.shape(e)) #(None, K.shape(e)[].shape[1]), int(e.shape[2]), 1))
        inputs = inputs * K.repeat_elements(a, inputs.shape[-1], -1)


        ##### END OF ATTENTION MECHANISM


        if 0 < self.dropout < 1.:
          inputs_i = inputs * dp_mask[0]
          inputs_f = inputs * dp_mask[1]
          inputs_c = inputs * dp_mask[2]
          inputs_o = inputs * dp_mask[3]
        else:
          inputs_i = inputs
          inputs_f = inputs
          inputs_c = inputs
          inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
          h_tm1_i = h_tm1 * rec_dp_mask[0]
          h_tm1_f = h_tm1 * rec_dp_mask[1]
          h_tm1_c = h_tm1 * rec_dp_mask[2]
          h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
          h_tm1_i = h_tm1
          h_tm1_f = h_tm1
          h_tm1_c = h_tm1
          h_tm1_o = h_tm1


        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i,
                              padding=self.padding)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f,
                              padding=self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c,
                              padding=self.padding)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o,
                              padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i,
                                  self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o,
                                  self.recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
          if training is None:
            h._uses_learning_phase = True

        return h, [h, c]

      def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
          conv_out = K.bias_add(conv_out, b,
                                data_format=self.data_format)
        return conv_out

      def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

      def get_config(self):
        config = {'filters': self.filters,
                  'attentive_filters': self.attentive_filters,
                  'kernel_size': self.kernel_size,
                  'attentive_kernel_size': self.attentive_kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'attentive_activation': activations.serialize(
                      self.attentive_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'attentive_initializer': initializers.serialize(
                      self.attentive_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'attentive_regularizer': regularizers.serialize(
                      self.attentive_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'attentive_constraint': constraints.serialize(
                      self.attentive_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'attentive_dropout': self.attentive_dropout}
        base_config = super(AttentiveConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentiveConvLSTM2D(ConvRNN2D):
  """Convolutional LSTM.
  It is similar to an LSTM layer, but the input transformations
  and recurrent transformations are both convolutional.
  Arguments:
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
        dimensions of the convolution window.
    strides: An integer or tuple/list of n integers,
        specifying the strides of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, time, ..., channels)`
        while `channels_first` corresponds to
        inputs with shape `(batch, time, channels, ...)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
        for the recurrent step.
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
        If True, add 1 to the bias of the forget gate at initialization.
        Use in combination with `bias_initializer="zeros"`.
        This is recommended in [Jozefowicz et al.]
        (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to.
    kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
    go_backwards: Boolean (default False).
        If True, process the input sequence backwards.
    stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
    dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
  Input shape:
    - if data_format='channels_first'
        5D tensor with shape:
        `(samples, time, channels, rows, cols)`
    - if data_format='channels_last'
        5D tensor with shape:
        `(samples, time, rows, cols, channels)`
  Output shape:
    - if `return_sequences`
         - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, filters, output_row, output_col)`
         - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, output_row, output_col, filters)`
    - else
        - if data_format ='channels_first'
            4D tensor with shape:
            `(samples, filters, output_row, output_col)`
        - if data_format='channels_last'
            4D tensor with shape:
            `(samples, output_row, output_col, filters)`
        where o_row and o_col depend on the shape of the filter and
        the padding
  Raises:
    ValueError: in case of invalid constructor arguments.
  References:
    - [Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
    The current implementation does not include the feedback loop on the
    cells output.
  """

  def __init__(self,
               filters,
               attentive_filters,
               kernel_size,
               attentive_kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               attentive_activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               attentive_initializer='zeros',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               attentive_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               attentive_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               dropout=0.,
               recurrent_dropout=0.,
               attentive_dropout=0.,
               **kwargs):
    cell = AttentiveConvLSTM2DCell(filters=filters,
                          attentive_filters=attentive_filters,
                          kernel_size=kernel_size,
                          attentive_kernel_size=attentive_kernel_size,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          dilation_rate=dilation_rate,
                          activation=activation,
                          recurrent_activation=recurrent_activation,
                          attentive_activation=attentive_activation,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          recurrent_initializer=recurrent_initializer,
                          attentive_initializer=attentive_initializer,
                          bias_initializer=bias_initializer,
                          unit_forget_bias=unit_forget_bias,
                          kernel_regularizer=kernel_regularizer,
                          recurrent_regularizer=recurrent_regularizer,
                          attentive_regularizer=attentive_regularizer,
                          bias_regularizer=bias_regularizer,
                          kernel_constraint=kernel_constraint,
                          recurrent_constraint=recurrent_constraint,
                          attentive_constraint=attentive_constraint,
                          bias_constraint=bias_constraint,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          attentive_dropout=attentive_dropout)

    super(AttentiveConvLSTM2D, self).__init__(cell,
                                     return_sequences=return_sequences,
                                     go_backwards=go_backwards,
                                     stateful=stateful,
                                     **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    return super(AttentiveConvLSTM2D, self).call(inputs,
                                        mask=mask,
                                        training=training,
                                        initial_state=initial_state)

  @property
  def filters(self):
    return self.cell.filters

  @property
  def attentive_filters(self):
    return self.cell.attentive_filters

  @property
  def kernel_size(self):
    return self.cell.kernel_size

  @property
  def attentive_kernel_size(self):
    return self.cell.attentive_kernel_size

  @property
  def strides(self):
    return self.cell.strides

  @property
  def padding(self):
    return self.cell.padding

  @property
  def data_format(self):
    return self.cell.data_format

  @property
  def dilation_rate(self):
    return self.cell.dilation_rate

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def attentive_activation(self):
    return self.cell.attentive_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def attentive_initializer(self):
    return self.cell.attentive_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def attentive_regularizer(self):
    return self.cell.attentive_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def attentive_constraint(self):
    return self.cell.attentive_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def attentive_dropout(self):
    return self.cell.attentive_dropout

  def get_config(self):
    config = {'filters': self.filters,
              'attentive_filters': self.attentive_filters,
              'kernel_size': self.kernel_size,
              'attentive_kernel_size': self.attentive_kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'attentive_activation': activations.serialize(
                  self.attentive_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'attentive_initializer': initializers.serialize(
                  self.attentive_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'attentive_regularizer': regularizers.serialize(
                  self.attentive_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'activity_regularizer': regularizers.serialize(
                  self.activity_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'attentive_constraint': constraints.serialize(
                  self.attentive_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout,
              'attentive_dropout': self.attentive_dropout}
    base_config = super(AttentiveConvLSTM2D, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)




#
# #### OLD ####
#
#
# # def oldAttentiveConvLSTM(x,h, nb_filters_out, nb_filters_att, nb_rows, nb_cols,
# #                      init='normal', inner_init='orthogonal', attentive_init='zero',
# #                      activation='tanh', inner_activation='sigmoid',
# #                      W_regularizer=None, U_regularizer=None,
# #                      weights=None, go_backwards=False):
# #
# #     # Original filters:
# #     # nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,nb_cols=3, nb_rows=3
# #
# #     #### ATTENTION MODULE #####
# #     x1 = Conv2D(nb_filters_att,
# #                 (nb_rows, nb_cols),
# #                 padding='same',
# #                 activation=None,
# #                 kernel_initializer=init)(x) # Wa
# #
# #     x2 = Conv2D(nb_filters_att,
# #                 (nb_rows, nb_cols),
# #                 padding='same',
# #                 activation=None,
# #                 kernel_initializer=init)(h) # Ua
# #
# #     x3 = Add()([x1,x2]) # Wa*X + Ua*H_(t-1)
# #     x3 = K.tanh(x3)
# #     (1, self.nb_rows, self.nb_cols, border_mode='same', bias=False, init=self.attentive_init)
# #     z = Conv2D(1,
# #                 (nb_rows, nb_cols),
# #                 padding='same',
# #                 activation='softmax',
# #                 bias=False,
# #                 kernel_initializer=attentive_init)(x3)  # Va * tanh(x3)
# #     # a = K.softmax(z) # Softmax(Z)
# #     x_m = Multiply()([z,x])  # Elementwise product
# #     # This x_m should now be used as input to the ConvLSTM
# #
# #     ##### END OF ATTENTION MODULE #####
# #
# #     convLSTM = ConvLSTM2D(nb_filters_out, (nb_rows, nb_cols), strides=(1, 1), padding='valid',
# #     data_format=None, dilation_rate=(1, 1), activation=activation, recurrent_activation='hard_sigmoid',
# #     use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
# #     bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
# #     recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
# #     kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
# #     return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
# #
# #     x = convLSTM(TimeDistibuted(x_m))
# #
# #     # Returns convLSTM output
# #     return x
#
#
#
# class OldAttentiveConvLSTM(Layer):
#     def __init__(self, nb_filters_in, nb_filters_out, nb_filters_att, nb_rows, nb_cols,
#                  init='normal', inner_init='orthogonal', attentive_init='zero',
#                  activation='tanh', inner_activation='sigmoid',
#                  W_regularizer=None, U_regularizer=None,
#                  weights=None, go_backwards=False,
#                  **kwargs):
#         self.nb_filters_in = nb_filters_in
#         self.nb_filters_out = nb_filters_out
#         self.nb_filters_att = nb_filters_att
#         self.nb_rows = nb_rows
#         self.nb_cols = nb_cols
#         self.init = initializations.get(init)
#         self.inner_init = initializations.get(inner_init)
#         self.attentive_init = initializations.get(attentive_init)
#         self.activation = activations.get(activation)
#         self.inner_activation = activations.get(inner_activation)
#         self.initial_weights = weights
#         self.go_backwards = go_backwards
#
#         self.W_regularizer = W_regularizer
#         self.U_regularizer = U_regularizer
#         self.input_spec = [InputSpec(ndim=5)]
#
#         super(AttentiveConvLSTM, self).__init__(**kwargs)
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape[:1] + (self.nb_filters_out,) + input_shape[3:]
#
#     def compute_mask(self, input, mask):
#         return None
#
#     def get_initial_states(self, x):
#         initial_state = K.sum(x, axis=1)
#         initial_state = K.conv2d(initial_state, K.zeros((self.nb_filters_out, self.nb_filters_in, 1, 1)), border_mode='same')
#         initial_states = [initial_state for _ in range(len(self.states))]
#
#         return initial_states
#
#     def build(self, input_shape):
#         self.input_spec = [InputSpec(shape=input_shape)]
#         self.states = [None, None]
#         self.trainable_weights = []
#
#         self.W_a = Convolution2D(self.nb_filters_att, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.U_a = Convolution2D(self.nb_filters_att, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.V_a = Convolution2D(1, self.nb_rows, self.nb_cols, border_mode='same', bias=False, init=self.attentive_init)
#
#         self.W_a.build((input_shape[0], self.nb_filters_att, input_shape[3], input_shape[4]))
#         self.U_a.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
#         self.V_a.build((input_shape[0], self.nb_filters_att, input_shape[3], input_shape[4]))
#
#         self.W_a.built = True
#         self.U_a.built = True
#         self.V_a.built = True
#
#         self.W_i = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.U_i = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)
#
#         self.W_i.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
#         self.U_i.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
#
#         self.W_i.built = True
#         self.U_i.built = True
#
#         self.W_f = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.U_f = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)
#
#         self.W_f.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
#         self.U_f.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
#
#         self.W_f.built = True
#         self.U_f.built = True
#
#         self.W_c = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.U_c = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)
#
#         self.W_c.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
#         self.U_c.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
#
#         self.W_c.built = True
#         self.U_c.built = True
#
#         self.W_o = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
#         self.U_o = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)
#
#         self.W_o.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
#         self.U_o.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
#
#         self.W_o.built = True
#         self.U_o.built = True
#
#         self.trainable_weights = []
#         self.trainable_weights.extend(self.W_a.trainable_weights)
#         self.trainable_weights.extend(self.U_a.trainable_weights)
#         self.trainable_weights.extend(self.V_a.trainable_weights)
#         self.trainable_weights.extend(self.W_i.trainable_weights)
#         self.trainable_weights.extend(self.U_i.trainable_weights)
#         self.trainable_weights.extend(self.W_f.trainable_weights)
#         self.trainable_weights.extend(self.U_f.trainable_weights)
#         self.trainable_weights.extend(self.W_c.trainable_weights)
#         self.trainable_weights.extend(self.U_c.trainable_weights)
#         self.trainable_weights.extend(self.W_o.trainable_weights)
#         self.trainable_weights.extend(self.U_o.trainable_weights)
#
#     def preprocess_input(self, x):
#         return x
#
#     def step(self, x, states):
#         x_shape = K.shape(x)
#         h_tm1 = states[0]
#         c_tm1 = states[1]
#
#         e = self.V_a(K.tanh(self.W_a(h_tm1) + self.U_a(x)))
#         a = K.reshape(K.softmax(K.batch_flatten(e)), (x_shape[0], 1, x_shape[2], x_shape[3]))
#         x_tilde = x * K.repeat_elements(a, x_shape[1], 1)
#
#         x_i = self.W_i(x_tilde)
#         x_f = self.W_f(x_tilde)
#         x_c = self.W_c(x_tilde)
#         x_o = self.W_o(x_tilde)
#
#         i = self.inner_activation(x_i + self.U_i(h_tm1))
#         f = self.inner_activation(x_f + self.U_f(h_tm1))
#         c = f * c_tm1 + i * self.activation(x_c + self.U_c(h_tm1))
#         o = self.inner_activation(x_o + self.U_o(h_tm1))
#
#         h = o * self.activation(c)
#         return h, [h, c]
#
#     def get_constants(self, x):
#         return []
#
#     def call(self, x, mask=None):
#         input_shape = self.input_spec[0].shape
#         initial_states = self.get_initial_states(x)
#         constants = self.get_constants(x)
#         preprocessed_input = self.preprocess_input(x)
#
#         last_output, outputs, states = K.rnn(self.step, preprocessed_input,
#                                              initial_states,
#                                              go_backwards=False,
#                                              mask=mask,
#                                              constants=constants,
#                                              unroll=False,
#                                              input_length=input_shape[1])
#
#         if last_output.ndim == 3:
#             last_output = K.expand_dims(last_output, dim=0)
#
#         return last_output
#
#
#
