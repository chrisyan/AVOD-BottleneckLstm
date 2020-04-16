import tensorflow as tf

from avod.core.feature_extractors import img_feature_extractor
#from research.lstm_object_detection.lstm import lstm_cells
#from /home/chris/models-master/research/lstm_object_detection.lstm import rnn_decoder
#from tensorflow.python.framework import ops as tf_ops
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


from tensorflow.contrib import slim as contrib_slim
from tensorflow.python.framework import ops as tf_ops

from tensorflow.keras import layers

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


#import pylab as plt

slim = tf.contrib.slim

#FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_integer('batch_size', 16,)


class ImgVggPyr(img_feature_extractor.ImgFeatureExtractor):
    """Modified VGG model definition to extract features from
    RGB image input using pyramid features.
    """

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def build(self,
              inputs,
              input_pixel_size,
              is_training,
         
              scope='img_vgg_pyr'):
        """ Modified VGG for image feature extraction with pyramid features.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config

        def my_convlstm_model(frames, channels, pixels_x, pixels_y, categories):
            trailer_input = Input(shape=(frames,channels,pixels_x,pixels_y), name = 'trailer_input')
            convlstm_layer = ConvLSTM2D(filters=20, kenel_size=(3,3),recurrent_activation='hard_sigmoid', activation='tanh', padding='same', return_sequences=True)(trailer_input)
            batch_normalization_layer = BatchNormalization()(convlstm_layer)
            pooling_layer = MaxPooling2D(pool_size=(1,2,2),padding='same')(batch_normalization_layer)
            return pooling_layer

        def convLSTM(input, hidden, filters, kernel, scope):

            with tf.variable_scope(scope, initializer= tf.truncated_normal_initializer(stddev=0.1)):
                cell = BasicConvLSTMCell.BasicConvLSTMCell([input.get_shape()[1], input.get_shape()[2]], kernel, filters)

                if hidden is None:
                    hidden = cell.zero_state(input.get_shape()[0], tf.float32)
                y_, hidden = cell(input, hidden)
            return y_,hidden




        def create_lstm_cell(self, batch_size, output_size, state_saver, state_name):
            """Create the LSTM cell, and initialize state if necessary.
            Args:
              batch_size: input batch size.
              output_size: output size of the lstm cell, [width, height].
              state_saver: a state saver object with methods `state` and `save_state`.
              state_name: string, the name to use with the state_saver.
            Returns:
              lstm_cell: the lstm cell unit.
              init_state: initial state representations.
              step: the step
            """
            lstm_cell = lstm_cells.BottleneckConvLSTMCell(
                filter_size=(3, 3),
                output_size=output_size,
                num_units=max(self._min_depth, self._lstm_state_depth),
                activation=tf.nn.relu6,
                visualize_gates=False)

            if state_saver is None:
                init_state = lstm_cell.init_state(state_name, batch_size, tf.float32)
                step = None
            else:
                step = state_saver.state(state_name + '_step')
                c = state_saver.state(state_name + '_c')
                h = state_saver.state(state_name + '_h')
                init_state = (c, h)
            return lstm_cell, init_state, step

        with slim.arg_scope(self.vgg_arg_scope(weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'img_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d,slim.fully_connected,  slim.max_pool2d,slim.separable_conv2d, slim.dropout],  outputs_collections=end_points_collection):
                    # Encoder
                    conv1 = slim.repeat(inputs,vgg_config.vgg_conv1[0],slim.separable_conv2d, vgg_config.vgg_conv1[1], [3, 3],normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}, depth_multiplier=1, scope='conv1')
                    #cnv1b, hidden1 = convLSTM(conv1, hidden_state[0], 32, [3,3], scope='cnv1_lstm')
                    drop1 = slim.dropout(conv1, keep_prob=0.5, is_training=True, scope='dropout1')
                    pool1 = slim.max_pool2d(drop1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,vgg_config.vgg_conv2[0], slim.separable_conv2d, vgg_config.vgg_conv2[1],[3, 3], normalizer_fn=slim.batch_norm, normalizer_params={ 'is_training': is_training}, depth_multiplier=1,scope='conv2')
                    #cnv2b, hidden2 = convLSTM(conv2, hidden_state[1], 64, [3,3], scope='cnv2_lstm')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,vgg_config.vgg_conv3[0], slim.separable_conv2d, vgg_config.vgg_conv3[1],  [3, 3],normalizer_fn=slim.batch_norm,normalizer_params={ 'is_training': is_training},depth_multiplier=1,scope='conv3')
                    #cnv3b, hidden2 = convLSTM(conv3, hidden_state[2], 128, [3,3], scope='cnv3_lstm')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,vgg_config.vgg_conv4[0], slim.separable_conv2d, vgg_config.vgg_conv4[1],[3, 3],normalizer_fn=slim.batch_norm,normalizer_params={ 'is_training': is_training},  depth_multiplier=1,scope='conv4')
                    print('conv4 vgg_config0',vgg_config.vgg_conv4[0])
                    print('conv4 vgg_config1',vgg_config.vgg_conv4[1])
                    


                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(conv4,vgg_config.vgg_conv3[1], [3, 3], stride=2,normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}, scope='upconv3')

                    concat3 = tf.concat((conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.separable_conv2d(concat3,vgg_config.vgg_conv2[1],[3, 3], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training},depth_multiplier=1,scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose( pyramid_fusion3,vgg_config.vgg_conv2[1], [3, 3], stride=2, normalizer_fn=slim.batch_norm,normalizer_params={'is_training': is_training}, scope='upconv2')

                    concat2 = tf.concat((conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.separable_conv2d( concat2,vgg_config.vgg_conv1[1], [3, 3], normalizer_fn=slim.batch_norm, normalizer_params={  'is_training': is_training},depth_multiplier=1,scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(pyramid_fusion_2,vgg_config.vgg_conv1[1],[3, 3],stride=2, normalizer_fn=slim.batch_norm,normalizer_params={ 'is_training': is_training}, scope='upconv1')

                    concat1 = tf.concat((conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.separable_conv2d(concat1,  vgg_config.vgg_conv1[1], [3, 3], normalizer_fn=slim.batch_norm, normalizer_params={ 'is_training': is_training},depth_multiplier=1,scope='pyramid_fusion1')

                    
                    print('pyramid_fusion1: ', pyramid_fusion1.shape)
                    print('pyramid_fusion1: ', pyramid_fusion1)
                    # print('pyramid_fusion1: ', pyramid_fusion1.shape[0])
                    # print('pyramid_fusion1: ', pyramid_fusion1.shape[1])
                    # print('pyramid_fusion1: ', pyramid_fusion1.shape[2])
                    # print('pyramid_fusion1: ', pyramid_fusion1.shape[3])
                    
                    #ConvLSTM layer
                    #last_layer= slim.LSTM(pyramid_fusion1,  vgg_config.vgg_conv1[1], [3, 3], normalizer_fn=slim.batch_norm, normalizer_params={ 'is_training': is_training},depth_multiplier=1,scope='pyramid_fusion1')
                    #pyramid_fusion1_after= tf.squeeze(pyramid_fusion1, axis=0).shape
                    # lstm_layer_flatten = layers.Flatten( ) \
                    #     (pyramid_fusion1)
                    #pyramid_fusion1_after = tf.squeeze(pyramid_fusion1)
                    #lstm_layer_after1 = tf.expand_dims(pyramid_fusion1, 0) 
                    #print('lstm_layer_after1: ',lstm_layer_after1)
                    
                    #last_layer = tf.keras.layers.ConvLSTM2D(filters=32,kernel_size=(3,3), return_sequences=True, padding='same', data_format='channels_last') \
                    #    (lstm_layer_after1)
                    #pyramid_fusion1_after = tf.squeeze(pyramid_fusion1)
                    # lstm_reshape = tf.reshape(lstm_layer_flatten,(1,1000,13824))
                    # #pyramid_fusion1_after = tf.squeeze(lstm_reshape)
                    # 
                    # 
                    # lstm_layer = layers.LSTM(units=32,
                    #                           activation="relu", dropout=0.1,
                    #                           recurrent_dropout=0.1,
                    #                           return_sequences=True ) \
                    #      (lstm_reshape)
                    # lstm_layer_after2 = tf.expand_dims(lstm_layer, 0) 
                    # print('lstm_layer: ',lstm_layer_after2)

                    #last_layer = my_convlstm_model(pyramid_fusion1.shape[0],pyramid_fusion1.shape[2],pyramid_fusion1.shape[3],pyramid_fusion1.shape[1],None)

                    last_layer = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',input_shape=(None,360,1200,32),data_format='channels_last')(pyramid_fusion1)

                    #print('last layer: ',last_layer)

                feature_maps_out = pyramid_fusion1
                #print('conv00: ', conv00)

                
                print('feature map out is: ', feature_maps_out)



                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points

                #print('feature map out is: ', feature_maps_out)
                #print('end points: ', end_points)

