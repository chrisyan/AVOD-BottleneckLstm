import tensorflow as tf

from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim

from avod.core.feature_extractors import lstm_cells
from avod.core.feature_extractors import rnn_decoder
from tensorflow.python.framework import ops as tf_ops


class BevVggPyr(bev_feature_extractor.BevFeatureExtractor):
    """Contains modified VGG model definition to extract features from
    Bird's eye view input using pyramid features.
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
            num_units=max(32, 32),
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

    def build(self,
              inputs,
              input_pixel_size,
              is_training,state_saver=None,state_name='lstm_state',
              scope='bev_vgg_pyr'):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        vgg_config = self.config
        #state_saver=None

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, 'bev_vgg_pyr', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Pad 700 to 704 to allow even divisions for max pooling
                    padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])

                    # Encoder
                    conv1 = slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')

                    batch_size = pyramid_fusion1.shape[0].value / 1
                    print('batch_size: ', batch_size)
                    
                    #with tf.variable_scope('LSTM', reuse=True) as lstm_scope:
                    lstm_cell, init_state1, _ = self.create_lstm_cell(batch_size, (pyramid_fusion1.shape[1].value, pyramid_fusion1.shape[2].value), state_saver,state_name)
                    net_seq1 = list(tf.split(pyramid_fusion1, 1))
                    print('net_seq1: ', net_seq1)
                    # Identities added for inputing state tensors externally.
                    c_ident = tf.identity(init_state1[0], name='lstm_state_in_c')
                    h_ident = tf.identity(init_state1[1], name='lstm_state_in_h')
                    init_state2 = (c_ident, h_ident)


                    #net_seq2, states_out = rnn_decoder.rnn_decoder(net_seq1, init_state2, lstm_cell,scope=lstm_scope)
                    net_seq2, states_out = rnn_decoder.rnn_decoder(net_seq1, init_state2, lstm_cell)


                    batcher_ops = None
                    # self._states_out = states_out
                    if state_saver is not None:
                        self._step = state_saver.state('%s_step' % state_name)
                        batcher_ops = [
                            state_saver.save_state('%s_c' % state_name, states_out[-1][0]),
                            state_saver.save_state('%s_h' % state_name, states_out[-1][1]),
                            state_saver.save_state('%s_step' % state_name, self._step + 1)
                        ]

                    with tf_ops.control_dependencies(batcher_ops):
                        pyramid_fusion1_after = tf.squeeze(net_seq2, axis=0)

                    # Identities added for reading output states, to be reused externally.
                    tf.identity(states_out[-1][0], name='lstm_state_out_c')
                    tf.identity(states_out[-1][1], name='lstm_state_out_h')

                    # Slice off padded area
                    sliced = pyramid_fusion1_after[:, 4:]

                feature_maps_out = sliced
                #print('end point',end_points_collection)

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points
