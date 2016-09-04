import tensorflow as tf

INPUT_WIDTH=256
INPUT_HEIGHT=144
INPUT_CHANNELS=3

global_step = None

# --------------------------------------------------------------------------------
# building each layer
# --------------------------------------------------------------------------------

def weight_variable(shape, dev=0.35, name=None):
    """create weight variable for conv2d(weight sharing)"""

    return tf.get_variable(name, shape,
        initializer=tf.truncated_normal_initializer(stddev=dev))

def bias_variable(shape, val=0.1, name=None):
    """create bias variable for conv2d(weight sharing)"""

    return tf.get_variable(name, shape, initializer=tf.constant_initializer(val))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def image_to_conv_layer(image, kernel_size, num_kernel):
    """ create layer convert image tensor to conved and maxpooled tensor

    Args:
        image: image tensor its shape is [-1, heigh, width, channel]

    Returns:
        maxpooled tensor
    """

    with tf.device('/cpu:0'):
        channels = image.get_shape()[3]
        w = weight_variable([kernel_size,kernel_size, channels,num_kernel], name="weight")
        b = bias_variable([num_kernel], name="bias")
    with tf.device('/gpu:0'):
        with tf.name_scope('conv_max'):
            h_conv = tf.nn.relu(conv2d(image, w) + b)
            h_pool = max_pool_2x2(h_conv)

    return h_pool

def conv_to_conv_layer(lower, kernel_size, num_kernel):
    """ create layer convert image tensor to conved and maxpooled tensor

    Args:
        lower: tensor its shape is [-1, height, width, channel]

    Returns:
        maxpooled tensor
    """

    with tf.device('/cpu:0'):
        channels = lower.get_shape()[3]
        w = weight_variable([kernel_size,kernel_size, channels,num_kernel], name="weight")
        b = bias_variable([num_kernel], name="bias")

    with tf.device('/gpu:0'):
        with tf.name_scope('conv_max'):
            h_conv = tf.nn.relu(conv2d(lower, w) + b)
            h_pool = max_pool_2x2(h_conv)

    return h_pool

def conv_to_fully_connected_layer(lower, dims):
    with tf.name_scope('reshape'):
        shape = lower.get_shape()
        reshaped_lower = tf.reshape(lower, [-1, int(shape[1]*shape[2]*shape[3])])
        w = weight_variable([shape[1]*shape[2]*shape[3], dims], name="weight")
        b = bias_variable([dims], name="bias")
    with tf.name_scope('softmax'):
        return tf.matmul(reshaped_lower, w) + b

def conv_to_conv_transpose_layer(lower, images):
    with tf.name_scope('deconvolution'), tf.device('/cpu:0'):
        shape = lower.get_shape()
        out_shape = images.get_shape()
        filter_var = tf.get_variable('filter',
            [5, 5, out_shape[3], shape[3]],
            initializer=tf.truncated_normal_initializer(stddev=0.35))
    with tf.name_scope('deconvolution'), tf.device('/gpu:0'):
        return tf.nn.conv2d_transpose(lower, filter_var, images.get_shape(), [1,2,2,1])

def conv_to_softmax_layer(lower, dims):
    with tf.name_scope('reshape'):
        shape = lower.get_shape()
        reshaped_lower = tf.reshape(lower, [-1, int(shape[1]*shape[2]*shape[3])])
        w = weight_variable([shape[1]*shape[2]*shape[3], dims], name="weight")
        b = bias_variable([dims], name="bias")
    with tf.name_scope('softmax'):
        return tf.nn.softmax(tf.matmul(reshaped_lower, w) + b)

# --------------------------------------------------------------------------------
# full network operation
# --------------------------------------------------------------------------------

def pre_train_loss(images, inferences):
    with tf.device('/gpu:0'):
        cross_entropy = tf.squared_difference(images,inferences)
    return cross_entropy

def pre_train_train(loss, val):
    with tf.device('/gpu:0'):
        global global_step
        global_step = tf.get_variable('global_step', (), initializer=tf.constant_initializer(0), trainable=False)
        train_op = tf.train.AdamOptimizer(val).minimize(loss, global_step=global_step)
    return train_op

def inference():
    pass

def loss():
    pass

def train():
    pass

# --------------------------------------------------------------------------------
# build network
# --------------------------------------------------------------------------------

def build_full_network(images,
    with_conv1_pre_train=False, with_conv2_pre_train=False,
    with_conv3_pre_train=False):

    result = []

    with tf.variable_scope('conv1'):
        conv1 = image_to_conv_layer(images, 5, 64)

        if with_conv1_pre_train:
            with tf.variable_scope('pre_train'):
                with tf.name_scope('inference'):
                    pt_inf = conv_to_conv_transpose_layer(conv1, images)
                image_summary = tf.image_summary('inference_image', pt_inf)
                with tf.name_scope('loss'):
                    pt_loss = pre_train_loss(images, pt_inf)
                    tf.scalar_summary('loss/conv1', tf.log(tf.reduce_max(pt_loss)))
                with tf.name_scope('train'):
                    pt_train = pre_train_train(pt_loss, 1e-4)
                    result = [pt_train] + result
    with tf.variable_scope('conv2'):
        conv2 = conv_to_conv_layer(conv1, 5, 128)

        if with_conv2_pre_train:
            with tf.variable_scope('pre_train'):
                with tf.name_scope('inference'):
                    pt_inf = conv_to_fully_connected_layer(
                        conv2, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS)
                with tf.name_scope('loss'):
                    pt_loss = pre_train_loss(images, pt_inf)
                    tf.scalar_summary('loss/conv2', tf.log(tf.reduce_max(pt_loss)))
                with tf.name_scope('train'), tf.device('/gpu:0'):
                    pt_train = pre_train_train(pt_loss, 1e-6)
                    result = [pt_train] + result
    with tf.variable_scope('conv3'):
        conv3 = conv_to_conv_layer(conv2, 5, 256)

        if with_conv3_pre_train:
            with tf.variable_scope('pre_train'):
                with tf.name_scope('inference'):
                    pt_inf = conv_to_fully_connected_layer(
                        conv3, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS)
                with tf.name_scope('loss'):
                    pt_loss = pre_train_loss(images, pt_inf)
                    tf.scalar_summary('loss/conv3', tf.log(tf.reduce_max(pt_loss)))
                with tf.name_scope('train'), tf.device('/gpu:0'):
                    pt_train = pre_train_train(pt_loss, 1e-6)
                    result = [pt_train] + result
    with tf.variable_scope('full_connection'):
        out_layer= conv_to_softmax_layer(conv3, 1024)
        with tf.name_scope('log'):
            tf.scalar_summary('mean/out', tf.reduce_max(out_layer))
        with tf.variable_scope('train'), tf.device('/gpu:0'):
            pass

    result = [out_layer] + result

    return tuple(result)

