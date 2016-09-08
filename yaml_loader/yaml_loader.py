import yaml
import tensorflow as tf

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def weight_variable(shape, dev=0.35, name=None):
    """create weight variable for conv2d(weight sharing)"""

    return tf.get_variable(name, shape,
        initializer=tf.truncated_normal_initializer(stddev=dev))

def bias_variable(shape, val=0.1, name=None):
    """create bias variable for conv2d(weight sharing)"""

    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(val))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# ------------------------------------------------------------
# YAML Graph Nodes
# ------------------------------------------------------------

class EmptyWith:
    def __enter__(self):
        pass

    def __exit__(self,t,v,tb):
        pass

class GraphRoot(yaml.YAMLObject):
    yaml_tag = u'!graph_root'

    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return 'GraphRoot'

    def build(self, x):
        self.__nodes = {'root': x}
        for node in self.nodes:
            self.__nodes = node.build(self.__nodes)

    def get_nodes():
        return self.__nodes

class With(yaml.YAMLObject):
    yaml_tag = u'!with'

    def __init__(self, nodes, variable=None, name=None, device=None):
        self.nodes = nodes

        if not variable and not name and not device:
            raise ValueError('variable, name, device are all None')

        self.variable = variable
        self.name = name
        self.device = device

    def __repr__(self):
        return 'With'

    @classmethod
    def from_yaml(cls, loader, node):
        dict_representation = loader.construct_mapping(node)
        nodes = dict_representation['nodes']
        variable = dict_representation.get('variable', None)
        name = dict_representation.get('name', None)
        device = dict_representation.get('device', None)
        return cls(nodes, variable=variable, name=name, device=device)

    def build(self, nodes):
        vs = lambda x: tf.variable_scope(x) if x else EmptyWith()
        ns = lambda x: tf.name_scope(x) if x else EmptyWith()
        dv = lambda x: tf.device(x) if x else EmptyWith()

        with vs(self.variable), ns(self.name), dv(self.device):
            for node in self.nodes:
                nodes = node.build(nodes)

        return nodes

class Conv2d(yaml.YAMLObject):
    yaml_tag = u'!conv2d'

    def __init__(self, name, width, height, kernel_num, source, init=0.1):
        self.name = name
        self.width = width
        self.height = height
        self.kernel_num = kernel_num
        self.source = source
        self.init = init

    def __repr__(self):
        return 'Conv2d'

    def build(self, nodes):
        source_node = nodes[self.source]
        channels = source_node.get_shape()[3]
        print("{}: {}".format(source_node.name, source_node.get_shape()))
        w = weight_variable(
            [self.width,self.height, channels,self.kernel_num],
            name="weight")
        b = bias_variable([self.kernel_num], name="bias")

        nodes[self.name] = conv2d(source_node, w) + b
        return nodes

class Conv2dTranspose(yaml.YAMLObject):
    yaml_tag = u'!conv2d_transpose'

    def __init__(self, name, width, height, source, shape_source):
        self.name = name
        self.source = source
        self.shape_source = shape_source
        self.width = width
        self.height = height

    def __repr__(self):
        return 'Conv2dTranspose'

    def build(self, nodes):
        source_node = nodes[self.source]
        shape_source_node = nodes[self.shape_source]

        with tf.name_scope('deconvolution'), tf.device('/cpu:0'):
            shape = source_node.get_shape()
            out_shape = shape_source_node.get_shape()
            filter_var = tf.get_variable('filter',
                [self.height, self.width, out_shape[3], shape[3]],
                initializer=tf.truncated_normal_initializer(stddev=0.35))
        with tf.name_scope('deconvolution'), tf.device('/cpu:0'):
            nodes[self.name] = tf.nn.conv2d_transpose(
                source_node, filter_var,
                shape_source_node.get_shape(), [1,1,1,1])

        return nodes

class Conv2dAELoss(yaml.YAMLObject):
    yaml_tag = u'!conv2d_ae_loss'

    def __init__(self, name, source1, source2):
        self.name = name
        self.source1 = source1
        self.source2 = source2

    def __repr__(self):
        return 'Conv2dAELoss'

    def build(self, nodes):
        source1_node = nodes[self.source1]
        source2_node = nodes[self.source2]

        with tf.device('/cpu:0'):
            nodes[self.name] = tf.squared_difference(
                source1_node, source2_node)

        return nodes

class AdamOptimizer(yaml.YAMLObject):
    yaml_tag = u'!adam_optimizer'

    def __init__(self, name, source, val):
        self.name = name
        self.source = source
        self.val = val

    def __repr__(self):
        return 'AdamOptimizer'

    def build(self, nodes):
        source_node = nodes[self.source]

        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', (),
                initializer=tf.constant_initializer(0), trainable=False)
            nodes[self.name] = tf.train.AdamOptimizer(self.val).minimize(
                source_node, global_step=global_step)

        return nodes

class MaxPool2x2(yaml.YAMLObject):
    yaml_tag = u'!max_pool_2x2'

    def __init__(self, name, source):
        self.name = name
        self.source = source

    def __repr__(self):
        return 'Conv2d'

    def build(self, nodes):
        source_node = nodes[self.source]
        print("{}: {}".format(source_node.name, source_node.get_shape()))
        nodes[self.name] = tf.nn.max_pool(
            source_node, ksize=[1,2,2,1],
            strides=[1,2,2,1], padding='SAME')
        return nodes

# ------------------------------------------------------------
# Loader function
# ------------------------------------------------------------

def load_yaml(path):
    graph= yaml.load(open(str(path)).read())

    if type(graph['root']) is not GraphRoot:
        raise IOError("yaml file does'nt have GraphRoot")

    return graph['root']

