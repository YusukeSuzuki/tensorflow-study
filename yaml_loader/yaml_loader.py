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
            return ValueError('variable, name, device are all None')

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
        return cls(nodes, variable=variable)

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
        w = weight_variable(
            [self.width,self.height, channels,self.kernel_num],
            name="weight")
        b = bias_variable([self.kernel_num], name="bias")

        nodes[self.name] = conv2d(source_node, w) + b
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
        nodes[self.name] = tf.nn.max_pool(
            source_node, ksize=[1,2,2,1],
            strides=[1,2,2,1], padding='SAME')[3]
        return nodes

# ------------------------------------------------------------
# Loader function
# ------------------------------------------------------------

def load_yaml(path):
    graph= yaml.load(open(str(path)).read())

    if type(graph['root']) is not GraphRoot:
        raise IOError("yaml file does'nt have GraphRoot")

    return graph['root']

