import yaml_loader as yl
import tensorflow as tf

graph_root = yl.load_yaml('model.yaml')
pf = tf.placeholder(tf.float32, [3,64,64,3])
graph_root.build(pf)

for node in tf.all_variables():
    print(node.name)

