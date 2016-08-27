from argparse import ArgumentParser as AP
import tensorflow as tf
import numpy as np
import abies_model as am

ROOT_VARIABLE_SCOPE='abiesnet'

# --------------------------------------------------------------------------------
# sub command methods
# --------------------------------------------------------------------------------

def do_train(namespace):
    # build
    images = tf.placeholder(tf.float32, [1,64,36,3])
    arr = np.ones([1,64,36,3], dtype=np.float)

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        out = am.build_full_network(images)

    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # ready to run

    sess = tf.Session()
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)

    sess.run(tf.initialize_all_variables())

    # run

    summary, res = sess.run( (merged, out), feed_dict={images:  arr} )
    print(type(res))
    print(np.argmax(res))
    writer.add_summary(summary, 0)
    writer.add_graph(tf.get_default_graph())
    saver.save(sess, namespace.modelfile)


    # finalize

    writer.close()

def do_test(namespace):
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

def do_eval(namespace):
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

# --------------------------------------------------------------------------------
# command line option parser
# --------------------------------------------------------------------------------

def create_parser():
    parser = AP(prog='aluns')
    parser.set_defaults(func=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--modelfile', type=str, default='model')
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser('train')
    sub_parser.set_defaults(func=do_train)
    sub_parser = sub_parsers.add_parser('test')
    sub_parser.set_defaults(func=do_test)
    sub_parser = sub_parsers.add_parser('eval')
    sub_parser.set_defaults(func=do_eval)

    return parser

# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

def run():
    parser = create_parser()
    namespace = parser.parse_args()

    if namespace.func:
        namespace.func(namespace)
    else:
        parser.print_help()

if __name__ == '__main__':
    run()

