from argparse import ArgumentParser as AP
from pathlib import Path
import tensorflow as tf
import numpy as np
import abies_model as am
import image_loader as il

ROOT_VARIABLE_SCOPE='abiesnet'
MODELS_DIR='models'

loader_param = {
    'flip_up_down': True,
    'flip_left_right': True,
    'random_brightness': True,
    'random_contrast': True
    }

# --------------------------------------------------------------------------------
# sub command methods
# --------------------------------------------------------------------------------

def do_train(namespace):
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir/namespace.modelfile
    model_backup_path = models_dir/(namespace.modelfile+'.back')

    # build
    reader = tf.WholeFileReader()
    with tf.variable_scope('image_loader'), tf.device('/cpu:0'):
        print('read samples directory')
        samples_dir = Path(namespace.samples)
        samples = []

        if samples_dir.is_dir():
            samples = [str(p.resolve())
                for p in samples_dir.iterdir() if p.suffix == '.jpg']
        else:
            samples = [l.rstrip() for l in  samples_dir.open().readlines()]

        batch_images = il.build_full_network(
            samples, am.INPUT_HEIGHT, am.INPUT_WIDTH, am.INPUT_CHANNELS, 16, reader,
            **loader_param)
        image_summary = tf.image_summary('input_image', batch_images)

    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        print('build network')
        out, train1 = am.build_full_network(batch_images,
            with_conv1_pre_train=True)

    saver = tf.train.Saver()
    merged = tf.merge_all_summaries()

    # ready to run

    print('initialize')
    sess = tf.Session()
    writer = tf.train.SummaryWriter(namespace.logdir, sess.graph)
    sess.run(tf.initialize_all_variables())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # run

    if namespace.restore:
        print('restore {}'.format(namespace.restore))
        saver.restore(sess, namespace.restore)

    print('train')
    writer.add_graph(tf.get_default_graph())

    for i in range(0, 100000):
        print('loop: {}'.format(i))
        summary, res, step = sess.run( (merged, train1, am.global_step), feed_dict={} )
        writer.add_summary(summary, step)

        if i % 1000 == 1:
            print('save backup to: {}'.format(model_backup_path))
            saver.save(sess, str(model_backup_path))

    print('save to: {}'.format(model_path))
    saver.save(sess, str(model_path))

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
    parser.add_argument('--modelfile', type=str, default='model.ckpt')
    parser.add_argument('--samples', type=str, default='./samples')
    parser.add_argument('--restore', type=str, default='')
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

