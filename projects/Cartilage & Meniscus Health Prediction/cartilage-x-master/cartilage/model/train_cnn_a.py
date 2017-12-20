# # Training a 3D CNN for MRI data
# Things to change/check before running:
# * Data directory (make sure data is there)
# * Image shape definition
# 
# Possible approach -- try optimizing the model wrt the following (in order):
# * Increase number of training steps to 1000-10000 (101 steps is a placeholder)
# * Change regularization for conv and FC layers to optimize
# * Vary dropout (conv and FC layers)
# * Change sizes of layers
# * After model is 'optimal' for equal sample weighting change whether the loss function is weighted or not
# * After getting best model for current amount of layers, change architecture

# ## Import and pre-process data
# 
# ### Set up directories

# Importing required packages
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from six.moves import cPickle as pickle  # useful package for saving and loading all sorts of datatype

# Define data root
data_root = '/data/bigbone4/ciriondo/'
#data_root = os.getcwd()
pickle_root = os.path.join(data_root, 'pickle_data_3')

train_dir = os.path.join(pickle_root, 'train')
valid_dir = os.path.join(pickle_root, 'valid')

# Create list of directories to loop through for each dataset:
train_filenames = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
valid_filenames = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir)]

print('Training samples: {}'.format(len(train_filenames)))
print('Valid samples: {}'.format(len(valid_filenames)))


# ### Use the TF Dataset API to create iterators

# **Function to split labels and convert to one-hot encoding:**
# 
# Based on categorization. In this case (processed in the previous script):
# * WORMS in {0, 1}: Y = 0
# * WORMS in {2, 3, 4, 5}: Y = 1
# FIXME?: potentially inefficient
def reformat_labels(label, bin_limits=[2]):
    """Convert labels to one-hot encoding"""
#     num_labels = y_batch.max() + 1
    label = np.array([label], dtype=np.float32)
    num_labels = 2
    label = np.digitize(label, bins=[2])
    label = (np.arange(num_labels) == label[:, None]).astype(np.float32)[0]
    return label


# **Functions that will create TF iterator objects from the file lists:**
def _pickle_load(filename):
    """Load each image from the filename"""
    with open(filename, 'rb') as f:
        save = pickle.load(f)
        image = save['image'].astype(np.float32)
        label = np.float32(save['label'])
        label = reformat_labels(label)
    return image, label


def _reshape_function(image, label):
    """Add channel dimension to each image"""
#     image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    return image, label


def make_dataset(filenames, name='datasets', batch_size=8, repeat=False):
    """Create TensorFlow Dataset objects from filenames"""
#     filenames = tf.placeholder(tf.string, shape=[None])
    with tf.variable_scope(name) as scope:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(
            lambda filename: tuple(tf.py_func(_pickle_load, 
                                              [filename], 
                                              [np.float32, np.float32])))
        dataset = dataset.map(_reshape_function)
        dataset = dataset.shuffle(buffer_size=1000)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    
with tf.name_scope('inputs') as scope:
    # create TF dataset objects using function defined above
    batch_size = 4
    train_dataset = make_dataset(
        train_filenames, name='train_dataset', batch_size=batch_size, repeat=True)
    valid_dataset = make_dataset(
        valid_filenames, name='valid_dataset', batch_size=batch_size)
    
    # used to evaluate full training dataset
    eval_train_dataset = make_dataset(
        train_filenames, name='eval_train_dataset', batch_size=batch_size)

    # feedable iterator defined by handle placeholder
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()
    # x and y can then be used generically in all our functions
    x_batch, y_batch = next_element

    # set up iterators
    # training dataset loops forever, others run once and need to be initialized each time
    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()
    eval_train_iterator = eval_train_dataset.make_initializable_iterator()


# ## Build CNN
# **Model architecture:**
# # Neatness functions:
# def variable_summaries(var):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)


# convolution layer
def conv_relu_layer(
        x, k, in_depth, out_depth, 
        name, 
        dropout_keep_prob=1.0, regularizer=None):
    """
    x: batch of input volumes with shape=[batch, in_depth, in_height, in_width, in_channels]
    k: size of kernel in all axes
    depth: number of filters
    name: name for layer tensor
    
    returns: output of 3D convolution
    """
    with tf.variable_scope(name):
        # set stddev to ensure variance of activation = 1
        stddev = np.sqrt(2.0/np.power(k, 3))
        
        # create filter weight and bias variables
        w = tf.get_variable(name='weight', dtype=tf.float32,
                           initializer=tf.truncated_normal([k, k, k, in_depth, out_depth], 
                                                           stddev=stddev),
                           regularizer=regularizer)
#         variable_summaries(w)
        
        b = tf.get_variable(name='bias', dtype=tf.float32,
                           initializer=tf.constant(0.0, shape=[out_depth]))
#         variable_summaries(b)

        # output convolution - stride of 1 and same padding fixed
        conv = tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')
        conv += b
        
        tf.summary.histogram('preactivations', conv)
        
        conv = tf.nn.relu(conv)
        tf.summary.histogram('activations', conv)
        
        # output acitvation function 
        return tf.nn.dropout(conv, dropout_keep_prob)


# pooling layer
def maxpool_layer(x, name, stride=[2, 2, 2]):
    """
    x: batch of input volumes with shape=[batch, in_depth, in_height, in_width, in_channels]
    k: size of stride in each dimension 
    name: name for layer tensor
    
    returns: max pooled convolution
    """
    # TODO: see whether it's worth keeping this scope
    with tf.variable_scope(name):
        return tf.nn.max_pool3d(x, ksize=[1, stride[0], stride[1], stride[2], 1], 
                                strides=[1, stride[0], stride[1], stride[2], 1],
                                padding='SAME')


# batch normalization for convolutions
def conv_batch_norm(conv, depth, name):
    """Return batch normalized output for convolutional layer (3D)"""
    with tf.variable_scope(name) as scope:
        epsilon = 1e-3
        mean, var = tf.nn.moments(conv, axes=[0, 1, 2, 3])
        scale = tf.get_variable(name='scale', dtype=tf.float32, initializer=tf.ones([depth]))
        beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=tf.zeros([depth]))
        conv = tf.nn.batch_normalization(conv, mean, var, beta, scale, epsilon)
        tf.summary.histogram('batch_norm', conv)
        return conv


# for convenience, also assumes SAME padding
def conv_to_fc_size(
    input_shape, conv_depth, pools,
    stride=[2, 2, 2],  padding='SAME',
    dropout_keep_prob=1.0):
    """Return convoution height, width, depth dimensions after a number of pool layers"""
    h, w, d = input_shape
    if padding == 'SAME':
        for i in range(pools):
            h = math.ceil(float(h) / float(stride[0]))
            w = math.ceil(float(w) / float(stride[1]))
            d = math.ceil(float(d) / float(stride[2]))            
    else:
        # 'VALID' padding
        pass
    
    return conv_depth * h * w * d


# fully connected layer
def fc_layer(
        x, in_n, out_n, name, 
        activation=None, regularizer=None, dropout_keep_prob=1.0, conv_input=False):
    """
    x: batch of input features or volumes
    out_n: number of output nodes
    name: name for layer tensor
    from_conv: True if the input layer is a convolution and needs to be reshaped
    """
    with tf.variable_scope(name):        
        # reshape x if it is the output of a conv layer
        if conv_input:
            x = tf.reshape(x, shape=[-1, in_n])
        
        # set stddev to ensure variance of activation = 1
        stddev = tf.sqrt(tf.divide(2.0, tf.cast(in_n, tf.float32)))
        
        # create filter weight and bias variables
        w = tf.get_variable(name='weight', dtype=tf.float32,
                           initializer=tf.truncated_normal([in_n, out_n], stddev=stddev),
                           regularizer=regularizer)
#         variable_summaries(w)
        
        b = tf.get_variable(name='bias', dtype=tf.float32,
                           initializer=tf.constant(0.0, shape=[out_n]))        
#         variable_summaries(b)
        
        # output convolution - stride of 1 and same padding fixed
        out = tf.matmul(x, w) + b
        tf.summary.histogram('preactivations', out)
               
        # output acitvation function 
        if activation is not None:
            out = activation(out)
        
        tf.summary.histogram('activations', out)
        
        return tf.nn.dropout(out, dropout_keep_prob)


def dense_batch_norm(x, out_n, name):
    with tf.variable_scope(name) as scope:
        epsilon = 1e-3
        mean, var = tf.nn.moments(x, axes=[0])
        scale = tf.get_variable(name='scale', dtype=tf.float32, initializer=tf.ones([out_n]))
        beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=tf.zeros([out_n]))
        x = tf.nn.batch_normalization(x, mean, var, beta, scale, epsilon)
        tf.summary.histogram('batch_norm', x)
        return x


# Making model:
# TODO/note: may have some redundancy in scoping

### define all layers
# function to seperate models with and without dropout
def model(name, conv_dropout_keep_prob=1.0, fc_dropout_keep_prob=1.0, reuse=None):
    """Create tensor for logits with optional dropout"""
    with tf.variable_scope(name, reuse=reuse) as scope:
        # input format
        input_shape = [100, 100, 60]  # can't actually get rid of this 
        num_labels = 2

        # architecture
        k = 5
        depth_1 = 32
        depth_2 = 32
        
        k = 3
        depth_3 = 64
        depth_4 = 64
        
        fc_num_0 = conv_to_fc_size(input_shape, conv_depth=depth_4, pools=2)
        fc_num_1 = 32

        # regularizers
        reg_conv = tf.contrib.layers.l2_regularizer(scale=1e-6)
        reg_fc = tf.contrib.layers.l2_regularizer(scale=1e-6)

        # 2 convolution and pooling layers
        conv_1 = conv_relu_layer(x_batch,
                                 5, 1, depth_1,
                                 regularizer=reg_conv,
                                 dropout_keep_prob=conv_dropout_keep_prob,
                                 name='conv_1')
#         conv_1 = maxpool_layer(conv_1, name='maxpool_1')

        conv_2 = conv_relu_layer(conv_1,
                                 5, depth_1, depth_2,
                                 regularizer=reg_conv,
                                 dropout_keep_prob=conv_dropout_keep_prob,
                                 name='conv_2')
        conv_2 = maxpool_layer(conv_2, name='maxpool_2')

        conv_3 = conv_relu_layer(conv_2,
                                 3, depth_2, depth_3,
                                 regularizer=reg_conv,
                                 dropout_keep_prob=conv_dropout_keep_prob,
                                 name='conv_3')

        conv_4 = conv_relu_layer(conv_3,
                                 3, depth_3, depth_4,
                                 regularizer=reg_conv,
                                 dropout_keep_prob=conv_dropout_keep_prob,
                                 name='conv_4')
        
        conv_4 = maxpool_layer(conv_4, name='maxpool_2')

        # 1 fully connected layer
        fc_1 = fc_layer(conv_4, 
                        fc_num_0, fc_num_1, 
                        conv_input=True, 
                        activation=tf.nn.relu,
                        regularizer=reg_fc,
                        dropout_keep_prob=fc_dropout_keep_prob,
                        name='fc_1')

        # output
        logits = fc_layer(fc_1,
                          fc_num_1, num_labels, 
                          regularizer=reg_fc,
                          dropout_keep_prob=fc_dropout_keep_prob,
                          name='out')

        return logits
    
# model with dropout for training
logits_drop = model(name='cnn',
                    conv_dropout_keep_prob=0.75,
                    fc_dropout_keep_prob=0.5)

# model without dropout for evaluation
# using the same name scope means the weights defined above are shared
logits = model(name='cnn',
               reuse=True)
    
# **Loss and optimization:**
# loss
with tf.name_scope('loss') as scope:
    # add regularization from layers created above
    l2_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    # define loss function (for convenience)
    # 0: cross-entropy without weighting
    # 1: cross-entropy with sample weighting
    # 2: sigmoid cross-entropy with class weighting
    weight_samples = 2
    
    # loss function
    if weight_samples == 0:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_batch, logits=logits_drop) + tf.add_n(l2_losses))
        tf.summary.scalar('loss', loss)
    elif weight_samples == 1:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y_batch, logits=logits_drop) + tf.add_n(l2_losses))
        tf.summary.scalar('loss', loss)
    elif weight_samples == 2:
        loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                    targets=y_batch, logits=logits_drop, pos_weight=10.0) + tf.add_n(l2_losses))
        tf.summary.scalar('sigmoid_weighted_loss', loss)

    
# optimizer
with tf.name_scope('train'):
    train_rate = 1e-4
    optimizer = tf.train.AdamOptimizer(train_rate).minimize(loss)


# **Predictions and accuracy:**
def create_reset_metric(metric, name='reset_metrics', **metric_args):
    """Create operations to accomodate streaming metrics"""
    with tf.variable_scope(name) as scope:
        metric_op, update_op = metric(**metric_args)
        running_vars = tf.contrib.framework.get_variables(
                     scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(running_vars)
        tf.summary.scalar(name, metric_op)
        return metric_op, update_op, reset_op
 

# model evaluation
with tf.variable_scope('eval') as scope:

    # non-dropout model for prediction
    prediction = tf.nn.softmax(logits)

    # training (batch) accuracy
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_batch, 1))
    batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # accuracy for multiple batches
    accuracy, accuracy_update, accuracy_reset = create_reset_metric(
        metric=tf.metrics.accuracy,
        name='stream_accuracy', 
        labels=tf.argmax(y_batch, -1),
        predictions=tf.argmax(prediction, -1))
    
    # recall for multiple batches
    recall, recall_update, recall_reset = create_reset_metric(
        metric=tf.contrib.metrics.streaming_recall,
        name='stream_recall', 
        labels=tf.argmax(y_batch, -1),
        predictions=tf.argmax(prediction, -1))
    
    # precision for multiple batches
    precision, precision_update, precision_reset = create_reset_metric(
        metric=tf.contrib.metrics.streaming_precision,
        name='stream_precision', 
        labels=tf.argmax(y_batch, -1),
        predictions=tf.argmax(prediction, -1))

    # TEMPORARY (or not): group together operations for convenience
    reset_ops = [accuracy_reset, recall_reset, precision_reset]
    update_ops = [accuracy_update, recall_update, precision_update]
    metric_ops = [accuracy, recall, precision]


summaries_dir = '/data/bigbone4/ciriondo/cartilageX3/cartilage-x/initial_testing/tensorboard'
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      tf.get_default_graph())
valid_writer = tf.summary.FileWriter(summaries_dir + '/test')


# function for convenience
with tf.variable_scope('eval') as scope:
    def get_stream_metric(session, iterator_handle):
        """Get total accuracy from a set of batch results"""
        v = session.run(reset_ops)
        while True:
            try:
#                 summary, _= session.run([merged, update_ops], feed_dict={handle: iterator_handle})
                _, _, _ = session.run([y_batch, prediction, update_ops], feed_dict={handle: iterator_handle})
                #                 print(np.argmax(ys,-1), np.argmax(preds,-1))
            except tf.errors.OutOfRangeError:
                metrics = session.run(metric_ops) # ==> "End of dataset"
                break    
        return summary, metrics


# **Summaries for TensorBoard:**
# **TRAINING**
# 
# To visualize, run:
# tensorboard --logdir=run1:/data/bigbone4/ciriondo/cartilageX3/cartilage-x/initial_testing/tensorboard/ port 6006

# needs to be increased when actually trianing stuff
num_steps = 1500

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:

    # initialize all variables
    tf.global_variables_initializer().run()
    sess.run(reset_ops)  # FIXME: metric variables not being intialized correctly

    # initialize handles to pass into iterator placeholder
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())
    eval_train_handle = sess.run(eval_train_iterator.string_handle())

    print("Initialized")

    # sess.run(iterator.initializer, feed_dict={handle: train_handle})
    for step in range(num_steps):   

        # record validation summaries every n steps
        if step % 50 == 0:
            summary, _, l, batch_acc = sess.run([merged, optimizer, loss, batch_accuracy],
                                           feed_dict={handle: train_handle})
            train_writer.add_summary(summary, step)      
            print('Step {} of {}. Loss: {:.2f}. Batch accuracy: {:.3f}'.format(step, 
                                                                               num_steps,
                                                                               l,
                                                                               batch_acc))

            # option to print less frequenty
            if step % 200 == 0:
                sess.run(valid_iterator.initializer)
                summary, valid_metrics = get_stream_metric(sess, valid_handle)
                
                print("\nLoss at step %i: %f" % (step, l))
                print("Batch accuracy: {:.3f}".format(batch_acc))

                # Validation accuracy
                print('VALIDATION')
                print("\t{:12} {:.3f}".format('Accuracy:', valid_metrics[0]))
                print("\t{:12} {:.3f}".format('Recall:', valid_metrics[1]))
                print("\t{:12} {:.3f}".format('Precision:', valid_metrics[2]))

        # run training step
        else:
            _ = sess.run(optimizer, feed_dict={handle: train_handle})       

    # Full training set accuracy        
    print('\nFULL TRAINING SET')
    sess.run(eval_train_iterator.initializer)
    _, train_metrics = get_stream_metric(sess, eval_train_handle)
    print("\t{:12} {:.3f}".format('Accuracy:', train_metrics[0]))
    print("\t{:12} {:.3f}".format('Recall:', train_metrics[1]))
    print("\t{:12} {:.3f}".format('Precision:', train_metrics[2]))

    # Valid accuracy
    print('\nVALID')
    sess.run(valid_iterator.initializer)
    _, valid_metrics = get_stream_metric(sess, valid_handle)
    print("\t{:12} {:.3f}".format('Accuracy:', valid_metrics[0]))
    print("\t{:12} {:.3f}".format('Recall:', valid_metrics[1]))
    print("\t{:12} {:.3f}".format('Precision:', valid_metrics[2]))
