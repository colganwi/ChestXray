import matplotlib.cm as cm

import tensorflow as tf

def colorize(weights):

    # squeeze last dim if it exists
    weights = tf.image.resize_images(weights,[224,224])
    weights = tf.squeeze(weights)

    # normalize
    wmin = tf.reduce_min(weights)
    wmax = tf.reduce_max(weights)
    weights = (weights - wmin) / (wmax - wmin) # vmin..vmax

    # quantize
    indices = tf.to_int32(tf.round(weights * 255))

    # gather
    cmap = cm.jet(range(256))[:,0:3]
    colors = tf.constant(cmap, dtype=tf.float32)
    colors = tf.gather(colors, indices)
    colors = colors * 128
    colors = tf.expand_dims(colors,0)

    return colors
