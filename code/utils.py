import matplotlib.cm as cm
import tensorflow as tf

def colorize(weights):

    # squeeze last dim if it exists
    weights = tf.squeeze(weights)

    # normalize
    #wmin = tf.reduce_min(weights)
    #wmax = tf.reduce_max(weights)
    #weights = (weights - wmin) / (wmax - wmin)

    # quantize
    indices = tf.to_int32(tf.round(weights * 255))

    # gather
    cmap = cm.jet(range(256))[:,0:3]
    colors = tf.constant(cmap, dtype=tf.float32)
    colors = tf.gather(colors, indices)
    colors = colors * 128
    colors = tf.expand_dims(colors,0)

    return colors

def IoU(preds,labels,thresh):
    labels = tf.cast(labels,tf.bool)
    preds = tf.cast(preds,tf.bool)
    iou = tf.divide(tf.reduce_sum(tf.cast(tf.logical_and(labels,preds),tf.float64),axis=1),
            tf.reduce_sum(tf.cast(tf.logical_or(labels,preds),tf.float64),axis=1))
    iou = tf.where(tf.is_nan(iou),tf.zeros_like(iou),iou)
    iou = tf.cast(tf.greater(iou,thresh),tf.float64)
    iou = tf.divide(tf.reduce_sum(iou,axis=0),
            tf.reduce_sum(tf.reduce_max(tf.cast(labels,tf.float64),axis=1),axis=0))
    iou = tf.where(tf.is_nan(iou),tf.zeros_like(iou),iou)
    return iou

def AUROC(probs,labels,num_classes):
    auroc = []
    for i in range(num_classes):
        auroc += [tf.metrics.auc(tf.slice(labels,[0,i],[num_classes,1]),
                tf.slice(probs,[0,i],[num_classes,1]))]
    auroc = tf.reduce_mean(tf.stack(auroc),axis=1)
    return auroc

def get_weights(labels):
    labels = tf.squeeze(labels)
    labels_neg = tf.cast(tf.equal(labels,0),labels.dtype)
    pos = tf.reduce_sum(labels,axis=0)
    neg = tf.reduce_sum(labels_neg,axis=0)
    beta_pos = tf.divide(tf.add(pos,neg),pos)
    beta_neg = tf.divide(tf.add(pos,neg),neg)
    weights_pos = tf.multiply(labels,beta_pos)
    weights_neg = tf.multiply(labels_neg,beta_neg)
    weights = tf.add(weights_pos,weights_neg)
    return tf.where(tf.is_nan(weights), tf.ones_like(weights), weights)
