import keras
import tensorflow as tf
try:
    from . import roi
except Exception as e:
    import roi

def classifier(base_layers, input_rois, num_rois, nb_classes = 4):
    input_shape = (num_rois,7,7,512)

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = roi.RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = keras.layers.TimeDistributed(keras.layers.Flatten(name='flatten'))(out_roi_pool)
    out = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation='relu', name='fc1'))(out)
    out = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(out)
    out = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation='relu', name='fc2'))(out)
    out = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = keras.layers.TimeDistributed(keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = keras.layers.TimeDistributed(keras.layers.Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
