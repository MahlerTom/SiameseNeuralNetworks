from tensorflow.keras import backend as K

def absolute_distance(tensors, K=K):
    return K.abs(tensors[0] - tensors[1])

def euclidean_distance(tensors, K=K):
    return K.sqrt(K.sum(K.square(absolute_distance(tensors, K=K)), axis=1, keepdims=True))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def distance_accuracy(y_true, y_pred, th=0.5, K=K):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < th, y_true.dtype)))

def contrastive_loss(y_true, y_pred, margin=0.5):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)