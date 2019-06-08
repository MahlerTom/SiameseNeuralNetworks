from .model import siamese_model, initialize_weights, initialize_weights_dense, initialize_bias
from .distance import absolute_distance
from .data import init_ds, image_name

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from skimage.transform import resize
from skimage import io
from tensorflow.random import set_seed
from numpy.random import seed

def model_fit(    
    train_paths_labels,
    val_paths_labels,
    table=None,
    _resize=[250, 250],
    norm=255.0,
    batch_size=128,
    filters=4,
    lr=1e-3,
    epochs=30,
    verbose=1, 
    pretrained_weights=None,
    model_path=None,
    distance=absolute_distance,
    distance_output_shape=None,
    prediction_activation='sigmoid',
    train_ds=None,
    val_ds=None,
    callbacks=None,
    steps_per_epoch=None,
    validation_steps=None,
    prefix='',
#     shuffle=True,
    patience=3,
    kernel_initializer=initialize_weights,
    kernel_initializer_d=initialize_weights_dense,
    kernel_regularizer=l2(2e-4),
    kernel_regularizer_d=l2(1e-3),
    bias_initializer=initialize_bias,
    kernel_size_list=[(10, 10), (7, 7), (4, 4), (4, 4)],
    units=4*64,
    optimizer=None,
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='Precision'), Recall(name='Recall')],
    tensorboard_histogram_freq=1,
):  
    start_time = time.time()
    input_shape = (_resize[0], _resize[1], 1)
    
    if optimizer is None:
      optimizer = Adam(lr=lr)

    run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
    run_name = f'{prefix}_{run_time}_lr{lr}_f{filters}_s{_resize[0]}_e{epochs}_b{batch_size}'
    log_dir = f'logs/fit/{run_name}'
  
    if steps_per_epoch is None:
        steps_per_epoch = len(train_paths_labels) // batch_size
    if validation_steps is None:
        validation_steps = len(val_paths_labels) // batch_size
#     if metrics is None:
#         metrics = [dice_coef, Precision(name='Percision'), Recall(name='Recall'), dice_coef_liver, dice_coef_tumor]
  
    # Print stat
    print(f'steps_per_epoch = {steps_per_epoch}, validation_steps = {validation_steps}')
    if table is not None:
#         from prettytable import PrettyTable
#         table = PrettyTable(['Run', 'Name', 'Optimizer', 'Batch Size', 'Resize', 'Filters', 'Learning Rate', 'Epochs'])
        table.add_row([run_time, prefix, 'Adam', batch_size, _resize[0], filters, lr, epochs])
        print(table)

    # Create Datasets
    train_ds = init_ds(train_ds, images_labels_paths=train_paths_labels, norm=norm, _resize=_resize, batch_size=batch_size)
  
    val_ds = init_ds(val_ds, images_labels_paths=val_paths_labels, norm=norm, _resize=_resize, batch_size=batch_size)

    # Create Model
    model = siamese_model(
        input_shape=input_shape, 
        filters=filters, 
        kernel_initializer=kernel_initializer,
        kernel_initializer_d=kernel_initializer_d,
        kernel_regularizer=kernel_regularizer,
        kernel_regularizer_d=kernel_regularizer_d,
        bias_initializer=bias_initializer,
        kernel_size_list=kernel_size_list,
        units=units,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        pretrained_weights=pretrained_weights,
        model_path=model_path,
        distance=distance,
        distance_output_shape=distance_output_shape,
        prediction_activation=prediction_activation,
    )

    if pretrained_weights is None:
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=tensorboard_histogram_freq,
        )
        early_stop = EarlyStopping(patience=patience, verbose=verbose)
        mc = ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)

        if callbacks is None:
            callbacks = []
        callbacks.append(tensorboard_callback)
        callbacks.append(early_stop)
        callbacks.append(mc)
        
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
#             shuffle=shuffle
        )  
#         model.save_weights(f'{run_name}.h5')
    print(f'--- {(time.time() - start_time):.2f} seconds ---')
    return model, table

def model_evaluate(model, images_labels_paths, ds=None, norm=255.0, _resize=[125, 125], verbose=1): 
    ds = init_ds(
        ds, 
        images_labels_paths=images_labels_paths, 
        norm=norm, _resize=_resize, batch_size=1)    
    return model.evaluate(ds, steps=len(images_labels_paths), verbose=verbose)

def model_predict(
    model,
    images_labels_paths,
    ds=None,
    steps=None,     
    verbose=1,
    images_to_print=0,
    norm=255.0,
    _resize=[250, 250],
    save_image_path=None,
):
    if steps is None:
        steps = len(images_labels_paths)
    
    ds = init_ds(
        ds, 
        images_labels_paths=images_labels_paths, norm=norm, _resize=_resize,
        batch_size=1, verbose=verbose)
    
    pred = np.squeeze(model.predict(ds, verbose=verbose, steps=steps))
    print(pred.shape)

    if images_to_print is None:
        images_to_print = len(images_labels_paths)

    num_of_splt = 2
    splt = 1
    plt.figure(figsize=(num_of_splt*4, images_to_print*4))
#     if save_image_path is not None:
#         for i in range(len(images_paths)):
#             Image.fromarray(quantizatize(np.squeeze(pred[i,:,:,:]), 3, 170.0)*255).convert('L').save(os.path.join(save_image_path, f'{image_name(images_paths[i])}.png'))
      
    for i in range(images_to_print):
        left_image, right_image, y_true = images_labels_paths[i]
        y_pred = pred[i]
        plt.subplot(images_to_print, num_of_splt, splt)
        plt.imshow(resize(io.imread(left_image, as_gray=True), (_resize[0], _resize[1])), cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} {image_name(left_image)}\n pred={y_pred:.2f}, true={y_true}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+1)
        plt.imshow(resize(io.imread(right_image, as_gray=True), (_resize[0], _resize[1])), cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} {image_name(right_image)}\n pred={y_pred:.2f}, true={y_true}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        splt += num_of_splt
    
    plt.show()
    return pred

def model_fit_eval(    
    train_paths_labels,
    val_paths_labels,
    test_paths_labels,
    eval_table=None,
    table=None,
    _resize=[250, 250],
    norm=255.0,
    batch_size=128,
    filters=4,
    lr=1e-3,
    epochs=30,
    verbose=1, 
    pretrained_weights=None,
    model_path=None,
    distance=absolute_distance,
    distance_output_shape=None,
    prediction_activation='sigmoid',
    train_ds=None,
    val_ds=None,
    callbacks=None,
    steps_per_epoch=None,
    validation_steps=None,
    prefix='',
#     shuffle=True,
    patience=3,
    kernel_initializer=initialize_weights,
    kernel_initializer_d=initialize_weights_dense,
    kernel_regularizer=l2(2e-4),
    kernel_regularizer_d=l2(1e-3),
    bias_initializer=initialize_bias,
    kernel_size_list=[(10, 10), (7, 7), (4, 4), (4, 4)],
    units=4*64,
    optimizer=None,
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='Precision'), Recall(name='Recall')],
    tensorboard_histogram_freq=1,
    random_seed=2,
):
    seed(random_seed)
    set_seed(random_seed)
    model, _ = model_fit(table=table, train_paths_labels=train_paths_labels,
                            val_paths_labels=val_paths_labels, _resize=_resize, norm=norm,
                            batch_size=batch_size, filters=filters, lr=lr, epochs=epochs,
                            loss=loss, metrics=metrics, verbose=verbose,
                            pretrained_weights=pretrained_weights, model_path=model_path,
                            prediction_activation=prediction_activation, 
                            distance=distance, distance_output_shape=distance_output_shape,
                            train_ds=train_ds, val_ds=val_ds, callbacks=callbacks,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                            prefix=prefix, patience=patience, tensorboard_histogram_freq=tensorboard_histogram_freq,
                        )
    scores = model_evaluate(model, images_labels_paths=test_paths_labels, norm=norm, _resize=_resize, verbose=verbose)
    if eval_table is not None:
        eval_table.add_row(scores)
        print(eval_table)
    else:
        print(scores)

    return model