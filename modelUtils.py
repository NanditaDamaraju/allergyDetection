import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow.keras as keras 
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, MobileNetV2, InceptionV3
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# since we will be testing many models, let's create a function to train, evaluate, and save statistics about each model 
# my_model: the keras model to train
def train_model(my_model): 
    
    # compile the model
    my_model.compile(loss='mean_squared_error',
        optimizer=optimizers.Adam(),
        metrics=['acc'])
    
    # create keras callbacks 
    es = EarlyStopping(monitor='val_loss', patience=5) 
        
    # train the model
    history = my_model.fit(
        imgs_train,
        labels_train, 
        validation_data=(imgs_val,labels_val), 
        batch_size=32, 
        epochs=25, 
        callbacks=[es] 
    )
    
    # calculate confusion matrix 
    predictions = my_model.predict(imgs_test)
    plot_confusion_matrix(my_model, labels_test.argmax(axis=1), predictions.argmax(axis=1), ['No Reaction', 'Yes Reaction'])
    
    # save training statistics 
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    
    plt.figure()

    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(my_model.name + '_accuracy.png') 
    
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(my_model.name + '_loss.png') 
    
    # Save the model
    my_model.save(my_model.name + '.h5')


   

# This function comes from a scikit example found here: 
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html 
def plot_confusion_matrix(model, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)
    return ax

def create_custom_model():
	# create the base pre-trained model
	pretrained_model = VGG16(weights='imagenet', include_top=False)

	# add a global spatial average pooling layer
	x = pretrained_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(2, activation='softmax')(x)

	# this is the model we will train
	custom_model = Model(inputs=pretrained_model.input, outputs=predictions)

	# freeze the layers that came in the pre-trained model
	for layer in pretrained_model.layers:
	    layer.trainable = False

	return custom_model
