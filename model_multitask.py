import tensorflow as tf
from tensorflow.keras import backend
import tensorflow.keras.backend as K
from typing import Callable
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

class Mean_IoU(MeanIoU):
    def __init__(self, num_classes, th):
        super().__init__(num_classes)
        self.th=th
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        #y_true = tf.argmax(y_true, axis=-1)
        y_pred_ = tf.cast(y_pred > self.th, tf.int32)
        return super().update_state(y_true, y_pred_, sample_weight)


def convert_to_logits(y_pred: tf.Tensor) -> tf.Tensor:
    """
    Converting output of sigmoid to logits.
    :param y_pred: Predictions after sigmoid (<BATCH_SIZE>, shape=(None, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1)).
    :return: Logits (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
    """
    # To avoid unwanted behaviour of log operation
    y_pred = K.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return K.log(y_pred / (1 - y_pred))
    
    
class UNETx():
    def __init__(self, img_size, weights=None, save_weights_path=None, dataset_name='default', lr=1e-3):
        self.img_size = img_size
        self.weights = weights
        self.save_weights_path = save_weights_path
        self.dataset_name = dataset_name
        self.lr = lr
        
    def get_model07_expaned(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.7, 'B_channel': 0.1, 'G_channel': 0.1, 'R_channel': 0.1}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model

    def get_model055_expaned(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.55, 'B_channel': 0.15, 'G_channel': 0.15, 'R_channel': 0.15}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model    
     
    def get_model04_expaned(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.4, 'B_channel': 0.2, 'G_channel': 0.2, 'R_channel': 0.2}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model
        
        
    def get_model07(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.7, 'B_channel': 0.1, 'G_channel': 0.1, 'R_channel': 0.1}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model
        
    def get_model055(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.55, 'B_channel': 0.15, 'G_channel': 0.15, 'R_channel': 0.15}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model
        
    def get_model04(self, Beta=15):
        img_size=self.img_size
        
        #Build the model
        inputs = tf.keras.layers.Input(shape=img_size + (1, ))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
        
        #Contraction path
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
         
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
         
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
         
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
         
        u7 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
         
        u8 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
         
        u9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs1 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='iris_segmentation')(c9)
        outputs2 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='B_channel')(c9)
        outputs3 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='G_channel')(c9)
        outputs4 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='R_channel')(c9)

        outputsx = [outputs1, outputs2, outputs3, outputs4]

        model = tf.keras.Model(inputs=[inputs], outputs=outputsx)
        model.summary()
        
        losses = {'iris_segmentation': 'binary_crossentropy', 'B_channel': 'mae', 'G_channel': 'mae', 'R_channel': 'mae'} 
        loss_weights = {'iris_segmentation': 0.4, 'B_channel': 0.2, 'G_channel': 0.2, 'R_channel': 0.2}
        
        metrics = {'iris_segmentation': ['accuracy', Mean_IoU(2, 0), Mean_IoU(2, 0.1), Mean_IoU(2, 0.2), Mean_IoU(2, 0.3), Mean_IoU(2, 0.4), Mean_IoU(2, 0.5), Mean_IoU(2, 0.6), Mean_IoU(2, 0.7), Mean_IoU(2, 0.8), Mean_IoU(2, 0.9), Mean_IoU(2, 1)], 'B_channel': ['mse', 'mae'], 'G_channel': ['mse', 'mae'], 'R_channel': ['mse', 'mae']}
        
        model.compile(optimizer=Adam(learning_rate=self.get_lr_schedule(start_lr=self.lr)), loss=losses, loss_weights=loss_weights, metrics=metrics, experimental_run_tf_function=False)
        
        if self.weights is not None:
            model.load_weights(self.weights)
        return model    
    def get_lr_schedule(self, start_lr=1e-3, decay_steps=50, decay_rate=0.95, staircase=True):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            start_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase)

        return lr_schedule 
    
    def get_callbacks(self, save_weights_path, dataset_name):
        if save_weights_path[-1] != '/':
            save_weights_path += '/'

        filepath = save_weights_path + 'UNet_{}_'.format(dataset_name) + '{epoch:04d}_mean_iou_{val_iris_segmentation_mean__io_u_5:05.4f}_B_channel_mae_{val_B_channel_mae:05.4f}_G_channel_mae_{val_G_channel_mae:05.4f}_R_channel_mae_{val_R_channel_mae:05.4f}.h5'

        cp_callback_0 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_iris_segmentation_mean__io_u_5',
                mode='max',
                save_best_only=True
        )

        cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_B_channel_mae',
                mode='min',
                save_best_only=True
        )

        cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_G_channel_mae',
                mode='min',
                save_best_only=True
        )
        
        cp_callback_3 = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                verbose=1,
                save_weights_only=True,
                monitor='val_R_channel_mae',
                mode='min',
                save_best_only=True
        )

        return [cp_callback_0, cp_callback_1, cp_callback_2, cp_callback_3]
    
    def binary_weighted_cross_entropy(self, beta: float, is_logits: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Weighted cross entropy. All positive examples get weighted by the coefficient beta:
            WCE(p, p̂) = −[β*p*log(p̂) + (1−p)*log(1−p̂)]
        To decrease the number of false negatives, set β>1. To decrease the number of false positives, set β<1.
        If last layer of network is a sigmoid function, y_pred needs to be reversed into logits before computing the
        weighted cross entropy. To do this, we're using the same method as implemented in Keras binary_crossentropy:
        https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        Used as loss function for binary image segmentation with one-hot encoded masks.
        :param beta: Weight coefficient (float)
        :param is_logits: If y_pred are logits (bool, default=False)
        :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
        """
        def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """
            Computes the weighted cross entropy.
            :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
            :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
            :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
            """
            if not is_logits:
                y_pred = convert_to_logits(y_pred)
    
            wce_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
    
            # Average over each data point/image in batch
            axis_to_reduce = range(1, K.ndim(wce_loss))
            wce_loss = K.mean(wce_loss, axis=axis_to_reduce)
    
            return wce_loss
    
        return loss
        
    def weighted_binary_crossentropy_update(self):
        
        def dyn_weighted_bincrossentropy(true, pred):
            """
            Calculates weighted binary cross entropy. The weights are determined dynamically
            by the balance of each category. This weight is calculated for each batch.
            
            The weights are calculted by determining the number of 'pos' and 'neg' classes 
            in the true labels, then dividing by the number of total predictions.
            
            For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
            These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
            1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
            
            This can be useful for unbalanced catagories.
            """
            # get the total number of inputs
            num_pred = K.sum(K.cast(pred < 0.5, true.dtype)) + K.sum(true)
            
            # get weight of values in 'pos' category
            zero_weight =  K.sum(true)/ num_pred +  K.epsilon() 
            
            # get weight of values in 'false' category
            one_weight = K.sum(K.cast(pred < 0.5, true.dtype)) / num_pred +  K.epsilon()
        
            # calculate the weight vector
            weights =  (1.0 - true) * zero_weight +  true * one_weight 
            
            #if is_logits:
            #   pred=convert_to_logits(pred)
                
            # calculate the binary cross entropy
            bin_crossentropy = K.binary_crossentropy(true, pred)
            
            # apply the weights
            weighted_bin_crossentropy = weights * bin_crossentropy 
        
            return K.mean(weighted_bin_crossentropy)
            
        return dyn_weighted_bincrossentropy
        
    def dice_metric_func(self, th=0.5):
        def dice_metric(y_pred, y_true):
            y_pred_ = tf.cast(y_pred > th, tf.float32)
            intersection = K.sum(K.sum(K.abs(y_true * y_pred_), axis=-1))
            union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred_), axis=-1))
            # if y_pred.sum() == 0 and y_pred.sum() == 0:
            #     return 1.0
        
            return 2*intersection / union
        return dice_metric
    
    def recall_func(self, th=0.5):    
        def recall(y_true, y_pred):
            y_true = K.ones_like(y_true)
            y_pred_ = tf.cast(y_pred > th, tf.int32)
            true_positives = K.sum(K.round(K.clip(y_true * y_pred_, 0, 1)))
            all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            
            recall = true_positives / (all_positives + K.epsilon())
            return recall
        return recall
        
        
    def precision_func(self, th=0.5):
        def precision(y_true, y_pred):
            y_true = K.ones_like(y_true) 
            y_pred_ = tf.cast(y_pred > th, tf.int32)
            true_positives = K.sum(K.round(K.clip(y_true * y_pred_, 0, 1)))
            
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        
        
    def f1_score_func(self, th=0.5):
        def f1_score(y_true, y_pred):
            precision = self.precision_func(th)
            recall = self.recall_func(th)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        return f1_score







