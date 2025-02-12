from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 

model_name = 'wv_k_8_c_5'

#some hyperparameters 
#Play with them!
input_shape = (50,50,3)
batch_size = 1024 
learning_rate = 0.001
epochs = 20

#model architecture (with Quantization Aware Training - QAT)
#Play with it!
inputs = keras.Input(shape=input_shape)
#

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, DepthwiseConv2D, Conv2D, BatchNormalization,
    Activation, Add, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

def identity_block(x, filters, kernel_size=3):
    """An identity block using depthwise separable convolutions."""
    shortcut = x

    # First depthwise separable block:
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Second depthwise separable block:
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut (assumed to have same dimensions) and activate
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, strides=2):
    """A convolutional block using depthwise separable convolutions for downsampling."""
    shortcut = x

    # Main branch: first separable block with downsampling via strides.
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Second separable block without additional downsampling.
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut branch: use a 1x1 convolution to match dimensions.
    shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # Add shortcut and main branch, then apply activation.
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNetLiteDepthwise(input_shape, num_classes):
    """Constructs a lightweight ResNet-Lite model using depthwise separable convolutions.
    
    Args:
        input_shape: Tuple of the input dimensions, e.g. (50, 50, 3) for Color images.
        num_classes: Number of output classes.
    
    Returns:
        A Keras Model instance.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution (not separable) to lift input channels.
    x = Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # First residual stage: no spatial downsampling; use identity block.
    x = identity_block(x, filters=32)
    
    # Second residual stage: downsample and increase filter count.
    x = conv_block(x, filters=64, strides=2)
    x = identity_block(x, filters=64)
    
    # Third residual stage: further downsampling.
    #x = conv_block(x, filters=128, strides=2)
    #x = identity_block(x, filters=128)
    
    # Global average pooling and the classification head.
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="ResNetLite_Depthwise")
    return model


model = ResNetLiteDepthwise(input_shape=(50, 50, 3), num_classes=2)
#model = keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,  # This flag enables XLA for the training step.
)


#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')
val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #add some data augmentation 
    #Play with it!
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])
    
train_ds = train_ds.shuffle(1000).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_sparse_categorical_accuracy',
    mode='max', save_best_only=True)
    
#training
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[model_checkpoint_callback])

#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
