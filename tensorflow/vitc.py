'''
ViTc implementation in TensorFlow
Based on implementation by Khalid Salama 
Source: https://keras.io/examples/vision/image_classification_with_vision_transformer/

Modified by Maria R. Lily Djami

'''


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Multilayer Perceptron (MLP)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patchify operation is implemented as 2 layers,
# the Patches layer and the PatchEncoder layer.
    
def create_vitc_classifier(input_shape, num_classes, data_augmentation, kernel = 72, stem="1GF", transformer_layers = 12, projection_dim = 64, mlp_head_units = [2048, 1024] ):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  

    ####################################################
    #   Here is the stem of the vision transformer.    #
    #   For ViTc, we want to use convolutional layers  #
    #   instead of patchify to get the encoded image   #
    ####################################################
    
    cnn_stem = keras.Sequential()
    
    if stem == '18GF':
        num_heads = 12

        cnn_stem.add(layers.Conv2D(64, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(128, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(128, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(256, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(256, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(512, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
    elif stem == '4GF':
        num_heads = 6
        
        cnn_stem.add(layers.Conv2D(48, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(96, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(192, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(384, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
    elif stem == '1GF': 
        num_heads = 3
        
        cnn_stem.add(layers.Conv2D(24, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(48, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(96, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
        cnn_stem.add(layers.Conv2D(192, kernel, strides=(2,2)))
        cnn_stem.add(layers.BatchNormalization())
        cnn_stem.add(layers.ReLU())
    else: 
        print("Invalid stem design given!")
        return -1
    
    cnn_stem.add(layers.Conv2D(projection_dim, 1, strides=(1,1)))
    
    encoded_cnn = cnn_stem(augmented)

    ######################################################
    #   The code block below is the transformer block.   #
    #   This remains the same between ViTp and ViTc.     #
    ######################################################

    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model