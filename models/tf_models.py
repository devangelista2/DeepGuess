from keras import layers, models

def ResUNet(input_shape, n_scales, conv_per_scale, init_conv=64, final_relu=False):
    """
    Define the ResUNet Model, following the paper A Residual Dense U-Net Neural Network for Image Denoising.

    input_shape -> Tuple, input dimension
    n_scales -> Number of downsampling
    conv_per_scale -> Number of convolutions for each scale
    init_conv -> Number of convolutional filters at the first scale
    """
    n_ch = init_conv
    skips = []

    x = layers.Input(input_shape)
    h = x

    # ANALYSIS
    for scale in range(n_scales):
        for c in range(conv_per_scale):
            h = layers.Conv2D(n_ch, 3, 1, padding='same')(h)
            h = layers.BatchNormalization()(h)
            h = layers.ReLU()(h)
        
        skips.append(h)
        h = layers.MaxPooling2D()(h)
        n_ch = n_ch * 2

    # FILTERING
    for c in range(conv_per_scale):
        h = layers.Conv2D(n_ch, 3, 1, padding='same')(h)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
    
    n_ch = n_ch // 2
    h = layers.Conv2DTranspose(n_ch, 3, 1, padding='same')(h)
    h = layers.UpSampling2D()(h)

    # SYNTHESIS
    for scale in range(n_scales):
        h = layers.Concatenate()([h, skips.pop(-1)])
        for c in range(conv_per_scale):
            h = layers.Conv2D(n_ch, 3, 1, padding='same')(h)
            h = layers.BatchNormalization()(h)
            h = layers.ReLU()(h)
    
        if scale < n_scales-1:
            n_ch = n_ch // 2
            h = layers.Conv2DTranspose(n_ch, 3, 1, padding='same')(h)
            h = layers.UpSampling2D()(h)

    y = layers.Conv2D(input_shape[-1], 1, 1, padding='same', activation='tanh')(h)
    y = layers.Add()([x, y])

    if final_relu:
        y = layers.ReLU()(y)
    return models.Model(x, y)


"""
3L-SSN -> The simple Convolutional Neural Network proposed in out layer.

input_shape -> Tuple, input dimension
"""
def get_3lSSN(input_shape):
    x = layers.Input(input_shape)
    
    # Synthesis Layer
    h = layers.Conv2D(128, 9, 1, padding='same')(x) 
    h = layers.ReLU()(h)
    
    # Sparsifier Layer
    h = layers.Conv2D(64, 3, 1, padding='same')(h)
    h = layers.ReLU()(h)
    
    # Analysis Layer
    y = layers.Conv2D(1, 5, 1, padding='same')(h)
    return models.Model(x, y)