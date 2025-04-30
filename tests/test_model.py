import pytest
import tensorflow as tf
from tensorflow.keras.models import Model

# Adjust the import path based on your project structure if necessary
from src.autoencoder.model import build_autoencoder

def test_build_autoencoder_structure():
    """
    Tests the structure of the autoencoder model built by build_autoencoder.
    """
    input_dim = 880  # Default value used in the function
    bottleneck_dim = 100 # Default value used in the function

    # build_autoencoder returns both autoencoder and encoder, we test the autoencoder
    model, encoder = build_autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)

    # 1. Check if the returned object is a Keras Model
    assert isinstance(model, Model), "The returned object should be a Keras Model."

    # 2. Check the number of layers
    # Expected layers: Input, Dense(256), Dense(bottleneck), Dense(256), Dense(output)
    expected_layers = 5
    assert len(model.layers) == expected_layers, f"Model should have {expected_layers} layers, but found {len(model.layers)}."

    # 3. Check input shape
    # Keras adds the batch dimension (None)
    expected_input_shape = (None, input_dim)
    assert model.input_shape == expected_input_shape, f"Model input shape should be {expected_input_shape}, but found {model.input_shape}."

    # 4. Check output shape
    expected_output_shape = (None, input_dim)
    assert model.output_shape == expected_output_shape, f"Model output shape should be {expected_output_shape}, but found {model.output_shape}."

    # 5. Check bottleneck layer units
    # The bottleneck layer is the 3rd layer in the sequence (index 2)
    bottleneck_layer_index = 2
    assert model.layers[bottleneck_layer_index].units == bottleneck_dim, \
        f"Bottleneck layer (index {bottleneck_layer_index}) should have {bottleneck_dim} units, but found {model.layers[bottleneck_layer_index].units}."

    # Optional: Check encoder model structure briefly
    assert isinstance(encoder, Model), "The returned encoder should also be a Keras Model."
    assert encoder.output_shape == (None, bottleneck_dim), f"Encoder output shape should be {(None, bottleneck_dim)}, but found {encoder.output_shape}."