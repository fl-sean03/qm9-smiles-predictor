# src/autoencoder/model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim: int = 880, bottleneck_dim: int = 100):
    inp = Input(shape=(input_dim,), name="ae_input")
    x1  = Dense(256, activation="relu", name="ae_enc1")(inp)
    btl = Dense(bottleneck_dim, activation="relu", name="bottleneck")(x1)
    x2  = Dense(256, activation="relu", name="ae_dec1")(btl)
    out = Dense(input_dim, activation="linear", name="ae_output")(x2)

    ae = Model(inp, out, name="autoencoder")
    ae.compile(optimizer="adam", loss="mse")
    encoder = Model(inp, btl, name="encoder")
    return ae, encoder
