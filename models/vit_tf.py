import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Config taken from official implementation
MODEL_CONFIGS = {
    "vit_T_16": {
        "dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "num_layers": 12,
        "dropout_rate": 0.0,
    },
    "vit_S_16": {
        "dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "num_layers": 12,
        "dropout_rate": 0.0,
    },
    "vit_B_16": {
        "dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "dropout_rate": 0.0,
    },
    "vit_L_16": {
        "dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "num_layers": 24,
        "dropout_rate": 0.1,
    },
    "vit_H_16": {
        "dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 3,
        "num_layers": 32,
        "dropout_rate": 0.1,
    },
}


class PatchingAndEmbedding(layers.Layer):
    def __init__(self, dim):
        super(PatchingAndEmbedding, self).__init__()
        self.dim = dim
        self.projection = layers.Conv2D(
            filters=self.dim,
            kernel_size=16,
            strides=16,
            padding="VALID",
        )

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=[1, 1, self.dim], name="class_token", trainable=True
        )
        self.num_patches = input_shape[1] // 16 * input_shape[2] // 16
        self.pos_embed = layers.Embedding(
            input_dim=self.num_patches + 1, output_dim=self.dim
        )

    def call(
        self,
        x,
    ):
        x = self.projection(x)
        patch_shapes = tf.shape(x)
        x1 = tf.reshape(
            x,
            shape=(
                patch_shapes[0],
                patch_shapes[-2] * patch_shapes[-2],
                patch_shapes[-1],
            ),
        )
        x2 = tf.shape(x1)
        x3 = tf.cast(
            tf.broadcast_to(
                self.class_token,
                [x2[0], 1, x2[-1]],
            ),
            dtype=x1.dtype,
        )
        x1 = tf.concat(
            [x3, x1], 1
        )
        pos = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        outputs = x1 + self.pos_embed(pos)
        return outputs


class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.1,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.norm1 = layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attention_dropout,
        )
        self.drop = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=layer_norm_epsilon)

        self.dense1 = layers.Dense(mlp_dim)
        self.dense2 = layers.Dense(dim)

        self.add = layers.Add()

    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x, x)
        x = self.drop(x)
        x = self.add([x, inputs])

        y = self.norm2(x)

        y = self.dense1(y)
        y = tf.nn.gelu(y)
        y = self.drop(y)
        y = self.dense2(y)
        y = self.drop(y)

        output = self.add([x, y])

        return output


def ViT(
    input_shape=(None, None, 3),
    classes=None,
    num_layers=None,
    num_heads=None,
    dropout_rate=None,
    dim=None,
    mlp_dim=None,
):
    inputs = layers.Input(input_shape)
    x = inputs
    x = PatchingAndEmbedding(dim)(x)
    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_layers):
        x = TransformerEncoder(
            dim=dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Lambda(lambda rep: rep[:, 0])(x)
    x = layers.Dense(classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=x)
    return model


def vit_T_16(
    input_shape=(None, None, 3),
    classes=None,
    **kwargs,
):
    return ViT(
        input_shape=input_shape,
        classes=classes,
        num_layers=MODEL_CONFIGS["vit_T_16"]["num_layers"],
        dim=MODEL_CONFIGS["vit_T_16"]["dim"],
        mlp_dim=MODEL_CONFIGS["vit_T_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["vit_T_16"]["num_heads"],
        dropout_rate=MODEL_CONFIGS["vit_T_16"]["dropout_rate"],
        **kwargs,
    )


def vit_S_16(
    input_shape=(None, None, 3),
    classes=None,
    **kwargs,
):
    return ViT(
        input_shape=input_shape,
        classes=classes,
        num_layers=MODEL_CONFIGS["vit_S_16"]["num_layers"],
        dim=MODEL_CONFIGS["vit_S_16"]["dim"],
        mlp_dim=MODEL_CONFIGS["vit_S_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["vit_S_16"]["num_heads"],
        dropout_rate=MODEL_CONFIGS["vit_S_16"]["dropout_rate"],
        **kwargs,
    )


def vit_B_16(
    input_shape=(None, None, 3),
    classes=None,
    **kwargs,
):
    return ViT(
        input_shape=input_shape,
        classes=classes,
        num_layers=MODEL_CONFIGS["vit_B_16"]["num_layers"],
        dim=MODEL_CONFIGS["vit_B_16"]["dim"],
        mlp_dim=MODEL_CONFIGS["vit_B_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["vit_B_16"]["num_heads"],
        dropout_rate=MODEL_CONFIGS["vit_B_16"]["dropout_rate"],
        **kwargs,
    )


def vit_L_16(
    input_shape=(None, None, 3),
    classes=None,
    **kwargs,
):
    return ViT(
        input_shape=input_shape,
        classes=classes,
        num_layers=MODEL_CONFIGS["vit_L_16"]["num_layers"],
        dim=MODEL_CONFIGS["vit_L_16"]["dim"],
        mlp_dim=MODEL_CONFIGS["vit_L_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["vit_L_16"]["num_heads"],
        dropout_rate=MODEL_CONFIGS["vit_L_16"]["dropout_rate"],
        **kwargs,
    )


def vit_H_16(
    input_shape=(None, None, 3),
    classes=None,
    **kwargs,
):
    return ViT(
        input_shape=input_shape,
        classes=classes,
        num_layers=MODEL_CONFIGS["vit_H_16"]["num_layers"],
        dim=MODEL_CONFIGS["vit_H_16"]["dim"],
        mlp_dim=MODEL_CONFIGS["vit_H_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["vit_H_16"]["num_heads"],
        dropout_rate=MODEL_CONFIGS["vit_H_16"]["dropout_rate"],
        **kwargs,
    )


#Testing
model = vit_T_16(input_shape=(224, 224, 3), classes=1000)
print(model.summary())
