# Vision Transformer

The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

![download](https://user-images.githubusercontent.com/88665786/221298818-ea06b9b4-d2c9-4633-b56f-35abb1c448ef.png)
