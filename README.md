# Vision Transformer

This repository is about an implementation of the research paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
" using Tensorflow.

The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder. In order to perform classification, the standard approach of adding an extra learnable “classification token” to the sequence is used.

# Model Architecture

![Architecture](https://github.com/IMvision12/ViT-tf/blob/main/images/arch.png)

# References

[1] ViT paper: https://arxiv.org/abs/2010.11929

[2] Official ViT Repo: https://github.com/google-research/vision_transformer
