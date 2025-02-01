## Introduction
This is a minimal implementation of the CLIP model proposed by OpenAI, using PyTorch. 

This implementation is based on the famous CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
TODO: Try the SigLIP loss from: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/pdf/2303.15343)

![clip_model](assets/clip_desc.png)


The idea of CLIP is basically to map two different modalities that describe similar things (ie an image and its caption) into a shared vector space. Practically speaking, the model leverages the very rich descriptive power of transformers (here we use ViT for pics and bert-uncased for text, as our encoders for feature extraction), in order to make sense of how an image and its corresponding text align together, allowing it to generalize across different features it has seen. It can be extended to other modalities such as speech and text etc.

CLIP is capable of:
- **Zero-shot classification:** Predicting the category of an image without any task-specific training.
- **Text-to-Image Retrieval:** Finding relevant images based on a text query.
- **Image-to-Text Retrieval:** Searching for descriptive text based on an image.

