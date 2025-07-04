## Introduction
This is a minimal implementation of the CLIP model proposed by OpenAI, using PyTorch. For all the gory 
details: [Learning Transferable Visual Models From Natural Language 
Supervision](https://arxiv.org/pdf/2103.00020)  
Attention maps adapted for ViT by following: [Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928)  
SigLIP loss from: [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/pdf/2303.15343)    

![clip_model](assets/clip_desc.png)   


The main idea behind CLIP is to map different modalities (e.g. images and their corresponding captions) into a common vector space. By doing so, the model learns to align _semantically_ similar pairs (e.g. an image and its _accurate_ description) while pushing away those that don’t match. This alignment is achieved by using a similarity matrix: the diagonal elements (representing matching pairs) are being forced via the loss function to have high similarity, while off-diagonal elements (representing non-matching pairs) are forced to have low similarity. (See animation below for a visual explanation.)   

<div align="center">
  <img src="assets/matrix.gif">
</div>
   
Practically speaking, CLIP leverages the information-rich features extracted from the transformers to capture the underlying semantics of each modality. In this implementation, I use a Vision Transformer (ViT) for processing images and a BERT-uncased model for handling text. Together with the similarity matrix, these transformer encoders allow the model to generalize across a diverse set of features. This method isn’t limited to images and text, it can be extended to other modalities, such as speech paired with text.  

This model can be trained as follows:
- **Projection Layer training only:** This requires to pre-extract image and text embeddings and then train 
small projection layers on these embeddings (Very fast training and ok results, only bottleneck is 
embedding extraction, not covered here).
- **Train everything:** This is our focus here in this project (i.e. backpropagate gradients to the transformers in a fine tuning fashion), this is slower - depends on 
your hardware, but yields best results.   

CLIP is capable of:
- **Zero-shot classification:** Predicting the category of an image without any task-specific training. ✅
- **Text-to-Image Retrieval:** Finding relevant images based on a text query. ✅
- **Image-to-Text Retrieval:** Searching for descriptive text based on an image. (🚧 todo: find test set descriptions)   


## General Requirements
- `Python >= 3.8`
- `Accelerate`
- `PyTorch`
- `Torchvision`
- `Transformers`
- `NumPy`
- `Matplotlib` 
- Other libraries: `tqdm`, `PIL`, `PyYAML`
- Dataset used: [`MS-COCO-17`](https://cocodataset.org/#download)
- CoCo labels used for Zero-Shot: [`coco-labels`](https://github.com/amikelive/coco-labels/tree/master)

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ntat/Lightweight_CLIP_model.git
   ```
2. Install dependencies via `pip`:
   ```bash 
   pip install -r requirements.txt
   ```
3. Download the dataset and adapt the paths in `config.yaml` 
4. If you have access to only one GPU, run the script with `python`:
   ```bash 
   python main.py
   ```
- If you have access to multiple GPUs, run the script with `accelerate`, specifying the number of processes `<N>`:
  ```bash 
   accelerate launch --num_processes <N> main.py
   ```
- If you have access to SLURM with multiple nodes and multiple GPUs adapt the `multi_node.sh`script to your cluster's config and run it as follows:
  ```bash 
   sbatch multi_node.sh
   ```
5. For inference, look into the `notebooks` section to see how to interact with the code. 

# Results
All results were obtained using the standard clip loss function. A trivial hyperparameter sweep was performed. Always adjust learning rates accordingly when you increase or decrease the number of GPUs.
## Text-to-Image Retrieval
Top-5 images retrieved from the test set given a text prompt.    
More results in `retrieval_result_pics` folder.
<div align="center">
  <figure>
    <img src="retrieval_result_pics/airplanes1.png" width="600">
  </figure>

  <figure>
    <img src="retrieval_result_pics/airplanes2.png" width="600">
  </figure>

  <figure>
    <img src="retrieval_result_pics/baseball_people.png" width="600">
  </figure>

  <figure>
    <img src="retrieval_result_pics/dogs_outdoors.png" width="600">
  </figure>

  <figure>
    <img src="retrieval_result_pics/descriptive_people.png" width="600">
  </figure>
</div>

---

## Zero-shot classification
We pick 5 pictures at random from the test set, and perform 0-shot classification.   
More results in `zero_shot_classification_results` folder.
<div align="center">
  <figure>
    <img src="zero_shot_classification_results/zero_out_6.png" width="600">
  </figure>

  <figure>
    <img src="zero_shot_classification_results/zero_out_10.png" width="600">
  </figure>

  <figure>
    <img src="zero_shot_classification_results/zero_out4.png" width="600">
  </figure>

  <figure>
    <img src="zero_shot_classification_results/zero_out_2.png" width="600">
  </figure>

  <figure>
    <img src="zero_shot_classification_results/zero_out_8.png" width="600">
  </figure>
</div>

# Discussion
- What is lightweight about this?  
  - With the current setup (batch=32, projection layer dimension=128) it can be trained in a few hours in a modern GPU with with 10-12GB of vram and still get solid results. And, it scales near linearly the more GPU compute is added.
- Can I have a huge batch size (like 32k)?  
  - No, and that's beyond the scope of this project. If you have the resources for something like this, you need to distribute the batch (along with loss function computations with regards to the similarity matrix etc) across multiple devices (that will be the most memory consuming part). Tip: with SigLip loss it is easier.   
- What worked best?  
  - 1\) Using all five available captions per training image, 2) Vanilla-CLIP loss + scheduling with Cosine Annealing Warm Restarts [in the (1,2) setting](assets/cos_warm_res.png), 3) Not decaying gains or biases. With this priority: 1) >> 2) > 3) - Data is the king!


