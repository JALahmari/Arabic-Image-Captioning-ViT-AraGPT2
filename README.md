# Arabic Image Captioning with ViT and AraGPT2

An Arabic image captioning system that combines a **Vision Transformer (ViT)** with **AraGPT2** to generate Arabic descriptions for images.

## Overview

This project explores image-to-text generation in Arabic by connecting a pretrained Vision Transformer with the Arabic GPT-2 model through a `VisionEncoderDecoderModel`.

The model takes an image as input and generates an Arabic caption describing its content.

## Model Architecture

**Image → ViT Encoder → Cross-Attention → AraGPT2 Decoder → Arabic Caption**

* **Encoder:** `google/vit-base-patch16-224-in21k`
* **Decoder:** `aubmindlab/aragpt2-base`
* **Framework:** PyTorch + Hugging Face Transformers
* **Generation:** Beam search with 4 beams
* **Maximum caption length:** 36 tokens

The ViT encoder was frozen during training while the decoder and newly added cross-attention components were trained.

## Dataset

The project uses the **Flickr8k dataset** together with Arabic image captions.

After preprocessing:

* **8,091 images**
* **24,184 Arabic captions**

The data was split by image to avoid overlap between training and evaluation sets:

| Split      | Images | Captions |
| ---------- | -----: | -------: |
| Train      |  6,472 |   19,353 |
| Validation |    809 |    2,413 |
| Test       |    810 |    2,418 |

Arabic captions were normalized by removing diacritics and applying basic character normalization before tokenization.

## Training

The model was trained for **4 epochs** using:

* Batch size: 8
* Gradient accumulation: 2
* Effective batch size: 16
* Learning rate: `5e-5`
* Weight decay: `0.01`
* Maximum target length: 36
* Random seed: 42

Training was performed using a **NVIDIA Tesla T4 GPU** in Kaggle.

## Image Enhancement Experiment

In addition to the baseline model, an image-enhancement experiment was conducted using:

* Autocontrast
* Contrast adjustment
* Color enhancement
* Sharpness enhancement

The purpose was to investigate whether improving image quality before encoding could improve Arabic caption generation.

## Results

The baseline model achieved the best BLEU-4 score:

| Experiment                        |    BLEU-1 |    BLEU-2 |    BLEU-3 |    BLEU-4 |
| --------------------------------- | --------: | --------: | --------: | --------: |
| ViT + AraGPT2                     | **0.407** | **0.271** | **0.175** | **0.113** |
| ViT + AraGPT2 + Image Enhancement |     0.407 |     0.268 |     0.168 |     0.104 |

The results suggest that the image-enhancement preprocessing used in this experiment did not improve caption-generation performance.

## Example

**Input:** Image of a dog in water

**Generated caption:**

> كلب ابيض واسود يقفز في الماء

The generated caption was compared with multiple human-written Arabic reference captions.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Vision Transformer (ViT)
* AraGPT2
* Pandas
* NumPy
* PIL
* NLTK
* BLEU evaluation

## Project Structure

```text
Arabic-Image-Captioning-ViT-AraGPT2/
│
├── Arabic-Image-Captioning-ViT-AraGPT2.ipynb
├── README.md
└── requirements.txt
```

## Future Improvements

* Fine-tune more of the vision encoder
* Experiment with larger Arabic language models
* Explore additional Arabic image-caption datasets
* Improve Arabic caption evaluation beyond BLEU
* Compare different vision-language architectures

## Author

**Joud Alahmari**
GitHub: [JALahmari](https://github.com/JALahmari)
