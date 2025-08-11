# ToxiLens: Multimodal Model for Misogyny Detection

ToxiLens is a multimodal classification model designed for misogyny detection in Chinese-language memes. It integrates visual and textual information—especially span-level semantic features—via a gated fusion mechanism. The model achieves over 94% marco F1 score on a custom misogyny dataset created by Ping Du.

## Brief Introduction

- **Task**: Misogyny Detection on Multimodal Memes (Image + Text + Span)
- **Language**: Chinese
- **Data Source**: Ping Du's curated Chinese misogyny dataset
- **Goal**: Binary classification (`misogynistic` vs. `non-misogynistic`)
- **Performance**: >94% Macro F1 on test set

Full model details: [Model Summary](docs/model_summary.md)

### Core Components

- **CLIPEncoder**: Extracts image and caption features using a pre-trained CLIP model
- **SpanEncoder**: Extracts textual representations from spans using a RoBERTa-based model
- **PrefixUniformLayer**: Projects and aligns feature dimensions across modalities
- **Bi-Gated Fusion**: Selectively combines features with a two-way gating mechanism
- **MLP + Classifier**: Processes the fused vector and outputs classification logits
- **FocalLoss**: Enhances robustness against class imbalance

## Project Structure

```plaintext
ToxiLens/
├── data/                     # Input datasets (images, text, spans)
├── dataset/
│   └── dataset.py            # Custom Dataset & DataLoader
├── models/
│   ├── encoders/
│   │   ├── clip_encoder.py              # CLIP-based image-text encoder
│   │   ├── span_extract_model.py        # Span feature extractor
│   └── layers/
│   │   ├── prefix_uniform.py            # Multimodal projection & attention layer
│   │   ├── focal_loss.py                # Focal loss implementation
│   └── main_model.py                    # Main fusion model
├── scripts/
│   ├── preprocess/
│   │   └── image_cap.ipynb              # Captioning for images
│   ├── trainer/
│   │   └── train.py                     # Training script
│   ├── utils/
│   │   ├── logger.py                    # Logger helper
│   │   ├── clean_up.py                  # Dataset cleaning script
├── config.py                            # Model & training configuration
├── main.py                              # Entry point for training/predict/infer
├── LICENSE
├── .gitignore
└── README.md                            # This file
````

## Installation

```bash
git clone https://github.com/pandalow/ToxiLens.git
cd ToxiLens
pip install -r requirements.txt
```

You can also manually install the core dependencies listed below.

## Usage

### Training

```bash
python main.py --mode train --config config.py
```

### Prediction / Inference

```bash
python main.py --mode predict --image path/to/image.jpg --text "meme caption" --span "target span"
```

You can adjust model configuration and paths in `config.py`.

## Dependencies

```txt
numpy==1.23.2
pandas==2.2.3
Pillow==11.1.0
scikit_learn==1.2.2
torch==2.6.0+cu126
torchvision==0.21.0+cu126
tqdm==4.64.0
transformers==4.30.2
```

## Citation

If you use this model in your work, please cite:

```bibtex

```

## Acknowledgements

* CLIP model: OpenAI
* Transformers library: Hugging Face
* Dataset contributor: Ping Du

## TODO

* [ ] Add attention heatmap visualization
* [ ] Expand to multi-label classification (e.g., sexist, insulting, belittling)
* [ ] Add multilingual support (EN / CN / DE / NL)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
