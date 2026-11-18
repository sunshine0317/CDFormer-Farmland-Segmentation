# CDFormer: A Dual-Branch CNN–Transformer Network With Semantic Decoupling and Contrast-Guided Feature Enhancement for Farmland Segmentation

This repository provides the official implementation of **CDFormer**, a dual-branch CNN–Transformer network for farmland segmentation from remote sensing imagery.

## 1. Overview

CDFormer integrates convolutional neural networks (CNNs) and Transformers to capture complementary local spatial details and global contextual information for accurate farmland segmentation.

The proposed network consists of three main components:

- **Adaptive Semantic Information Decoupling (ASID):** Explicitly separates high-level representations into foreground, background, and boundary features to alleviate semantic ambiguity.
- **Contrast-Guided Feature Enhancement (CGFE):** Exploits semantic differences among foreground, background, and boundary representations through parallel cross-attention pathways to enhance boundary discrimination.
- **Feature Fusion Module (FFM):** Integrates complementary CNN and Transformer representations through channel refinement and bidirectional cross-attention.

## 2. Network Architecture

CDFormer adopts a dual-branch architecture consisting of a ResNet-50-based CNN encoder and a MaxViT-based Transformer encoder.

The two branches extract complementary local and global representations, which are progressively integrated through the FFM. The ASID and CGFE modules further enhance semantic discrimination and boundary representation.

## 3. Datasets

Experiments are conducted on the following farmland segmentation datasets:

- FGFD
- FIT
- UAV-Farmland

**Note:** The datasets are not included in this repository. Users should obtain the datasets from their respective sources and organize them according to the data-loading implementation.

## 4. Requirements

The implementation is based on PyTorch.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Please ensure that the installed PyTorch and CUDA versions are compatible with your hardware and environment.

## 5. Repository Structure

```text
CDFormer-Farmland-Segmentation/
├── README.md
├── requirements.txt
├── train.py
└── test.py
```

Additional model definitions and utility modules are organized according to the released implementation.

## 6. Training

To train CDFormer, configure the dataset paths and training parameters in the corresponding implementation files.

Run:

```bash
python train.py
```

Additional command-line arguments can be specified according to the training script.

## 7. Testing

To evaluate the trained model, configure the testing dataset and checkpoint path, then run:

```bash
python test.py
```

## 8. Pretrained Models

Pretrained weights and model checkpoints will be provided separately when available.

## 9. Citation

If you find this work useful in your research, please cite the corresponding paper:

```text
Q. Bu et al.,
"CDFormer: A Dual-Branch CNN–Transformer Network With
Semantic Decoupling and Contrast-Guided Feature Enhancement
for Farmland Segmentation,"
IEEE Transactions on Geoscience and Remote Sensing.
```

## 10. Contact

For questions regarding the implementation, please open an issue in this repository or contact the authors.

