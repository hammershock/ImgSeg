## Image Segmentation on Cityscapes Dataset

This project implements image segmentation using U-Net architectures to handle the Cityscapes dataset. 

- torch-style implementation of Model Inference with ONLY Numpy!
  Just dive into the details about how the Network actually works.
- model training using pytorch.

### Requirements

- Python 3.8+
- PyTorch 1.7.0+ (for model training)
- OpenCV 4.5.2 (for image processing)
- NumPy 1.19.5
- tqdm
- Pillow (for image processing)
- TensorBoard (for model training)

### Structure

- `train.py`: Main script for training models on the Cityscapes dataset.
- `inference_numpy.py`: Inference script using a NumPy-based implementation of U-Net.
- `NumPyTorch/`: Submodule containing the NumPy-based neural network framework.

### Setup

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/hammershock/ImgSeg.git
   cd ImgSeg
   git submodule init
   git submodule update
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset

This project uses the [Cityscapes dataset](https://www.cityscapes-dataset.com/), which needs to be downloaded and structured as specified in the `train.py` script comments.

### Usage

#### Training
you should download your Cityscape Dataset first and modify the dataset dir in `train.py`.

Run the `train.py` script to start training:
```bash
python train.py
```

#### Inference

torch-style implementation of Model Inference with ONLY Numpy!
Just dive into the details about how the Network actually works.

Execute the `inference_numpy.py`:
```bash
python inference_numpy.py
```

### Contributing

Feel free to fork the project and submit pull requests. You can also send suggestions and report issues on the [GitHub repository issues page](https://github.com/hammershock/ImgSeg/issues).

---
