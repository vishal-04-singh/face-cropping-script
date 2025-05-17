# Face Detector and Auto-Cropper

This tool automatically detects faces in images and crops them with appropriate padding for portraits or profile pictures. It processes multiple folders of images in batch mode and saves the results in a structured output directory.

## Features

- **Face Detection**: Uses OpenCV's DNN face detector (preferred) or Haar Cascade as fallback
- **Facial Landmark Detection**: Improves cropping alignment when available
- **Batch Processing**: Process entire folders of images at once
- **Smart Cropping**: Optimized padding ratios for professional-looking portraits
- **Quality Preservation**: Maintains original image quality and format
- **High-Quality Resizing**: Uses multi-step approach for better quality when enlarging
- **Debug Mode**: Visualizes detection areas for troubleshooting

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy

## Installation

1. Clone or download this repository
2. Install required packages:
   ```
   pip install opencv-python numpy
   ```
3. (Optional) Download DNN face detection models for improved accuracy:
   - Create a `models` folder in the project directory
   - Download the model files:
     - `opencv_face_detector_uint8.pb`
     - `opencv_face_detector.pbtxt`
     - `lbfmodel.yaml` (for facial landmarks)

## Directory Structure

```
├── input_folders/
│   ├── folder1/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── folder2/
│   │   └── ...
│   └── ...
├── cropped_results/
│   ├── folder1/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── folder2/
│   │   └── ...
│   └── ...
└── models/
    ├── opencv_face_detector_uint8.pb
    ├── opencv_face_detector.pbtxt
    └── lbfmodel.yaml
```

## Usage

1. Place your images in folders inside the `input_folders` directory
2. Run the script:
   ```
   python face_crop.py
   ```
3. Cropped images will be saved to corresponding folders in `cropped_results`

### Command Line Options

- `--debug`: Show detection rectangles and save debug images
- `--high-quality` or `-hq`: Enable high-quality mode with larger output images (1200px min dimension)

## Configuration

You can modify these variables at the top of the script to customize behavior:

```python
min_dim = 800                # Minimum dimension of output images
side_padding_ratio = 0.9     # Padding on left/right sides
top_padding_ratio = 0.6      # Padding above face
bottom_padding_ratio = 0.8   # Padding below face
confidence_threshold = 0.8   # Minimum confidence for DNN detector
output_quality = 100         # JPEG output quality (1-100)
sharpen_amount = 0           # Amount of sharpening (0 = disabled)
```

## How It Works

1. The script scans all folders within `input_folders`
2. For each image, it:
   - Detects faces using DNN or Haar cascade
   - When available, uses facial landmarks for better alignment
   - Applies smart padding based on face dimensions
   - Ensures minimum output dimensions
   - Preserves original format (PNG/JPG) and quality
   - Saves to the corresponding output folder

## Troubleshooting

- **No faces detected**: Try adjusting the `confidence_threshold` to a lower value
- **Poor cropping**: Use `--debug` to visualize detection areas
- **Low quality results**: Use `--high-quality` mode or adjust `output_quality` and `min_dim`
- **Missing models**: The script will fall back to Haar cascade if DNN models are not available

## License

[MIT License](LICENSE)
