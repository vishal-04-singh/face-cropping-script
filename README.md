
# OpenCV DNN Face Detection

This project demonstrates how to use OpenCV's DNN (Deep Neural Network) module to detect faces in images using a pre-trained TensorFlow model.

## 📦 Features

- Uses OpenCV's DNN module for face detection
- Loads TensorFlow `.pb` model and `.pbtxt` configuration
- Detects faces in images with confidence scores

---

## 📁 Folder Structure

```
project/
├── models/
│   ├── opencv_face_detector_uint8.pb
│   └── opencv_face_detector.pbtxt
├── your_script.py
├── input.jpg
└── README.md
```

---

## 🧠 Model Files

This project uses a pre-trained face detector model from OpenCV:

- `opencv_face_detector_uint8.pb` (TensorFlow model)
- `opencv_face_detector.pbtxt` (Model configuration)

> You must download both files and place them in the `models/` folder.

### 🔗 Download Links

- [opencv_face_detector_uint8.pb](https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb?raw=true)
- [opencv_face_detector.pbtxt](https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detector/opencv_face_detector.pbtxt?raw=true)

---

## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/face-detection-dnn-opencv.git
cd face-detection-dnn-opencv
```

### 2. Install dependencies

```bash
pip install opencv-python
```

### 3. Run the script

```bash
python your_script.py
```

> Replace `your_script.py` with the actual filename.

---

## 📝 Notes

- Ensure your input image path is correct in the script.
- The model detects frontal faces with good accuracy and speed.
- Modify the script to work with live webcam feed, video, or multiple images.

---

## 🧑‍💻 Author

**Your Name**  
📧 Email: your.email@example.com  
🔗 LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the MIT License.
