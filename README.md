# 🚗 Automatic Number Plate Recognition (ANPR) System

---

## 🧩 1. Problem Statement

With the rapid growth of vehicles, monitoring and managing traffic efficiently has become a challenge.  
Manual tracking of vehicle registration numbers is time-consuming and error-prone.  
The goal of this project is to automate **vehicle number plate detection and recognition** using computer vision and deep learning.

---

## 🎯 2. Objective

The main objective of this project is to:
- Detect vehicle number plates from images or video feeds.
- Extract and recognize the text (license number) using OCR.
- Build a **real-time web-based application** that can handle both image uploads and live webcam feeds.

---

## ⚙️ 3. Methodology

This ANPR system combines **object detection** and **optical character recognition (OCR)** into a unified pipeline:

1. **Detection:** YOLOv8 model detects and localizes number plates in input images.  
2. **Preprocessing:** Detected plate regions are cropped, converted to grayscale, and binarized to enhance text visibility.  
3. **Recognition:** EasyOCR extracts the alphanumeric text from the processed plate.  
4. **Display:** Detected and recognized results are displayed in a user-friendly Streamlit web app.

---

## 🧠 4. Technologies Used

- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **NumPy**
- **EasyOCR**
- **Streamlit**
- **Google Colab**
- **Ngrok / Cloudflared** *(for web deployment)*

---

## 📁 5. Dataset Description

The dataset consists of vehicle images and their corresponding YOLO annotations.

```
/content/drive/MyDrive/Final_project/dataset/
│
├── images/
│   ├── train/
│   └── test/
│
├── labels/
│   ├── train/
│   └── test/
│
└── data.yaml
```

**data.yaml**
```yaml
path: /content/drive/MyDrive/Final_project/dataset
train: train/images
val: test/images
nc: 1
names: ['Number Plate']
```

---

## 🧩 6. Model Development

The model was trained using **YOLOv8s** for number plate detection.

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/content/drive/MyDrive/Final_project/dataset/data.yaml",
    epochs=60,
    imgsz=1080,
    batch=16,
    name="plate_detector"
)
```

The trained model is saved at:
```
/content/runs/detect/plate_detector/weights/best.pt
```

### 🧮 Model Metrics
| Metric | Value |
|--------|--------|
| Precision | 0.875 |
| Recall | 0.696 |
| mAP@50 | 0.758 |
| mAP@50-95 | 0.446 |

---

## 🔍 7. OCR Integration

The **EasyOCR** library is used to extract alphanumeric text from detected plates.

### Preprocessing Techniques:
- Grayscale conversion  
- Otsu thresholding (binarization)  
- Contrast enhancement  

```python
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ocr_result = reader.readtext(binary)
```

This improves OCR accuracy and reduces noise for clearer text extraction.

---

## 💻 8. Web Application (Streamlit)

A custom **Streamlit app** was developed to provide an interactive interface for:
- Uploading images
- Using a webcam for live detection
- Displaying detected plates and recognized text

Run the app on Colab:
```bash
!streamlit run app.py & npx localtunnel --port 8501
```

Access the generated public URL to use the web app.

---

## 📊 9. Results

| Input Image | Detected Plate | OCR Output |
|--------------|----------------|-------------|
| ![plate1](assets/plate1.jpg) | ✅ | KL07BV4258 |
| ![plate2](assets/plate2.jpg) | ✅ | MH12AB9898 |

The system efficiently detects and recognizes vehicle plates with high precision on clear images.

---

## 🚀 10. Future Scope

- Improve OCR accuracy with character segmentation and deep-learning-based recognition.  
- Add **real-time video stream** support for multiple vehicles.  
- Integrate **database connectivity** for vehicle record management.  
- Explore **YOLOv10** for improved speed and accuracy.  

---

## 👨‍💻 11. Author

**Name:** Karthik muruganathem  
**Role:** Data Scientist   
**Tools Used:** Python, OpenCV, YOLOv8, Streamlit, EasyOCR  


---

## 🙏 12. Thank You Note

> Thank you for taking the time to explore this project!  
> Your feedback and suggestions are always welcome to improve the system further.  
> Built with ❤️ by **Karthik muruganathem**, combining technology and innovation for smarter traffic solutions.
