# CIFAR-10 Object Detection & Prediction

This project is a deep learning application that uses a CNN trained on the **CIFAR-10 dataset** to classify objects into one of 10 categories:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

It provides:
- A **Jupyter Notebook (`CIFAR10_project.ipynb`)** for training and experimentation  
- A **Flask web app (`app.py`)** for serving predictions  
- A **frontend (`index.html`)** to upload images or provide an image URL and get predictions  

---

## Features
- Upload or provide URL for an image  
- Preprocessing with OpenCV (resize, normalize)  
- TensorFlow/Keras CNN model inference  
- Web interface built with Flask + Tailwind CSS  

---

## Project Structure
```
.
├── CIFAR10_project.ipynb   # Jupyter Notebook (model training)
├── app.py                  # Flask backend
├── index.html              # Frontend interface
├── requirements.txt        # Dependencies
└── README.md               # Documentation

```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Model
The Flask app expects a trained model file:
- `my_cnn_model.h5`  
You can either:
- Train it yourself using the provided notebook (`CIFAR10_project.ipynb`)  
- Or i have provided me pretrained model in this repo. 


---

## Example
- Upload or provide an image URL  
- The app will return one of the 10 CIFAR-10 class predictions  

---

## Requirements
See [requirements.txt](requirements.txt) for details.

