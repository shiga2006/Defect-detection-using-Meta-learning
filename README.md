3D Printing Defect Detection using Prototypical Networks

👋 Introduction

Have you ever faced issues like warping or layer shifting while using a 3D printer? These kinds of defects are common in Fused Deposition Modeling (FDM), the most widely used 3D printing technique. In this project, we aim to build a smart and efficient system that detects and classifies printing defects in real-time — even when we have very few labeled images to train on!

That’s where Prototypical Networks come into play — a powerful few-shot learning technique that mimics how humans can learn new things from just a few examples.

🎯 Project Goal
To develop a lightweight image classification system that:

Classifies FDM 3D printed parts into defect categories using a small dataset

Learns defect patterns through prototypes (average embeddings) of support images

Can generalize well to new or unseen defect examples

🔍 What We Detect
Our system focuses on six major 3D printing defect types:

Defect Type	Description
✅ Normal	No defect; smooth and expected print quality
❌ Bed Not Stick	Initial layer doesn't adhere properly to the print bed
❌ Cracking	Visible breaks between printed layers, usually in taller prints
❌ Layer Shifting	Layers are misaligned due to mechanical disturbances
❌ Stringing	Thin, string-like threads appear between parts due to oozing
❌ Warping	Edges of the printed object curl or lift due to temperature imbalance

🧠 How It Works (Prototypical Networks in Simple Terms)
Input Images

We take images of 3D prints with known defect labels (support set) and one unknown image (query).

Feature Extraction using CNN

A convolutional neural network (CNN) extracts deep features (embeddings) from each image.

Prototype Formation

For each class (defect type), we calculate the mean embedding from the support images — this is the prototype of the class.

Query Matching

The system compares the embedding of the query image with all the class prototypes.

The class with the closest prototype (based on distance) is assigned to the query image.

Classification Output

The model returns the predicted defect type based on similarity.

🖼️ Visual Workflow
(You can replace this with the actual image you shared earlier)

🧪 Evaluation Metrics
We evaluate the model using:

Accuracy: Correct predictions vs total predictions

Precision, Recall, F1-score: For multi-class analysis

Confusion Matrix: Visual breakdown of predictions per class

🗂️ Dataset
Real images of FDM-printed parts with diverse defect types

Collected from print logs, online datasets, and internal experimentation

Small sample size to simulate few-shot learning

🔧 Tech Stack
Python, PyTorch

CNN-based feature extractor

Prototypical Networks for few-shot learning

Matplotlib, scikit-learn for evaluation and visualization

💡 Why Few-Shot Learning?
Traditional deep learning models require thousands of images per class, but in real-world 3D printing, we often have only a few examples of each defect. That’s why Prototypical Networks are ideal — they’re built to learn from limited data efficiently.

🤝 Use Cases
Early detection of print failures

Automated monitoring in industrial additive manufacturing

Cost and material savings by reducing failed prints

📈 Future Enhancements
Real-time deployment with cameras on printers

Incorporating time-series sensor data with vision

Extension to other 3D printing techniques (SLA, SLS, etc.)
