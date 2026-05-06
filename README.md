Built a real-time yoga posture classification system using:

MediaPipe Holistic for full-body landmark detection (pose, hands, face)

OpenCV for video capture and preprocessing

Custom CNN trained on extracted keypoints for multi-class yoga pose classification
→ Architecture: 3 hidden layers, ReLU activations, dropout regularization, Softmax output
→ Achieved 97% classification accuracy, average latency <0.8s

The system runs efficiently on CPU-based devices with no external sensors, enabling scalable deployment for wellness, fitness, and rehabilitation use cases.

Tech Stack: Python, TensorFlow, OpenCV, MediaPipe, NumPy, Scikit-learn
