## Jordanian Currency Detection

## Description 🧰
This project aims to detect and count currency papers and coins from images, videos, or live camera feeds using a pre-trained model deployed on the Roboflow platform. The model detects various types of coins and currency papers and estimates their total value based on their classifications.

### Data Collection 🛒📸
- The primary source for gathering image data was self snapping.
- Approximately 100 to 150 images (before Augmentation) were systematically collected for each class.

### Preprocessing Steps 🔨🔧
1.📷 **Image Loading and Annotation:** Utilizing Roboflow, streamlined the process of loading and annotating images. Roboflow's intuitive interface allowed for efficient annotation of images, ensuring accurate labeling of objects of interest within the dataset. 
2.🖨📦 **Data Augmentation:** Leveraging Roboflow, augmented the dataset to 2738 images by employing techniques such as blurring, adjusting brightness, adding noise. This augmentation process enhances the diversity of the dataset, leading to improved model performance and robustness.


### Model Description 👨‍🏫
- **YOlO V7:** YOLOv7 builds upon the principles of previous YOLO versions by employing a single neural network to simultaneously predict bounding boxes and class probabilities for multiple objects within an image.
----
## Visualizations 📐🧬
![alt text](/images/Ex3.JPG)

![alt text](/images/Ex2.JPG)

![alt text](/images/Ex.JPG)


## Models Results 🧮
| Model                 | mAP      | Precision | Recall   |
|-----------------------|----------|-----------|----------|
| First try with aug    | 91.5%    | 89.3%     | 87.9%    |
| Try with aug and 100  | 77.1%    | 74.7%     | 74.3%    |
| check point new aug   | 93.8%    | 92.2%     | 93.1%    |
| Mid Night             | 92.7%    | 90.7%     | 88.2%    |
| **Exploding**             | **97.7%**    | **95.6%**    | **95.2%**    |
|Last but not Least     | 92.3%    | 86.6%     | 90.1%    |


## Testing model on robflow 🎪
[Exploding model](https://app.roboflow.com/morning/money-classifier-5/visualize/1)
[Mid Night](https://app.roboflow.com/money-qxzoi/money-classifier-3-all-data/visualize/1)
[check point new aug](https://app.roboflow.com/money-qxzoi/money-classifier-diff-spaces/visualize/3)


## Code Dependencies 📚
- OpenCV: Used for image and video processing.
- Roboflow: Provides access to pre-trained deep learning models for object detection.
- Inference SDK: Enables interaction with the Roboflow API for model inference.

## Usage Locally 💾
```python
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Run the main script:
```python
python main.py
```

To install the required dependencies, run:

```python
pip install -r requirements.txt
```

Follow the on-screen prompts to choose the input source (image, video, or camera) and provide the path to the corresponding file.
Contributing

# End
