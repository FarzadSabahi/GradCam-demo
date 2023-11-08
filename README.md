# Grad-CAM Visualization with TensorFlow

This Jupyter notebook demonstrates the use of Gradient-weighted Class Activation Mapping (Grad-CAM) with a pre-trained convolutional neural network using TensorFlow. It provides a visualization overlay on images to point out the important regions used for predicting the class of the image.

## Prerequisites

Before running the notebook, ensure you have the following libraries installed:

```bash
pip install tensorflow
pip install opencv-python
pip install matplotlib
```


## Using the Grad-CAM Implementation

The notebook includes a function to apply Grad-CAM to images in a specified directory. It loads the images, performs predictions using a pre-trained model, and overlays the generated heatmaps on the images to highlight the contributing regions.

```bash
# Example code snippet
from gradcam import GradCAM

# Instantiate the GradCAM class
# Assuming there is a function defined as GradCAM in the notebook
gradcam = GradCAM(model, target_layer)

# Generate heatmap
heatmap = gradcam.compute_heatmap(image)
