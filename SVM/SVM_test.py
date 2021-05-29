### IMPORT NECESSARY LIBRARY
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc


def get_image(row_id, root="datasets/"):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    filename = "C:/Users/zhins/OneDrive/Desktop/Civil FYP/{}.jpg".format(row_id)
    file_path = os.path.join(root, filename)
    img = Image.open(file_path)
    return np.array(img)

labels = pd.read_csv("SVM/training_data1.csv", index_col=0)
labels.head()
crack_row = labels[labels.crack == 1].index[0]

#IMAGE MANIPULATION WITH RGB2GREY

crack = get_image(crack_row)
gray_crack = rgb2grey(crack)
#plt.imshow(gray_crack, cmap=mpl.cm.gray)
#plt.show()

### HISTOGRAM OF ORIENTED GRADIENTS

hog_features, hog_image = hog(gray_crack,
                              visualize=True,
                              block_norm='L2-Hys',
                              pixels_per_cell=(16, 16))

plt.imshow(hog_image, cmap=mpl.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()