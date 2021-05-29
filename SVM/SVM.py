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

labels = pd.read_csv("SVM/training_data.csv", index_col=0)
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

#plt.imshow(hog_image, cmap=mpl.cm.gray)
#plt.show()

###CREATE IMAGES FEATURES AND FALTTEN INTO A SINGLE ROW

def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

### LOOP OVER IMAGES TO PREPROCESS 

def create_feature_matrix(label_dataframe):
    features_list = []
    
    for img_id in label_dataframe.index:
        # load image
        img = get_image(img_id)
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)

    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(labels)

#### SCALE FEATURE MATRIX + PCA

# get shape of feature matrix
print('Feature matrix shape is: ', feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
crack_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
crack_pca = ss.fit_transform(crack_stand)
# look at new shape
print('PCA matrix shape is: ', crack_pca.shape)

### SPLITTING TRAINING AND TEST SETS

X = pd.DataFrame(crack_pca)
y = pd.Series(labels.crack.values)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.3,
                                                    random_state=1234123)

# look at the distrubution of labels in the train set
print(pd.Series(y_train).value_counts())

### TRAIN MODEL

# define support vector classifier
svc = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svc.fit(X_train, y_train)

# SCORE MODEL
# generate predictions
y_pred = svc.predict(X_test)
print(len(y_test),len(y_pred))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    y_actual = list(y_actual)
    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

TP,FP,TN,FN = perf_measure(y_test, y_pred)
print("TP:",TP)
print("FP:",FP)
print("TN:",TN)
print("FN:",FN)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)

### VISUALIZING RESULTS AND TRAINING ACCURACY

# predict probabilities for X_test using predict_proba
probabilities = svc.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()