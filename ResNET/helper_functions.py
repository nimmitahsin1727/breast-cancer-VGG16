import torch
import numpy as np
import glob
import cv2 # For converting paths into data
import random # For shuffling the dataset
from torch.utils.data import Dataset, DataLoader # For creating dataloader
from sklearn.model_selection import train_test_split # For spliting the training data.

def calculate_accuracy(model, data_loader, device = 'cpu'):
    """ For calculating the accuracy on the data-loader
        Args:
            model: Trained model.
            data_loader: Data to check the accuracy on.
            device: Select the appropriate device.
        
        Return:


    """

    correct_pred, num_examples = 0, 0
    predicted_labels = None
    
    for i, (features, targets) in enumerate(data_loader):            
    
        features = features.to(device)
        targets = targets.to(device)

        features = torch.tensor(features, dtype=torch.float32)
        features = torch.transpose(features, 3, 1)#(x, 1, 3)
        targets = torch.tensor(targets, dtype=torch.float32)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
    
        num_examples += targets.size(0)

        targets = np.argmax( targets ,axis=1)
        
        ## For debugging
        #print('predicted_labels: ', predicted_labels[:,None].shape , 'targets: ', targets.shape, 'num_examples: ', num_examples)
        #print('predicted_labels: ', predicted_labels , 'targets: ', targets)
        
        correct_pred += (predicted_labels == targets).sum()
    
    print('correct_pred: ', correct_pred, ', num_examples: ', num_examples )
    
    if num_examples > 0:
        return correct_pred.float()/num_examples * 100, predicted_labels
    else:
        return 0, predicted_labels


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor an alternative of keras.utils.to_categorical
    adapted from https://stackoverflow.com/a/49217762

     Args:
        y: tensor
        num_classes: number of classes to adapt the tensor.
     
     Returns:
        A tensor of (y, num_classes)
    """

    return np.eye(num_classes, dtype='uint8')[y]


def prepare_data_for_resnet(path, data_size = 1000, batch_size = 128):
    """ For preparing the data for the ResNet implementation
    args:
        path: Data path on the disk
        data_size: Number of data to select for this model
        batch_size: For dataloader. To creat batches of the data

    return:
        train_loader: Training data 
        valid_loader: Validataion data
        test_loader: Testing data
    
    """

    #DATASET_PATH = './data/**/*.png'
    data = glob.glob(path, recursive=True)

    non_cancer_images = []
    cancer_images = []

    # Dividing the dataset as per their lables.

    for img in data:
        if img[-5] == '0' :
            if len(non_cancer_images) <= data_size:
                non_cancer_images.append(img)
                
        elif img[-5] == '1' :
            cancer_images.append(img)
            if len(cancer_images) > data_size:
                break
    
    #print(len(non_cancer_images), ', cancer_images: ', len(cancer_images))

    non_img_arr = []
    can_img_arr = []

    img_height, img_width = (224, 224) # Setting width and height as per the ResNET Requirements.

    for img in non_cancer_images:
            
        n_img = cv2.imread(img, cv2.IMREAD_COLOR)
        n_img_size = cv2.resize(n_img, (img_height, img_width), interpolation = cv2.INTER_LINEAR)
        non_img_arr.append([n_img_size, 0])
        
    for img in cancer_images:
        
        c_img = cv2.imread(img, cv2.IMREAD_COLOR)
        c_img_size = cv2.resize(c_img, (img_height, img_width), interpolation = cv2.INTER_LINEAR)
        can_img_arr.append([c_img_size, 1])

    ## Creating data and labels

    X = []
    y = []

    breast_img_arr = np.concatenate((non_img_arr, can_img_arr))
    random.shuffle(breast_img_arr)

    for feature, label in breast_img_arr:
        X.append(feature)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)

    #print(len(X), ', Y: ', len(y))

    ## splitting the data
    X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(X, y, train_size = 0.95)

    ## Creating a data Loader
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])
        
    valid_data = []
    for i in range(len(X_valid)):
        valid_data.append([X_valid[i], y_valid[i]])

    ## Creating dataloaders from the splitted data
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def split_data(X, y, train_size):
    """ Responsible for splitting the data into training testing and validation
        args:
            X: Data
            y: Labels of the data
            train_size: Training size for the split purpose.
        return:
            X_train: Data for training
            X_test: Data for testing
            X_valid: Data for validation
            y_train: labels for training
            y_test: labels for testing
            y_valid: labels for validation

    """

    ## Splitting the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = train_size, random_state = 7)

    ## Rate to divide the training into a training and validation.
    rate = 0.8
    num = int(X.shape[0] * rate)

    X_test = X_train[num:]
    X_train = X_train[:num]

    y_test = y_train[num:]
    y_train = y_train[:num]

    ## Changing the output dim for the loss function
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    y_valid = to_categorical(y_valid, 2)

    return X_train, X_test, X_valid, y_train, y_test, y_valid 