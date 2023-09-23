from skimage.transform import resize
from sklearn import metrics, svm
from utils import preprocess_data, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams

# 1. Get the dataset
X, y = read_digits()

# Sizes to loop over
image_sizes = [(4, 4), (6, 6), (8, 8)]

# Data split sizes
train_size = 0.7
dev_size = 0.1
test_size = 0.2

for img_size in image_sizes:
    # 2. Resize the images
    X_resized = [resize(image, img_size) for image in X]
    
    # 3. Data splitting
    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X_resized, y, train_size=train_size, test_size=test_size, dev_size=dev_size)
    
    # 4. Data preprocessing (assuming necessary)
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    # 5. Train and Tune Hyperparameters
    h_params_combinations = get_hyperparameter_combinations(h_params)
    best_hparams, best_model, best_accuracy  = tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combinations)

    # 6. Evaluate model
    test_acc = predict_and_eval(best_model, X_test, y_test)
    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc = best_accuracy

    # Print results
    print(f"image size: {img_size[0]}x{img_size[1]} train_size: {train_size} dev_size: {dev_size} test_size: {test_size} train_acc: {train_acc:.2f} dev_acc: {dev_acc:.2f} test_acc: {test_acc:.2f}")