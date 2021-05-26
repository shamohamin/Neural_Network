import numpy as np

(STANDARD_SCALAR, MIN_MAX_SCALAR) = ("standard_scalar", "MIN_MAX_SCALAR")

def apply_k_fold(X_t: np.ndarray, y_t: np.ndarray, number_of_split=10) -> np.ndarray:
    X_temp = X_t.reshape(-1, X_t.shape[0]).copy()
    y_temp = y_t.reshape(-1, y_t.shape[0]).copy()
    dataset = np.hstack([X_temp, y_temp])
    np.random.shuffle(dataset)
    index = int(np.random.uniform(high=number_of_split))
    splited = np.split(dataset, number_of_split)
    X_valid_raw = splited[index]
    X_train_raw = np.zeros((1, X_t.shape[0] + y_t.shape[0]))
    for i in range(number_of_split):
        if i == index:
            continue
        X_train_raw = np.vstack([X_train_raw, splited[i]])
    X_train_raw = X_train_raw[1:, :].reshape(X_train_raw.shape[0] - 1, X_train_raw.shape[-1])
    X_train_or = X_train_raw[:, :X_t.shape[0]].reshape(X_t.shape[0], -1)
    y_train_or = X_train_raw[:, X_t.shape[0]:].reshape(y_t.shape[0], -1)
    X_valid_or = X_valid_raw[:, :X_t.shape[0]].reshape(X_t.shape[0], -1)
    y_valid_or = X_valid_raw[:, X_t.shape[0]:].reshape(y_t.shape[0], -1)
    return  X_train_or, y_train_or, X_valid_or, y_valid_or

def scaling(X: np.ndarray, method=STANDARD_SCALAR) -> np.ndarray or None:
    # first_shape = X.shape
    # try:
    #     X_cop = X.reshape(-1, 1).copy()
    # except Exception as ex:
    #     raise Exception("Shape of input must be (feature_counts, -1)")
    
    if method == STANDARD_SCALAR:    
        return (X - np.mean(X)) / np.std(X)
    elif method == MIN_MAX_SCALAR:
        return X / np.max(X.ravel())
    else:
        raise Exception("Scaling not supported !!!!")
    
def make_train_set(feature_list: list) -> np.ndarray:
    """
        make dataset from feature list
            feature_list: list of feature vectors
        returns:
            created dataset of features
    """
    feature_temp = []
    for feature in feature_list:
        if not isinstance(feature, np.ndarray):
            raise Exception("Illigal Argument passed")
        feature_temp.append(feature.reshape(1, -1))
    return np.vstack([*feature_temp])
        
    