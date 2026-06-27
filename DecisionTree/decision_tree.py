import numpy as np

class DecisionTreeClassifier:
    def __itit__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def best_split(self, X, y):
        best_gain = -1
        n_features = X.shape[1]
        
        for i in range(n_features):
            X_column = X[:,i]
            x_values = np.unique(X_column)
            x_values = np.sort(x_values)
            thresholds = (x_values[:-1] + x_values[1:]) / 2

            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_threshold_idx = i
        return best_threshold_idx, best_threshold

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        proba = counts / len(y)
        entropy = -np.sum(proba * np.log2(proba))
        return entropy
    
    def split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
    def information_gain(self,y, X_column, threshold):
        # Calculate the information gain of parent node
        parent_entropy = self.entropy(y)

        # Split the data into left and right
        left_idxs, right_idxs = self.split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate entropy of left and right child nodes
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (len(left_idxs) / len(y)) * e_l + (len(right_idxs) / len(y)) * e_r

        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def fit(X, y, depth=0):
        
