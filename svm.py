import numpy as np
import cvxopt as cvx
import csv
import sklearn.metrics

FILENAME = "./winequality-white.csv"


class Kernel:
    def RBF(sigma: float):
        return lambda x1, x2: np.exp(-np.sqrt(np.linalg.norm(x1 - x2) ** 2 / (2 * sigma **2)))
    def lineral():
        return lambda x1, x2: np.dot(x1, x2)
    

def load_csv(file_name: str, split=False) -> np.array:
    data = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            if not row:
                continue
            data_row = []
            for i in row:
                data_row.append(float(i))
            data.append(np.array(data_row))
    return np.array(data)

def get_x_training(data_csv: np.array) -> np.array:
    X = []
    range_x = int(len(data_csv) * 0.8)
    for row in data_csv[:range_x]:
        X.append(np.delete(row,11))
    return X

def get_y_training(data_csv: np.array) -> np.array:
    Y = []
    range_x = int(len(data_csv) * 0.8)
    for row in data_csv[:range_x]:
        if row[-1] < 6:
            Y.append(-1)
        else:
            Y.append(1)
    return Y

def get_x_testing(data_csv: np.array) -> np.array:
    X = []
    range_x = int(len(data_csv) * 0.8)
    for row in data_csv[range_x:]:
        X.append(np.delete(row,11))
    return X

def get_y_testing(data_csv: np.array) -> np.array:
    Y = []
    range_x = int(len(data_csv) * 0.8)
    for row in data_csv[range_x:]:
        if row[-1] < 6:
            Y.append(-1)
        else:
            Y.append(1)
    return Y


class SVM:
    def __init__(self, kernel: str, kernel_parametr: float, c: float):
        if kernel == "lineral":
            self.kernel = Kernel.lineral()
        if kernel == "RBF":
            self.kernel = Kernel.RBF(kernel_parametr)
        self.c = c
        self.data = load_csv(FILENAME)
        self.X = np.array(get_x_training(self.data))
        self.Y = np.array(get_y_training(self.data))
        self.X_testing = np.array(get_x_testing(self.data))
        self.Y_testing = np.array(get_y_testing(self.data))
        self.N, self.N_features = self.X.shape
        self.bias = 0.0

    def kernel_matrix(self) -> np.array:
        kernel_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                kernel_matrix[i, j] = self.kernel(self.X[i], self.X[j])
        return kernel_matrix
    
    def get_alfa(self) -> np.array:
        K = self.kernel_matrix()
        P = cvx.matrix(np.outer(self.Y, self.Y) * K)
        q = cvx.matrix(np.ones(self.N) * -1)
        A = cvx.matrix(self.Y, (1, self.N), tc = 'd')
        b = cvx.matrix(0.0)
        G = cvx.matrix(np.vstack((np.diag(np.ones(self.N) * -1), np.identity(self.N))))
        h = cvx.matrix(np.hstack((np.zeros(self.N), np.ones(self.N) * self.c)))
        result = cvx.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.ravel(result['x'])
        return self.alphas
    
    def get_bias(self) -> float:
        self.bias = np.mean([y_i - self.predict(x_i) for y_i, x_i in zip(self.Y, self.X)])
        return self.bias
    
    def predict(self, x: np.array) -> float:
        result = self.bias
        for x_i, y_i, a_i in zip(self.X, self.Y, self.alphas):
            result += a_i * y_i * self.kernel(x_i, x)
        return result
    
    def test(self):
        Y = []
        for x in self.X_testing:
            if self.predict(x) < 0:
                Y.append(-1)
            else:
                Y.append(1)
        accuracy = sklearn.metrics.accuracy_score(self.Y_testing, Y)
        return accuracy


def main(kernel: str, kernel_parametr: float, c: float):
    svm = SVM(kernel, kernel_parametr, c)
    svm.get_alfa()
    svm.get_bias()
    accuracy = svm.test()
    print(f"Accuracy for kernel {kernel} with parametr {kernel_parametr} and c = {c} is {accuracy}")

if __name__ == "__main__":
    print(main("lineral", None, 1))






        


