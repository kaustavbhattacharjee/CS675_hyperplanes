import sys,random
from sklearn import svm


#Class definition
class random_hyperplane():
    def __init__(self, dataset, all_labels, k):
        self.dataset = dataset
        self.len = len(dataset)
        self.all_labels = all_labels
        self.label_only = []
        for i in sorted(self.all_labels):
            self.label_only.append(self.all_labels[i])
        self.train_data = []
        self.test_data = []
        for i in range(self.len):
            self.dataset[i].append(1.0) # adding 1 as instructed by the project description
            if self.all_labels.get(i) is not None:
                self.train_data.append(self.dataset[i])
            else:
                self.test_data.append(self.dataset[i])
        self.len2 = len(dataset[0])
        self.k = k

    def dp(a, b):
        dp = map(lambda x, y: x * y, a, b)
        return sum(dp)

    def sign(a): #For determing whether the data is positive or negative
        if a >= 0:
            return 1
        else:
            return -1

    def data_hyperplane(self):
        Z = []
        Z_bar = []
        for i in range(self.k):

            self.w_array = []
            for column in range(self.len2 - 1):
                self.w_array.append(random.uniform(-1, 1))

            temp_array = []
            for x in range(self.len - 1):
                temp_array.append(random_hyperplane.dp(self.dataset[x], self.w_array))

            w0 = random.uniform(min(temp_array), max(temp_array))
            self.w_array.append(w0)

            if Z == []:
                for row in range(len(self.train_data)):
                    Z.append(
                        [(1 + random_hyperplane.sign(random_hyperplane.dp(self.train_data[row], self.w_array))) / 2]) #(1+sign(zi))/2
            else:
                for row in range(len(self.train_data)):
                    Z[row].append(
                        (1 + random_hyperplane.sign(random_hyperplane.dp(self.train_data[row], self.w_array))) / 2)

            if Z_bar == []:
                for row in range(len(self.test_data)):
                    Z_bar.append(
                        [(1 + random_hyperplane.sign(random_hyperplane.dp(self.test_data[row], self.w_array))) / 2])
            else:
                for row in range(len(self.test_data)):
                    Z_bar[row].append(
                        (1 + random_hyperplane.sign(random_hyperplane.dp(self.test_data[row], self.w_array))) / 2)

        return Z, Z_bar

    def best_C(self, train, labels):
        random.seed()
        all_C = [.001, .01, .1, 1, 10, 100] #array of all C values 
        error_dict = {}
        test_error = 0
        for j in range(0, len(all_C), 1):
            error_dict[all_C[j]] = 0
        id_row = []
        for i in range(0, len(train), 1):
            id_row.append(i)
        number_of_splits = 10
        for x in range(0, number_of_splits, 1):

            train_new = []
            labels_new = []
            validation_data = []
            validation_labels = []

            random.shuffle(id_row)  # re-ordering the row numbers randomly
            #print(id_row)

            for i in range(0, int(.9 * len(id_row)), 1):
                train_new.append(train[i])
                labels_new.append(labels[i])
            for i in range(int(.9 * len(id_row)), len(id_row), 1):
                validation_data.append(train[i])
                validation_labels.append(labels[i])

            # Prediction with SVM linear kernel
            for j in range(0, len(all_C), 1):
                C = all_C[j]
                classifier = svm.LinearSVC(C=C) #LinearSVC
                classifier.fit(train_new, labels_new)
                predictor = classifier.predict(validation_data) #CV error

                error = 0
                for i in range(0, len(predictor), 1):
                    if (predictor[i] != validation_labels[i]):
                        error = error + 1

                error = error / len(validation_labels)
                error_dict[C] += error

                predictor_train = classifier.predict(train_new)  #check
                error_train = 0
                for i in range(0, len(predictor_train), 1):
                    if (predictor_train[i] != labels_new[i]):
                        error_train = error_train + 1

                error_train = error_train / len(labels_new)  # test error
                if(x==0): test_error = error_train

        bestC = 0
        minimum_error = 100
        keys = list(error_dict.keys())
        for i in range(0, len(keys), 1):
            key = keys[i]
            error_dict[key] = error_dict[key] / number_of_splits
            if (error_dict[key] < minimum_error):
                minimum_error = error_dict[key]
                bestC = key

        return bestC, minimum_error,test_error

    def prediction_original_data(self):
        bestC, minimum_error,test_error = self.best_C(self.train_data, self.label_only)
        classifier = svm.LinearSVC(C=100)
        classifier.fit(self.train_data, self.label_only)
        predictor = classifier.predict(self.test_data)
        return bestC, minimum_error, test_error,predictor

    def prediction_random_hyperplane_data(self):
        train, test = self.data_hyperplane()
        bestC, minimum_error, test_error = self.best_C(train, self.label_only)
        classifier = svm.LinearSVC(C=0.001) #max_iter=10000 taking too much time
        classifier.fit(train, self.label_only)
        predictor = classifier.predict(test)
        return bestC, minimum_error, test_error,predictor


#Data file reading

file_data = open(sys.argv[1]).readlines()
file_data = [line.split() for line in file_data]
file_data = [list(map(float, lines)) for lines in file_data]
#Label file reading
labels_train={}
with open(sys.argv[2]) as f:
   x = f.readline()
   while x != '':
       a = x.split()
       labels_train[int(a[1])] = int(a[0])
       x = f.readline()

k=int(sys.argv[3])
my_model=random_hyperplane(file_data,labels_train,k)
original_bestC,original_minimum_error,original_test_error,predictor1=my_model.prediction_original_data()
hyperplane_bestC,hyperplane_minimum_error,hyperplane_test_error,predictor2=my_model.prediction_random_hyperplane_data()

# All prints for the document
"""
print("Original data: Linear SVC best C = "+str(original_bestC)+", best CV error = "+str(original_minimum_error*100)+"%, test error = "+str(original_test_error*100)+"%")
print("Random hyperplane data:")
print("For k = "+str(k))
print("Linear SVC best C = "+str(hyperplane_bestC)+", best CV error = "+str(hyperplane_minimum_error*100)+"%, test error = "+str(hyperplane_test_error*100)+"%")
"""

value=[]
for i in range(len(file_data)):
    if labels_train.get(i) is None:
        value.append(i)

for i in range(len(value)):
    print(predictor2[i],value[i])





