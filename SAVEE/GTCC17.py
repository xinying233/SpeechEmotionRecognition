from scipy.io import wavfile
from spafe.features.gfcc import gfcc
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np



gfcc_model = np.zeros((1,17))     # inital array of gfcc model
shape = np.zeros((1,7),dtype = int)    # record the number of frames of each sentiment

PATH = [os.path.join(os.getcwd(),"anger"),    # path of audio files
        os.path.join(os.getcwd(),"disgust"),
        os.path.join(os.getcwd(),"fear"),
        os.path.join(os.getcwd(),"happiness"),
        os.path.join(os.getcwd(),"neutral"),
        os.path.join(os.getcwd(),"sadness"),
        os.path.join(os.getcwd(),"surprise"),
        ]    

for index,x in enumerate(PATH):     # for each sentiment:
    for filename in os.listdir(x):  # for each file in each sentiment
       rate,audio = wavfile.read(os.path.join(x,filename))    
       gfcc_feat= gfcc(audio, fs = rate, num_ceps = 17,win_len = 0.020,nfft = 2048)
       gfcc_model = np.vstack((gfcc_model, gfcc_feat))    # add the new frame into gfcc model array
    shape[0,index] = gfcc_model.shape[0]    #record the number of frames of each sentiment
    print(shape)

# set up target  
labels = np.zeros((gfcc_model.shape[0]-1,))   
for index in range(6):
    labels[shape[0,index]-1:shape[0,index+1]-1] = index+1


gfcc_model = np.delete(gfcc_model,0, axis = 0) # delete the first inital 0 row

# print model and target
print(gfcc_model)
print(gfcc_model.shape)
print(labels.shape)


# save mfcc array and labels as npy file 
np.save('data17.npy',gfcc_model)
np.save('target17.npy',labels)



# load data and target
data = np.load('data17.npy')
target =np.load('target17.npy')

#Rescaling 13 MFCC features
scaler = MinMaxScaler()


sca_data = scaler.fit_transform(data)
print(sca_data)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(sca_data, target, test_size=0.2)

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)

# evaluate the performance
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
acc = metrics.accuracy_score(y_test, y_pred)

# print out confusion matrix
label = ["anger","disgust","fear","happiness","neutral","sadness","surprise"]


disp = plot_confusion_matrix(knn, X_test, y_test,
                                 display_labels=label,
                                 cmap=plt.cm.Blues,
                                 normalize="true")

plt.title("normalized confusion matrix with 13 MFCC features K = 5 \n overall accuracy: %1.2f"%acc)

print("normalized confusion matirx")
print(disp.confusion_matrix)

plt.show()








