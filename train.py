import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def crossValidate(model, prediction, modelName, messages, splits):
    vect = CountVectorizer(analyzer='word', stop_words='english') #uses 1 - 3 word length grams ngram_range=(1, 3),
    #K fold validation
    kf = KFold(n_splits=splits)

    i = 0
    #loop through cross validated input
    for train_index, test_index in kf.split(messages["SentimentText"], messages["Sentiment"]):
        i = i + 1

        X_train, X_test = messages["SentimentText"][train_index], messages["SentimentText"][test_index]
        Y_train, Y_test = messages["Sentiment"][train_index], messages["Sentiment"][test_index]
        #length_train, length_test = messages["length"][train_index].as_matrix()[:,np.newaxis], messages["length"][test_index].as_matrix()[:,np.newaxis]
        #length_train, length_test = scipy.sparse.coo_matrix(length_train), scipy.sparse.coo_matrix(length_test)
    
        vect.fit(X_train)
        X_train_df = vect.transform(X_train)
        X_test_df = vect.transform(X_test)
        
        #add length feature to the feature vector
        #X_train_df = hstack([vect.transform(X_train), length_train])
        #X_test_df = hstack([vect.transform(X_test), length_test])
        model.fit(X_train_df, Y_train)
        if (i%10 == 0):
            print( "Iteration "+str(i)+": "+str(accuracy_score(Y_test, model.predict(X_test_df))))

    #really simple to create a model and train it.
    #h = scipy.sparse.coo_matrix(messages['length'].as_matrix()[:,np.newaxis])
    #X_df = hstack([vect.transform(messages["text"]), h])
	predict(model)
    #print out the accuracy score over all the messages, because we've k cross validated


def predict(model):
	messages = pd.read_csv('test.csv', encoding='latin-1')
	X_df = vect.transform(messages["SentimentText"])
	messages["Sentiment"] = model.predict(X_df)
	messages = messages[['ItemID', 'Sentiment']]
	messages.to_csv('./predict.csv', encoding='latin-1', index=False)


if __name__=='__main__':
	data = pd.read_csv('train.csv', encoding='latin-1')
	data['length'] = data['SentimentText'].map(len)

	X_train,X_test,y_train,y_test = train_test_split(data["SentimentText"],data["Sentiment"], test_size = 0.2, random_state = 10)
	vect = CountVectorizer(analyzer='word', stop_words='english')
	vect.fit(X_train)
	X_train_df = vect.transform(X_train)
	X_test_df = vect.transform(X_test)
	prediction = dict()

	crossValidate(MultinomialNB(), prediction, "Multnomial NB with 300 fold cross validation", data, 5000)



