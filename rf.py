from sklearn.ensemble import RandomForestRegressor
import os
import config
import sys
from process_data import get_dataset, train_test_validation_set_split
import _pickle as cPickle
from sklearn.ensemble import RandomForestClassifier


def model(x_train, x_test, x_valid, y_train, y_test, y_valid, channel):
    
    """
    print("started")
    regressor = RandomForestClassifier(n_estimators=1000, random_state=0)
    regressor.fit(x_train, y_train)
    """
    
    
    with open('rf_10', 'rb') as f:
        rf_10 = cPickle.load(f)

    with open('rf_100', 'rb') as f:
        rf_100 = cPickle.load(f)    

    with open('rf_1000', 'rb') as f:
        rf_1000 = cPickle.load(f)         


    y_pred_t = rf_10.predict(x_test)
    acc = rf_10.score(x_test,y_test)
    print("Accuracy of rf_10 model:", acc)    
    
    y_pred_t = rf_100.predict(x_test)
    acc = rf_100.score(x_test,y_test)
    print("Accuracy of rf_100 model:", acc)    
    
    y_pred_t = rf_1000.predict(x_test)
    acc = rf_1000.score(x_test,y_test)
    print("Accuracy of rf_1000 model:", acc)    


    """
    with open('rf_1000', 'wb') as f:
        cPickle.dump(regressor, f)
    print(regressor.feature_importances_)
    print("started")
    """


def soa(x,y):

    x_train, x_test, x_valid, y_train, y_test, y_valid = train_test_validation_set_split(x, y, config.train_ratio,
                                                                                         config.test_ratio,
                                                                                         config.validation_ratio)
  
    if not os.path.exists(config.models):
        os.mkdir(config.models)

    if not os.path.exists(config.graphs):
        os.mkdir(config.graphs)

    model(x_train, x_test, x_valid, y_train, y_test, y_valid,"G")


if __name__ == "__main__":

    x, y = get_dataset()
    
    config.num_classes = len(set(y))
    soa(x,y)
