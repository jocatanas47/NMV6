from sklearn import datasets

data = datasets.load_breast_cancer()

ulaz = data.data
izlaz = data.target

from sklearn.model_selection import train_test_split
ulaz_trening, ulaz_test, izlaz_trening, izlaz_test = train_test_split(ulaz, izlaz,
                                                                      test_size=0.2,
                                                                      random_state=20)

# skaliranje - minmax scaler x_norm = (x - min)/(max - min)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(ulaz_trening)
ulaz_trening_norm = scaler.transform(ulaz_trening)
# test skup se pravimo da ne poznajemo pa skaliramo sa podacima za trenirajuci
ulaz_test_norm = scaler.transform(ulaz_test)

n_in = ulaz_test_norm.shape[1]
from keras import Sequential
from keras.layers import Dense
def make_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=n_in))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss='binary_crossentropy', metrics='accuracy')
    return model

# menjamo tip modela da bi radile funkcije iz scikit_learn-a
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=make_model, verbose=0)

# trazenje hiperparametara
# grid search - trazimo optimalnu kombinaciju dva parametra u mrezi njihovih vrednosti
# kombinacija parametara - batch_size i epochs
param = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 15]
}
from sklearn.model_selection import GridSearchCV
# KFold sluzi za validaciju
from sklearn.model_selection import KFold
grid = GridSearchCV(estimator=model, param_grid=param, cv=KFold())
grid = grid.fit(ulaz_trening_norm, izlaz_trening)
best_param = grid.best_params_
best_acc = grid.best_score_
print(best_param)
print(best_acc)

model = make_model()
model.fit(ulaz_trening_norm, izlaz_trening, epochs=best_param['epochs'], batch_size=best_param['batch_size'])
print(model.evaluate(ulaz_test_norm))