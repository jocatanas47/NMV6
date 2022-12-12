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
# pravimo svoj optimizator
from keras.optimizers import Adam
# hp - hiperparametri
def make_model(hp):
    model = Sequential()
    # optimizujemo broj neurona u skrivenom sloju
    no_units = hp.Int('units', min_value=3, max_value=15, step=2)
    # aktivaciju biramo na slucajni nacin iz liste
    act = hp.Choice('activation', values=['sigmoid', 'relu', 'tanh'])
    model.add(Dense(units=no_units, activation=act, input_dim=n_in))
    model.add(Dense(1, activation='sigmoid'))
    # konstanta obucavanja - nasumicno
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
    return model

import keras_tuner as kt
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5)
# overwrite da obrise sve iz prethodne iteracije jer cuva zauvek
tuner = kt.RandomSearch(make_model, objective='val_accuracy', max_trials=10, overwrite=True)
tuner.search(ulaz_trening_norm, izlaz_trening, epochs=50,
             validation_data=(ulaz_test_norm, izlaz_test), callbacks=[es])
best_hp = tuner.get_best_hyperparameters()[0]
best_no_units = best_hp['units']
best_act = best_hp['activation']
best_lr = best_hp['learning_rate']
print(best_no_units)
print(best_act)
print(best_lr)

model = tuner.hypermodel.build(best_hp)
model.fit(ulaz_trening_norm, izlaz_trening, epochs=50, validation_data=(ulaz_test_norm, izlaz_test), verbose=0)
print(model.evaluate(ulaz_test_norm, izlaz_test))