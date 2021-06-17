from keras.utils import plot_model
from keras.models import Sequential, Model, load_model
model_pickle_path = './model_checkpoints/model_sn_2018-03-15 21 30 42.h5'
model = load_model(model_pickle_path)
plot_model(model, './model_arch.png', show_shapes=True)
