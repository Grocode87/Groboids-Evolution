#from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
import numpy
#n = FeedForwardNetwork()

#inLayer = LinearLayer(3)
#hiddenLayer = SigmoidLayer(5)
#outLayer = LinearLayer(3)

#n.addInputModule(inLayer)
#n.addModule(hiddenLayer)
#n.addOutputModule(outLayer)

#in_to_hidden = FullConnection(inLayer, hiddenLayer)
#hidden_to_out = FullConnection(hiddenLayer, outLayer)

#n.addConnection(in_to_hidden)
#n.addConnection(hidden_to_out)

#n.sortModules()

#n.activate([1, 2, 5])

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(8, input_shape=(3,), ),
    Activation('tanh'),
    Dense(2),
    Activation('tanh'),
])

print(len(model.get_weights()))
for layer in model.get_weights():
    print(layer)

model.set_weights(model.get_weights())


t_array = numpy.array([[0, 0, 1],[1, 1, 0.5]])
print(t_array) 
print("Prediction:", model.predict(t_array))
