# NeuralNetwork

Simple library for creating dense neural networks in c++

'''
  NeuralNetwork net;
  net.add(DenseLayer(784));
  net.add(DenseLayer(16));
  net.add(DenseLayer(10));
  net.build();
  
  net.train(inputs, labels);
  
  net.forward(inputs);
  
  vector<float> output = net.getOutput();
  float prediction = vectorToClass(output);
'''

Note: This only supports training on cpu and not planning on supporting gpu 
(Since this project was made for learning porpuses)
