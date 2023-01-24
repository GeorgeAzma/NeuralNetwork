# NeuralNetwork

Simple library for creating Dense Neural Networks in c++

```C++
  vector<vector<float>> inputs = loadImages("images");
  vector<vector<float>> labels = loadLabels("labels");

  NeuralNetwork net;
  net.add(DenseLayer(784));
  net.add(DenseLayer(16, ActivationFunctions::RELU));
  net.add(DenseLayer(16, ActivationFunctions::RELU));
  net.add(DenseLayer(10, ActivationFunctions::LINEAR));
  net.build();
  
  net.train(inputs, labels);
  
  net.forward(inputs);
  
  vector<float> output = net.getOutput();
  float prediction = vectorToClass(output);
  
  { //Save trained network (weights,biases...)
    ofstream save("save.net");
    save << net;
  }
  
  { //Load trained network (weights,biases...)
    ifstream save("save.net");
    save >> net;
  }
```

Note: This only supports training on cpu and not planning on supporting gpu 
(Since this project was made for learning porpuses)
