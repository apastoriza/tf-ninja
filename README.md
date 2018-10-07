# TensorFlow samples to support Ninja Talk 
A compilation of sample to introduce in the TersorFlow ecosystem

## hello
A hello world 

## tf101
TensorFlow concepts to get start
 
### model101
Tensor types: scalar, vector, matrix, cubes,... and how to create a tensor from NumPy

### session101
First TensorFlow session and first declarative execution

### session102
First tensor Variable

### session103
Multiple returns un a single session run

### session104
Using num session NumPy in session

### session105
Session boarding with TensorBoard: Optimizing a single neuron running a model/function.

## tf102
Feed-forward neural networks and classification.

### softmax101
If you want the outputs of a network to be interpretable as posterior
probabilities for a categorical target variable, it is highly desirable for
those outputs to lie between zero and one and to sum to one. 
The purpose of the softmax activation function is to enforce these constraints on the
outputs.  

In the example: 
 - We take an input of [1, 2, 3, 4, 1, 2, 3], the softmax of that is [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]. 
 - The output has most of its weight where the '4' was in the original input. 
 - This is what the function is normally used for: to highlight the largest values and suppress values which are significantly below the maximum value.

_NOTE: Raw softmax function in here. No tensorflow API in order tu fully understand._

### softmax101_numpy
A softmax sample using numpy. Inputs changed in order to **clarify that softmax is not scale invariant**.

### softmax102
The softmax function is often used in the final layer of a neural network-based classifier.
Using softmax classifier on [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits in a 
single layer of neurons

### softmax102_sigmoid
A _Softmax Normalization_. Use the Sigmoid function for normalization is a simple way to reduce the influence of extreme values 
(outlier) in the data without removing them from the dataset.

In the sample, A five layers network is built.
 - The data are non-linearly transformed using a sigmoidal functions as the output of every layer.
 - Last layer, classify using softmax


### softmax102_relu
Change the Sigmoid function with the Linear Rectifier (ReLU). ReLU is faster than Sigmoid because it does not require exponential 
calculus, wich is hard on computing. ReLU comes at a price and accuracy may penalty to pay. 
However ReLU but works fine with the right amount of data.   

### softmax102_relu_dropout
A _Dropout optimization_. One step further. Reduction of the weights to be updated in the learning phase. 
This technique limits the connected neurons with the next layer by setting weights to `0` so, 
neuron activation in the next layer is _dropped_.

Besides, learning rete is also a variable input.


# Sources used for this purpose
 - [ai-faq](http://www.faqs.org/faqs/ai-faq/)  
 - [Book: Deep Learning with TensorFlow by Packt](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow)
 - [Book: TensorFlow for Deep Learning](http://shop.oreilly.com/product/0636920065869.do)