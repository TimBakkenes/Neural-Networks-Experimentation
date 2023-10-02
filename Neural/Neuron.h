#ifndef NEURON_H
#define NEURON_H

#include "edge.h"
/*
    This includes another header file, likely defining the Edge class, which indicates that the Neuron 
    class is related to edges in a neural network.
*/
#include "../misc/functions.h"
#include "layer.h"
/*
    This includes another header file, presumably defining the Layer class
*/

#include <iostream>

#include <vector>
#include <limits>

class NeuralNetwork;
class Neuron;
class Layer;
using namespace std;

typedef unsigned int uint;


enum ActivationFunction{
	LINEAR,
	SIGMOID,
	RELU
};
/*
    Enumeration (enum ActivationFunction { ... };): This declares an enumeration called ActivationFunction, 
    which defines possible activation functions for neurons. The three possible values are LINEAR, SIGMOID, and RELU. 
    These are likely used to specify the activation function used by instances of the Neuron class.
*/

class Neuron{

    public:
        Neuron(int id_neuron, Layer* layer, ActivationFunction function = LINEAR, bool is_bias = false);

        ~Neuron();
        /*
            The ~Neuron(); declaration you see in the class definition is for the destructor of the Neuron class. 
            In C++, a destructor is a special member function that is automatically called when an object of the class is destroyed, 
            typically when it goes out of scope or when delete is called on a dynamically allocated object.
        */

        void trigger();

        double in();

        double output();

            double outputDerivative();

	        double outputRaw();

        void clean();

        void addAccumulated(double v);

            void addNext(Neuron* n);

	        void addPrevious(Edge* e);

        int getNeuronId() const;

        void setAccumulated(double v);

        void alterWeights(const vector<double>& weights);

	        vector<double*> getWeights();

	        vector<Edge*> getEdges();

	        void randomizeAllWeights(double abs_value);

        string toString();

	        void shiftWeights(float range);

	        void shiftBackWeights(const vector<double>& range);

	        vector<double> getBackpropagationShifts(const vector<double>& target);

	        bool isBias() const;

    public:
        Layer* _layer = NULL;
        int _id_neuron = 0;
        double _accumulated = 0.0;

            double _threshold = 0.0;
	        vector<Edge*> _next;
	        vector<Edge*> _previous;
	        ActivationFunction _activation_function;
	        bool _is_bias = false;
};

#endif // NEURON_H