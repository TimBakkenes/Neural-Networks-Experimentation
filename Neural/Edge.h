/* 
    Header Guards in C++ are conditional compilation directives that help to avoid errors 
    that arise when the same function or variable is defined more than once by the mistake of a programmer
*/

#ifndef EDGE_H
#define EDGE_H


#include <iostream>
/*
    This includes the standard C++ input/output library, which may be used for input and output operations.
*/


#include <vector>
/*
    This includes the C++ standard library's vector container, which is likely used to manage collections of objects.
*/
#include <limits>
/*
    This includes the C++ standard library's limits header, which may be used to obtain various numeric limits 
    (e.g., minimum and maximum values for data types).
*/


class NeuralNetwork;
class Neuron;
class Layer;
/*
    In C++, Forward declarations are usually used for Classes. In this, the class is pre-defined before its use 
    so that it can be called and used by other classes that are defined before this.

    Example explanation: Here the compiler throws this error because, in class B, the object of class A is being used,
    which has no declaration till that line. Hence compiler couldn’t find class A.
*/


using namespace std;
/*
    The std is a short form of standard, the std namespace contains the built-in classes and declared functions. 
    You can find all the standard types and functions in the C++ "std" namespace. There are also several namespaces inside "std."
*/


typedef unsigned int uint;
/*
    This defines an alias uint for the data type unsigned int, which can be used as a shorter name for unsigned int throughout the class.
*/


extern double LEARNING_RATE;
/*
    This declares a global variable LEARNING_RATE as a double. The extern keyword indicates that the actual definition of this variable is 
    expected to be found in another source file (not in this header).

    The extern keyword in C++ is used to declare a global variable or function which can be accessed from any part of the program or from 
    other files included in the program's header.

    "double" is a data type in the C++ programming language. "double" is used to represent double-precision floating-point numbers, 
    which are typically used to store real numbers with a decimal point. 
    It provides a higher degree of precision compared to the "float" data type, which is single-precision.
*/


class Edge{
/*
    This class represents an edge/connection between two neurons in a neural network
*/
    public:
    // All the public functions

        Edge(Neuron* n, Neuron* start, double w );
        /*
            Constructor that initializes an Edge object with pointers to two neurons (n and start) and a weight (w)
        */

            Neuron* neuron() const;
            /* 
                Returns a pointer to one of the neurons connected by this edge (_n)
            */

            Neuron* neuronb() const;
            /*
                Returns a pointer to the other neuron connected by this edge (_nb)
            */

            double weight();
            /*
                Returns the weight of this edge (_w)
            */

            double* weightP();
            /*
                Returns a pointer to the weight of this edge.
            */

        void propagate(double neuron_output);
        /*
            Propagates information through this edge, likely for forward propagation in the neural network
        */

        void alterWeight(double w);
        /*
            Changes the weight of this edge to the specified value (w)
        */

            void shiftWeight(double dw);
            /*
                Adjusts the weight of this edge by a specified amount
            */

            double getLastShift() const;
            /*
                Returns the last weight shift that occurred 
            */

            void resetLastShift();
            /*
                Resets the last weight shift value
            */

            double backpropagationMemory() const;
            /* 
                Returns a value related to backpropagation (presumably a memory or error value)
            */

            void setBackpropagationMemory(double v);
            /*
                Sets the backpropagation memory value to the specified value (v)
            */

        
    public:
    // All the public variables
	    Neuron* _n = nullptr; // Pointer to one of the neurons connected by this edge, nullptr = nullpointer
	    Neuron* _nb = nullptr; // Pointer to the other neuron connected by this edge
            double _w = 0.0; // Weight of this edge
	        double _last_shift = 0; // Stores the last weight shift that occurred

	        double _backpropagation_memory; // Presumably, a variable for storing information related to backpropagation

};



#endif // EDGE_H