#include "edge.h"
#include "neuron.h"

/*
    An asterisk is used in C++ to declare a pointer. 
    Pointers allow you to refer directly to values in memory, and allow you to modify elements that would otherwise only be copied.
*/

Edge::Edge(Neuron *n, Neuron *nb, double w) :  _n(n), _nb(nb), _w(w)
{

}

Neuron *Edge::neuron() const
{
    return _n;
}

Neuron* Edge::neuronb() const
{
	return _nb;
}

double Edge::weight() 
{
    return _w;
    /*
        Returns the weight of this edge (_w)
    */
}

double* Edge::weightP()
{
    return &_w;
    /*
        Returns a pointer to the weight of this edge (_w)
    */
}


void Edge::propagate(double neuron_output)
{
	neuron()->addAccumulated(neuron_output * weight());
    /*
        It calls the addAccumulated method on the neuron() (presumably a method of the Neuron class), 
        passing in the product of neuron_output and the weight of the edge.
    */
}

void Edge::alterWeight(double w)
{
    _w = w;

}

void Edge::shiftWeight(double dw)
{
	dw *= LEARNING_RATE;
	_w += dw;
	_last_shift = dw;
}

void Edge::resetLastShift()
{
	_w -= _last_shift;
}

double Edge::getLastShift() const
{
	return _last_shift;
}


double Edge::backpropagationMemory() const
{
	return _backpropagation_memory;
}

void Edge::setBackpropagationMemory(double v)
{
	_backpropagation_memory = v;
}