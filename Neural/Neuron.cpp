#include "neuron.h"
#include <algorithm>
#include "neuralnetwork.h"

Neuron::Neuron(int id_neuron, Layer* layer, ActivationFunction function, bool is_bias):
    _id_neuron(id_neuron),
    _layer(layer),
	_activation_function(function),
	_is_bias(is_bias)
{

}

Neuron::~Neuron(){
    for(Edge* e : _next){
        delete e;
    }
        
} // Destructor for the class

void Neuron::trigger(){
    for(Edge* e : _next){
        e->propagate(output());
    } // Propagates information from this neuron to its connected edges.
}

double Neuron::in(){
    return _accumulated;
}

double Neuron::output(){
    if(_is_bias){
        return 1;
    }

    if(_layer->getType() == LayerType::INPUT){
        return outputRaw();
        /*
            _layer is a pointer to an object, presumably of a class or type that has a member function called getType().

            When _layer->getType() is called, it effectively dereferences the _layer pointer and then accesses the getType() 
            member function of the object that _layer points to.
        */
    }

    //return random(-10, 10);
	if (_activation_function == ActivationFunction::LINEAR){
        return _accumulated;
    }
	if(_activation_function == ActivationFunction::RELU){
        return relu(_accumulated);
    }
	if (_activation_function == ActivationFunction::SIGMOID){
        return sigmoid(_accumulated);
    }
	return outputRaw();
}

double Neuron::outputDerivative(){
    if(_activation_function == ActivationFunction::LINEAR){
        return 1;
    }
    if(_activation_function == ActivationFunction::RELU){
        return relu_derivative(output());
    }
    if(_activation_function == ActivationFunction::SIGMOID){
        return sigmoid_derivative(outputRaw());
    }
    return _accumulated;
}

double Neuron::outputRaw(){
    return _accumulated;
}

void Neuron::clean(){
    setAccumulated(0);
}

void Neuron::addAccumulated(double v){
	//cout << this->_layer->getId() << ":" << this->getNeuronId() << " added " << v << " on " << _accumulated << endl;

    setAccumulated(_accumulated + v);
}

void Neuron::addNext(Neuron *n){
    _next.push_back(new Edge(n, this, random(-5, 5)));
	n->addPrevious(_next.back());
}

void Neuron::addPrevious(Edge* e){
	_previous.push_back(e);
}

int Neuron::getNeuronId() const{
    return _id_neuron;
}

void Neuron::setAccumulated(double v){
    _accumulated = v;
}
 
void Neuron::alterWeights(const vector<double>& weights){
    for(size_t i_edge=0; i_edge < weights.size(); ++i_edge){
        _next[i_edge]->alterWeight(weights[i_edge]);
        /* 
        _next: This is a vector of pointers to objects, likely instances of the Edge class. 
        Each element in this vector is a pointer to an Edge object. 

        ->: This is the arrow operator (->). It is used to access a member (function or variable) of an object through a pointer. 
        In this case, it is used to access a member function of the Edge object pointed to by the i_edge-th element of the _next vector.
        */
        // This is the indexing operator ([])
    }
}

vector<double*> Neuron::getWeights(){
	vector<double*> w;
	w.reserve(_next.size());
    // This line reserves memory in the vector w to accommodate the same number of elements as in the _next vector. 
    // This is done to improve performance by avoiding frequent reallocation of memory.
	for (size_t i_edge = 0; i_edge < _next.size(); ++i_edge)
		w.push_back(_next[i_edge]->weightP());
	return std::move(w);
}

vector<Edge*> Neuron::getEdges(){
	vector<Edge*> w;
	w.reserve(_next.size());
	for (size_t i_edge = 0; i_edge < _next.size(); ++i_edge)
		w.push_back(_next[i_edge]);
        /*
            Inside the loop, this line accesses the i_edge-th element of the _next vector, which is a pointer to an Edge object. It then calls the weightP() member function on that Edge object, which returns a pointer to the weight of the edge. This pointer is added to the w vector.
        */
	return std::move(w);
        /*
            Finally, the w vector containing pointers to edge weights is returned. The std::move() function is used here to indicate that ownership of the vector can be transferred efficiently, although it doesn't make a significant difference in this case.
        */
}

void Neuron::randomizeAllWeights(double abs_value){
    for(Edge* e : _next)
		e->alterWeight(random(-abs_value, abs_value));
}

string Neuron::toString(){
    string weights;
    for(Edge* e : _next)
        weights.append( to_string(e->weight()) + ",");
    string str =  "[" +  to_string(_layer->getId()) + "," + to_string(_id_neuron) + "]" +"("+ weights +")";
    return str;
}

void Neuron::shiftWeights(float range){
	for (Edge* e : _next)
		e->alterWeight(e->weight() + random(-range, range));
}

void Neuron::shiftBackWeights(const vector<double>& w){
	for (size_t i = 0; i < _previous.size(); i++)
		_previous[i]->shiftWeight(w[i]);
}

//gradient descent
vector<double> Neuron::getBackpropagationShifts(const vector<double>& target){
	vector<double> dw(_previous.size(),0);
    // This line declares a local vector called dw and initializes it with zeros. 
    // The size of this vector is set to _previous.size(), which is the number of incoming edges to the neuron.
	if (_layer->getType() == LayerType::OUTPUT){
    // This conditional statement checks if the layer to which the neuron belongs is of type LayerType::OUTPUT. 
    // If it is, it executes the code inside the if block.
		double d0 = output(); // Computes and stores the output of the neuron.
		double d1 = output() - target[this->getNeuronId()]; // Computes the difference between the neuron's output and the target output specified by the target vector.
		double d2 = outputDerivative(); // Computes the derivative of the neuron's output.
		for (size_t i = 0; i < _previous.size(); ++i){
            /*
                A loop iterates over the _previous vector (incoming edges) and computes adjustments (dw[i]) to their weights
                based on the values of d1, d2, and the output of the connected neurons.
            */
			dw[i] = (-d1*d2*_previous[i]->neuronb()->output());
			_previous[i]->setBackpropagationMemory(d1*d2);
            // The backpropagation memory of each incoming edge is updated using setBackpropagationMemory()
		}
		//cout << _layer->getId() << " " << d1 << " " << d2 << " " << d3 << " " << d1*d2*d3 << endl;

	}
	else 
	{
		double d = 0;
		for (size_t i = 0; i < _next.size(); i++){
            d += _next[i]->backpropagationMemory() * _next[i]->weight();
        }
        /*
            A loop iterates over the _next vector (outgoing edges) and accumulates the product of each edge's 
            backpropagation memory and weight.

            Backpropagation is a supervised learning algorithm used for training neural networks. During training, 
            the network computes an output based on the input data, and then it compares this output to the desired target output. 
            Any difference between the actual and target outputs is called an error or loss.
        */
			
		d *= outputDerivative();
        // The accumulated value d is then multiplied by the derivative of the neuron's output.
		for (size_t i = 0; i < _previous.size(); i++){
			_previous[i]->setBackpropagationMemory(d);
			dw[i] = -d * _previous[i]->neuronb()->output();
		}
        /*
            Another loop iterates over the _previous vector (incoming edges) and computes adjustments (dw[i]) to their weights 
            based on the value of d and the output of the connected neurons.

            The backpropagation memory of each incoming edge is updated using setBackpropagationMemory()
        */
	}
	return dw;
}

bool Neuron::isBias() const{
	return _is_bias;
}