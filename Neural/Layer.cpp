#include "layer.h"

Layer::Layer(int id_layer, NeuralNetwork* net, unordered_map<string, double> parameters){
    _id_layer = id_layer;
	_net = net;
	_parameters = parameters;
	_type = static_cast<LayerType>(static_cast<int>(parameters["type"])); //erk
	_activation = static_cast<ActivationFunction>(static_cast<int>(_parameters["activation"]));
	initLayer();
}

Layer::~Layer(){
    for(Neuron* n : _neurons)
        delete n;
    _neurons.clear();
}

int Layer::getId() const{
	return _id_layer;
}

void Layer::initLayer(){
	_neurons.clear();
    // This line clears the _neurons vector, presumably to start with an empty neuron collection.
    
	if (_type == LayerType::STANDARD || _type == LayerType::INPUT)
	{
		_parameters["size"] += 1; // It increments the size parameter by 1 to account for a bias neuron for the next layer.
		_neurons.reserve(static_cast<int>(_parameters["size"])); // It reserves memory in the _neurons vector for the expected number of neurons.
		for (int i_neuron = 0; i_neuron < _parameters["size"]; ++i_neuron) {
            _neurons.push_back(new Neuron(i_neuron, this, _activation, i_neuron == _neurons.capacity()-1));
        }// A loop runs from 0 to _parameters["size"] - 1
         /*
            It creates a new Neuron object and adds it to the _neurons vector

            The Neuron object is initialized with an ID (i_neuron), a reference to the current layer (this), 
            an activation function (_activation), and a flag indicating whether it is a bias neuron (i_neuron == _neurons.capacity() - 1).
         */
			
	}
	else if (_type == LayerType::OUTPUT)
	{
		_neurons.reserve(static_cast<int>(_parameters["size"]));
		for (int i_neuron = 0; i_neuron < _parameters["size"]; ++i_neuron)
			_neurons.push_back(new Neuron(i_neuron, this, _activation));
	}
}

void Layer::clean(){
    for(Neuron* n : _neurons)
        n->clean();
}

void Layer::trigger(){
    for(Neuron* n : _neurons)
        n->trigger();
} // Trigger the neurons, see Neuron.cpp

void Layer::connectComplete(Layer *next){
    for(Neuron* n1 : _neurons)
        for(Neuron* n2 : next->_neurons)
        /*
            The -> operator is used to access a member function or variable of an object through a pointer. 
            In this case, you have a pointer next to a Layer object, and you are using next->_neurons to access 
            the _neurons member of the Layer object pointed to by next.
        */
			if(!n2->isBias())
				n1->addNext(n2);
}

vector<double> Layer::output(){
    vector<double> outputs; // This line declares a local vector called outputs that will store the computed output values of the neurons
    outputs.reserve(_neurons.size());
    /*
        This line reserves memory in the outputs vector to accommodate the same number of elements as there are neurons in the layer. 
        Reserving memory in advance can improve performance by avoiding frequent reallocation of memory as elements are added to the vector.
    */
    for(Neuron* n : _neurons)
    /*
        This is a range-based for loop that iterates through each element of the _neurons vector of the current layer (this->_neurons). 
        For each iteration, it assigns the pointer to a Neuron object to the local variable n
    */
        outputs.push_back(n->output());
        /*
            it calls the output() member function on the Neuron object n. This function computes and returns the output value of the neuron, 
            which is then added to the outputs vector using the push_back method
        */
    return std::move(outputs);
    /*
        Finally, the outputs vector containing the computed output values of all the neurons in the layer is returned. 
        The std::move function is used here to indicate that ownership of the outputs vector can be transferred efficiently, 
        although it doesn't make a significant difference in this case.
    */
}

const vector<Neuron *>& Layer::neurons() const{
    return _neurons;
    /*
        The & in the function signature const vector<Neuron *>& Layer::neurons() const indicates that the function neurons() 
        returns a reference to a constant vector of pointers to Neuron objects. Let's break down this function signature and 
        understand its purpose:

        const vector<Neuron *>&: This part of the signature specifies the return type of the function. It says that the function 
        returns a reference (&) to a constant (const) vector of pointers (Neuron *) to Neuron objects.

        vector<Neuron *>: This indicates that the function returns a vector containing pointers to Neuron objects.

        const: This means that the returned vector cannot be modified through the reference. 
        It ensures that the function does not modify the vector it's returning.
    */
}

void Layer::alterWeights(const vector<vector<double> >& weights){
    for(size_t i_neuron=0;i_neuron < weights.size(); ++i_neuron)
        _neurons[i_neuron]->alterWeights(weights[i_neuron]);
        /*
            ->: The arrow operator -> is used to access a member function or variable of an object through a pointer. 
            In this case, it's used to call a member function of the Neuron object pointed to by _neurons[i_neuron]
        */
}

void Layer::shiftBackWeights(const vector<vector<double> >& weights){
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		_neurons[i_neuron]->shiftBackWeights(weights[i_neuron]);
}

vector<vector<double*> > Layer::getWeights(){
	vector<vector<double*>> w; // This line declares a local variable w as a vector of vectors of double pointers (double*)
	w.reserve(_neurons.size());
    /*
        This line reserves memory in the w vector to accommodate the same number of elements as there are neurons in the layer. 
        Reserving memory in advance can improve performance by avoiding frequent reallocation of memory as elements are added to the vector.
    */
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		w.push_back(std::move(_neurons[i_neuron]->getWeights()));
        /*
            Inside the loop, it calls the getWeights() member function on each Neuron object _neurons[i_neuron]. 
            This function returns a vector of double* (pointers to double values) representing the weights of the neuron's connections. 
            The push_back method is used to add this vector of weights to the w vector. std::move() is used here to efficiently 
            transfer ownership of the weight vectors.
        */
	return std::move(w);
}

vector<vector<Edge*> > Layer::getEdges(){
	vector<vector<Edge*>> w;
	w.reserve(_neurons.size());
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		w.push_back(std::move(_neurons[i_neuron]->getEdges()));
	return std::move(w);
}

void Layer::randomizeAllWeights(double abs_value){
    for(Neuron* neuron : _neurons)
        neuron->randomizeAllWeights(abs_value);
}

string Layer::toString(){
    string str = "layer:" + to_string(_id_layer) + "\n";
    for (Neuron* neuron: _neurons){
        str += neuron->toString() + "\n";
    }
    return str;
}

void Layer::shiftWeights(float range){
	for(Neuron* neuron : _neurons)
		neuron->shiftWeights(range);
}

const unordered_map<string, double>& Layer::getParameters() const{
	return _parameters;
}

vector<vector<double>> Layer::getBackpropagationShifts(const vector<double>& target){
	vector<vector<double>> dw(_neurons.size());
    /*
        This line declares a local vector of vectors called dw where the backpropagation weight adjustment shifts will be stored. 
        It is initialized with a size equal to the number of neurons in the layer (_neurons.size()).
    */
	for (size_t i = 0; i < _neurons.size(); i++){
		Neuron* n = _neurons[i];
		dw[i] = n->getBackpropagationShifts(target);
	}
	return dw;
}

LayerType Layer::getType() const{
	return _type;
}

ActivationFunction Layer::getActivation() const{
	return _activation;
}