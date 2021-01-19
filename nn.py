class NeuralNetwork: 
    import numpy as np
    def __init__(self,num_input_node,num_hidden_node,num_output_node,learning_rate=0.01): 
        self.num_input_node = num_input_node
        self.num_hidden_node = num_hidden_node
        self.num_output_node = num_output_node
        self.learning_rate = learning_rate 
        self.weight_for_in_hidden = self.np.random.normal(0.0, pow(num_input_node, -0.5),
                                            (self.num_hidden_node, self.num_input_node))
        self.weight_for_out_hidden =  numpy.random.normal(0.0, pow(self.num_hidden_node, -0.5), 
                                                 (self.num_output_node, self.num_hidden_node))
    def sigmoid(self,z):
        return expit(z)
    def train(self, inputs_list, targets_list):
        inputs = self.np.array(inputs_list, ndmin=2).T
        targets = self.np.array(targets_list, ndmin=2).T
        hidden_inputs = self.np.dot(self.weight_for_in_hidden,inputs)
        hidden_output = self.sigmoid(hidden_inputs)
        final_outputs   = self.sigmoid(self.np.dot(self.weight_for_out_hidden,hidden_output))
        output_errors = targets - final_outputs ##error for output layer [0,1,2]
        hidden_node_error = self.np.dot(self.weight_for_out_hidden.T,output_errors) 
        self.weight_for_out_hidden += self.learning_rate * self.np.dot((output_errors
                *final_outputs*(1.0-final_outputs)),self.np.transpose(hidden_output))
      
        self.weight_for_in_hidden += self.learning_rate * self.np.dot((hidden_node_error
                *hidden_output*(1.0- hidden_output)),self.np.transpose(inputs))    
        
    def query(self,inputs_list): 
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weight_for_in_hidden , inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = numpy.dot(self.weight_for_out_hidden , hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs
