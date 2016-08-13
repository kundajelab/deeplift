from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .core import *


class RNN(SingleInputMixin, Node):

    def __init__(self, hidden_states_exposed, reverse_input=False, **kwargs):
        self.reverse_input = reverse_input 
        self.hidden_states_exposed = hidden_states_exposed
        super(RNN, self).__init__(**kwargs) 

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(RNN, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['reverse_input'] = self.reverse_input
        kwargs_dict['hidden_states_exposed'] = self.hidden_states_exposed
        return kwargs_dict

    def _compute_shape(self, input_shape):
        #assumes there's an attribute called weights_on_x_for_h
        if (self.hidden_states_exposed):
            return (None, input_shape[1], self.weights_on_x_for_h.shape[1])
        else:
            return (None, self.weights_on_x_for_h.shape[1])

    def _build_fwd_pass_vars(self):
        """
           It is important that all the outputs of the Node have been
            built before the node is built, otherwise the value of
            mxts will not be correct 
        """
        self._build_fwd_pass_vars_core()

        #all the _through_time variables should be a list of tensors
        self._activation_vars,\
         self._hidden_state_activation_vars_through_time =\
          self._build_activation_vars(self._get_input_activation_vars())

        self._default_activation_vars,\
         self._default_hidden_state_activation_vars_through_time =\
          self._build_default_activation_vars()

        self._diff_from_default_vars = self._activation_vars\
                                        - self._default_activation_vars
        self._diff_from_default_hidden_vars_through_time =\
         [x - y for (x, y) in\
          zip(self._hidden_state_activation_vars_through_time,
              self._default_hidden_state_activation_vars_through_time)]

        #if the net's hidden vars were exposed,
        #then self._mxts has a time axis
        self._mxts = B.zeros_like(self._default_activation_vars)

    def _build_activation_vars(self, input_act_vars):
        self._initial_hidden_states = self._get_initial_hidden_states()
        #input_act_vars are assumed to have dims:
        # samples, time, ...
        hidden_states_activation_vars_through_time =\
                        B.for_loop(
                         step_function=self.forward_pass_step_function,
                         inputs=[RNN.flip_dimensions(input_act_vars)],
                         initial_hidden_states=self._initial_hidden_states,
                         go_backwards=self.reverse_input)
        return self.get_final_output_from_output_of_for_loop(
                     hidden_states_activation_vars_through_time),\
               hidden_states_activation_vars_through_time 

    @staticmethod
    def flip_dimensions(tensor):
        return B.dimshuffle(tensor, [1,0]+[x for x in xrange(2, tensor.ndim)])

    @staticmethod
    def flip_dimensions_and_reverse(tensor):
        return RNN.flip_dimensions(tensor)[::-1]

    def _get_mxts_increments_for_inputs(self):

        #First, prepare the initial hidden states for the backward pass

        #the first hidden variables for the backwards pass are the multipliers
        #flowing to the hidden state at t from the hidden state at t+1
        #in the beginning, this is zero as the hidden state at t+1 does not
        #exist
        backward_pass_initial_hidden_states =\
         [B.zeros_like(var) for var in self._initial_hidden_states]
        #the next hidden variables for the backwards pass are the multipliers
        #on the hidden state at time t+1, as well as the multipliers on the
        #inputs at time t+1. These are just outputs and are not used
        #in computation, and in the first iteration of the loop this also
        #refers to a time that does not exist, so we initialize with zeros  
        backward_pass_initial_hidden_states +=\
         [B.zeros_like(var) for var in self._initial_hidden_states]
        backward_pass_initial_hidden_states +=\
         [B.zeros_like(self._get_input_diff_from_default_vars()[:,0])]

        #Now prepare the inputs for the backward pass

        #shuffle dimensions to put the time axis first, then reverse the
        #order for the backwards pass
        slice_to_all_before_last = slice((1 if self.reverse_input else None),
                                         (None if self.reverse_input else -1))
        slice_to_all_after_first = slice((None if self.reverse_input else 1),
                                         (-1 if self.reverse_input else None))

        if (self.hidden_states_exposed):
            multipliers_from_above_to_hidden_states = self.get_mxts()
        else:
            #if the hidden state was not exposed, then it's only the last
            #timepoint that will have multipliers flowing into it. The rest
            #will be all zeros
            #It is assumed that the first entry in
            #self._hidden_state_activation_vars_through_time is the one
            #exposed to the rest of the net
            multipliers_from_above_to_hidden_states =\
             B.zeros_like(self._hidden_state_activation_vars_through_time[0])
            multipliers_from_above_to_hidden_states =\
             B.set_subtensor(multipliers_from_above_to_hidden_states[:,-1],
                            self.get_mxts())

        default_hidden_vars_tm1_list = []
        activation_hidden_vars_tm1_list = []
        for default_hidden_var, activation_hidden_var, initial_hidden_state in\
            zip(self._default_hidden_state_activation_vars_through_time,
                self._hidden_state_activation_vars_through_time,
                self._initial_hidden_states):
            assert self.reverse_input==False,\
             "The slicing code below is incorrect for reverse_input=True;"+\
             " it needs to be updated"
            #prepare default at t-1
            #the first position is fixed at the initial hidden states 
            default_hidden_var_tm1 = B.zeros_like(default_hidden_var)
            default_hidden_var_tm1 =\
             B.set_subtensor(default_hidden_var_tm1[:,1:],
                             default_hidden_var[:,:-1])
            default_hidden_var_tm1 =\
             B.set_subtensor(default_hidden_var_tm1[:,0], initial_hidden_state)
            #prepare activations at t-1
            activation_hidden_var_tm1 = B.zeros_like(activation_hidden_var)
            activation_hidden_var_tm1 =\
             B.set_subtensor(activation_hidden_var_tm1[:,1:],
                             activation_hidden_var[:,:-1])
            activation_hidden_var_tm1 =\
             B.set_subtensor(activation_hidden_var_tm1[:,0],
                             initial_hidden_state)
            #add to the list
            default_hidden_vars_tm1_list.append(default_hidden_var_tm1)
            activation_hidden_vars_tm1_list.append(activation_hidden_var_tm1)
                            
        inputs = [RNN.flip_dimensions_and_reverse(x) for x in
                   ([multipliers_from_above_to_hidden_states]+
                   activation_hidden_vars_tm1_list+
                   default_hidden_vars_tm1_list+
                   [self._get_input_activation_vars()]+
                   [self._get_input_default_activation_vars()])]
                   
        (multipliers_flowing_to_hidden_states,
         multipliers_on_hidden_states,
         multipliers_on_inputs) =\
                B.for_loop(
                 step_function=self.backward_pass_multiplier_step_function,
                 inputs=inputs,
                 initial_hidden_states=backward_pass_initial_hidden_states,
                 go_backwards=self.reverse_input)

        #reverse them through time
        multipliers_on_hidden_states = multipliers_on_hidden_states[:,::-1]
        multipliers_on_hidden_states = multipliers_on_hidden_states[:,::-1]
        multipliers_on_inputs = multipliers_on_inputs[:,::-1]

        self.multipliers_on_hidden_states = multipliers_on_hidden_states
        return multipliers_on_inputs

    def _get_initial_hidden_states(self):
        return [B.zeros((self._get_input_activation_vars().shape[0], #batch len
                         self.weights_on_x_for_h.shape[1]))] #num hidden units

    def forward_pass_step_function(self):
        """
            Reminder of the API:
                first arguments are inputs at time t, subsequent 
                 arguments are the hidden states after t-1
                 This is the function that will be passed
                 *directly* to theano.scan. Should return an array
                 of the hidden states after time t. If there are multiple
                 hidden states, the first one returned should be the output
        """
        raise NotImplementedError() 

    def get_final_output_from_output_of_for_loop(self, output_of_for_loop):
        """
            output_of_for_loop is like the output of theano.scan; it is a
             list of tensors, and each tensor has a first dimension which
             is time. This function decides how to extract the output of
             the net from the output of this for loop
        """
        raise NotImplementedError() 

    def backward_pass_multiplier_step_function(self):
        """
            API: the arguments provided are in the following order:
             - multipliers flowing to hidden state at time t from rest of net 
             - activation of hidden vars at time t-1
             - default value of hidden vars at time t-1
             - input activation vars at time t
             - default value of input activation vars at time t
             ^ (those are all passed in as inputs)
             - multipliers flowing to hidden state at time t from
               the hidden state at time t+1
             - multipliers on the hidden state at time t+1 (not used; output)
             - multipliers on inputs at time t+1 (not used; output)
             ^ (multipliers flowing to the hidden state at t-1,
                multipliers on the hidden state at time t (computed as a
                simple sum of the multipliers flowing from the next timestep
                and the multipliers flowing from the net above), and
                the multipliers on the input at time t are the outputs
                of the loop)
        """
        raise NotImplementedError()

    def get_final_output_from_output_of_for_loop(self, output_of_for_loop):
        if (self.hidden_states_exposed):
            return output_of_for_loop[0] 
        else:
            return output_of_for_loop[0][:,-1]


class RNNActivationsMixin(object):
    """
        just a class with some helper functions for setting
        gate_activation_name and hidden_state_activation_name
    """
    def set_activations(self, gate_activation_name,
                              hidden_state_activation_name,
                              **kwargs):
        self.gate_activation_name = gate_activation_name
        self.hidden_state_activation_name = hidden_state_activation_name
        self.gate_activation = RNNActivationsMixin.map_name_to_activation(
                                                   gate_activation_name)
        self.hidden_state_activation =\
                           RNNActivationsMixin.map_name_to_activation(
                            hidden_state_activation_name)

    def add_activation_kwargs_to_dict(self, kwargs_dict):
        kwargs_dict[self.gate_activation_name] = self.gate_activation_name
        kwargs_dict[self.hidden_state_activation_name] =\
                        self.hidden_state_activation_name

    @staticmethod
    def map_name_to_activation(activation_name):
        if (activation_name==ActivationNames.sigmoid):
            return B.sigmoid
        elif (activation_name==ActivationNames.hard_sigmoid):
            return B.hard_sigmoid
        elif (activation_name==ActivationNames.tanh):
            return B.tanh
        elif (activation_name==ActivationNames.relu):
            return B.relu
        elif (activation_name==ActivationNames.linear):
            return lambda x: x 
        else:
            raise RuntimeError("Unsupported activation:",activation_name)


class GRU(RNN, RNNActivationsMixin):
 
    def __init__(self, weights_lookup,
                 gate_activation_name,
                 hidden_state_activation_name, **kwargs):

        self.weights_on_x_for_z = weights_lookup['weights_on_x_for_z']
        self.weights_on_x_for_r = weights_lookup['weights_on_x_for_r'] 
        self.weights_on_x_for_h = weights_lookup['weights_on_x_for_h']
        
        self.weights_on_h_for_z =\
         weights_lookup['weights_on_h_for_z']
        self.weights_on_h_for_r =\
         weights_lookup['weights_on_h_for_r']
        self.weights_on_h_for_h =\
         weights_lookup['weights_on_h_for_h']

        self.bias_for_h = weights_lookup['bias_for_h']
        self.bias_for_z = weights_lookup['bias_for_z']
        self.bias_for_r = weights_lookup['bias_for_r']

        super(GRU, self).__init__(**kwargs) 
        super(GRU, self).set_activations(
                          gate_activation_name=gate_activation_name,
                          hidden_state_activation_name=
                           hidden_state_activation_name)

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(GRU, self).\
                       get_yaml_compatible_object_kwargs()
        super(GRU, self).add_activation_kwargs_to_dict(self, kwargs_dict)
        kwargs_dict['weights_lookup'] = OrderedDict([
            ('weights_on_x_for_z', self.weights_on_x_for_z),
            ('weights_on_x_for_r', self.weights_on_x_for_r),
            ('weights_on_x_for_h', self.weights_on_x_for_h),
            ('weights_on_h_for_z',
              self.weights_on_h_for_z),
            ('weights_on_h_for_r',
              self.weights_on_h_for_r),
            ('weights_on_h_for_h',
              self.weights_on_h_for_h),
            ('bias_for_z', self.bias_for_z),
            ('bias_for_r', self.bias_for_r),
            ('bias_for_h', self.bias_for_h)])
        return kwargs_dict

    def forward_pass_step_function(self, x_at_t, hidden_at_tm1):
        (hidden,
         proposed_hidden_through_1_minus_z_gate,
         hidden_at_tm1_through_z_gate,
         proposed_hidden,
         hidden_input_from_h,
         hidden_at_tm1_through_reset_gate,
         hidden_input_from_x,
         z_gate,
         z_input_from_h,
         z_input_from_x,
         r_gate,
         r_input_from_h,
         r_input_from_x) = self.get_all_intermediate_nodes_during_forward_pass(
                            x_at_t=x_at_t, hidden_at_tm1=hidden_at_tm1)
        return [hidden]

    def get_all_intermediate_nodes_during_forward_pass(self, x_at_t,
                                                       hidden_at_tm1):
        r_input_from_x = B.dot(x_at_t, self.weights_on_x_for_r)
        r_input_from_h = B.dot(hidden_at_tm1,
                               self.weights_on_h_for_r)
        r_gate = self.gate_activation(r_input_from_x
                                      + r_input_from_h
                                      + self.bias_for_r)

        z_input_from_x = B.dot(x_at_t, self.weights_on_x_for_z) 
        z_input_from_h = B.dot(hidden_at_tm1,
                               self.weights_on_h_for_z)
        z_gate = self.gate_activation(z_input_from_x
                                      + z_input_from_h 
                                      + self.bias_for_z)

        hidden_input_from_x = B.dot(x_at_t, self.weights_on_x_for_h)\
                                                      + self.bias_for_h
        hidden_at_tm1_through_reset_gate = r_gate*hidden_at_tm1
        hidden_input_from_h = B.dot(hidden_at_tm1_through_reset_gate,
                                    self.weights_on_h_for_h)
        proposed_hidden = self.hidden_state_activation(
                               hidden_input_from_x + hidden_input_from_h)
        hidden_at_tm1_through_z_gate = z_gate*hidden_at_tm1
        proposed_hidden_through_1_minus_z_gate = (1-z_gate)*proposed_hidden
        hidden = hidden_at_tm1_through_z_gate +\
                 proposed_hidden_through_1_minus_z_gate 
        return (hidden,
                proposed_hidden_through_1_minus_z_gate,
                hidden_at_tm1_through_z_gate,
                proposed_hidden,
                hidden_input_from_h,
                hidden_at_tm1_through_reset_gate,
                hidden_input_from_x,
                z_gate,
                z_input_from_h,
                z_input_from_x,
                r_gate,
                r_input_from_h,
                r_input_from_x) 

    def backward_pass_multiplier_step_function(self,
                                               mult_flowing_to_h_t_from_above,
                                               act_hidden_tm1,
                                               def_act_hidden_tm1,
                                               act_inp_vars_t,
                                               def_act_inp_vars_t,
                                               mult_flowing_to_h_t_from_h_tp1,
                                               mult_h_tp1,
                                               mult_inp_tp1):
        """
            API: the arguments provided are in the following order:
             - multipliers flowing to hidden state at time t from rest of net 
             - activation of hidden vars at time t-1
             - default value of hidden vars at time t-1
             - input activation vars at time t
             - default value of input activation vars at time t
             ^ (those are all passed in as inputs)
             - multipliers flowing to hidden state at time t from
               the hidden state at time t+1
             - multipliers on the hidden state at time t+1 (not used; output)
             - multipliers on inputs at time t+1 (not used; output)
             ^ (multipliers flowing to the hidden state at t-1,
                multipliers on the hidden state at time t (computed as a
                simple sum of the multipliers flowing from the next timestep
                and the multipliers flowing from the net above), and
                the multipliers on the input at time t are the outputs
                of the loop)
        """
        (act_hidden,
         act_proposed_hidden_through_1_minus_z_gate,
         act_hidden_at_tm1_through_z_gate,
         act_proposed_hidden,
         act_hidden_input_from_h,
         act_hidden_at_tm1_through_reset_gate,
         act_hidden_input_from_x,
         act_z_gate,
         act_z_input_from_h,
         act_z_input_from_x,
         act_r_gate,
         act_r_input_from_h,
         act_r_input_from_x) =\
         self.get_all_intermediate_nodes_during_forward_pass(
          x_at_t=act_inp_vars_t,
          hidden_at_tm1=act_hidden_tm1) 

        m_h_at_t = mult_flowing_to_h_t_from_h_tp1 +\
                   mult_flowing_to_h_t_from_above

        compute_multipliers_kwargs = {
            'm_h_at_t':m_h_at_t,
            'act_r_gate':act_r_gate,
            'act_z_gate':act_z_gate,
            'act_proposed_hidden':act_proposed_hidden,
            'act_hidden_tm1':act_hidden_tm1,
            'act_r_input_from_x':act_r_input_from_x,
            'act_r_input_from_h':act_r_input_from_h,
            'act_z_input_from_x':act_z_input_from_x,
            'act_z_input_from_h':act_z_input_from_h,
            'act_hidden_input_from_x':act_hidden_input_from_x,
            'act_hidden_input_from_h':act_hidden_input_from_h
        }

        use_conditional = True

        (same_x_m_hidden_at_tm1,
         same_x_m_h_at_t,
         same_x_m_x_at_t) = self.compute_multipliers(
            def_x_at_t=(act_inp_vars_t if use_conditional
                        else def_act_inp_vars_t),
            def_act_hidden_tm1=def_act_hidden_tm1,
            **compute_multipliers_kwargs)

        (same_h_m_hidden_at_tm1,
         same_h_m_h_at_t,
         same_h_m_x_at_t) = self.compute_multipliers(
            def_x_at_t=def_act_inp_vars_t,
            def_act_hidden_tm1=(act_hidden_tm1 if use_conditional else
                                 def_act_hidden_tm1),
            **compute_multipliers_kwargs)

        return [same_x_m_hidden_at_tm1, same_x_m_h_at_t, same_h_m_x_at_t]

    def compute_multipliers(self,
        def_x_at_t, def_act_hidden_tm1, 
        m_h_at_t, act_r_gate, act_z_gate,
        act_proposed_hidden, act_hidden_tm1,
        act_r_input_from_x, act_r_input_from_h,
        act_z_input_from_x, act_z_input_from_h,
        act_hidden_input_from_x, act_hidden_input_from_h):
        
        (def_act_hidden,
         def_act_proposed_hidden_through_1_minus_z_gate,
         def_act_hidden_at_tm1_through_z_gate,
         def_act_proposed_hidden,
         def_act_hidden_input_from_h,
         def_act_hidden_at_tm1_through_reset_gate,
         def_act_hidden_input_from_x,
         def_act_z_gate,
         def_act_z_input_from_h,
         def_act_z_input_from_x,
         def_act_r_gate,
         def_act_r_input_from_h,
         def_act_r_input_from_x) =\
         self.get_all_intermediate_nodes_during_forward_pass(
          x_at_t=def_x_at_t,
          hidden_at_tm1=def_act_hidden_tm1) 

        #experimental:
        #def_act_r_gate = 0.5*B.ones_like(def_act_r_gate)
        #def_act_z_gate = 0.5*B.ones_like(def_act_z_gate)
        #def_act_r_input_from_x = B.zeros_like(def_act_r_input_from_x)
        #def_act_r_input_from_h = B.zeros_like(def_act_r_input_from_h)
        #def_act_z_input_from_x = B.zeros_like(def_act_z_input_from_x)
        #def_act_z_input_from_h = B.zeros_like(def_act_z_input_from_h)
        
        diff_def_act_r_gate=(act_r_gate-def_act_r_gate)
        diff_def_act_z_gate=(act_z_gate-def_act_z_gate)
        diff_def_act_proposed_hidden=\
              (act_proposed_hidden-def_act_proposed_hidden)
        diff_def_act_hidden_tm1=(act_hidden_tm1-def_act_hidden_tm1)
        diff_def_act_r_input_from_x=\
             (act_r_input_from_x-def_act_r_input_from_x)
        diff_def_act_r_input_from_h=\
         (act_r_input_from_h-def_act_r_input_from_h)
        diff_def_act_hidden_input_from_x=\
         (act_hidden_input_from_x-def_act_hidden_input_from_x)
        diff_def_act_hidden_input_from_h=\
         (act_hidden_input_from_h-def_act_hidden_input_from_h)
        diff_def_act_z_input_from_x=\
         (act_z_input_from_x-def_act_z_input_from_x)
        diff_def_act_z_input_from_h=\
         (act_z_input_from_h-def_act_z_input_from_h)

        #hidden = hidden_at_tm1_through_z_gate +\
        #         proposed_hidden_through_1_minus_z_gate 
        #Therefore:
        m_proposed_hidden_through_1_minus_z_gate = m_h_at_t
        m_hidden_at_tm1_through_z_gate = m_h_at_t

        #proposed_hidden_through_1_minus_z_gate = (1-z)*proposed_hidden
        #Therefore, as per rule for products in the paper
        m_1_minus_z, m_proposed_hidden =\
                         distribute_over_product(
                          def_act_var1=(1-def_act_z_gate),
                          diff_def_act_var1=-1*diff_def_act_z_gate,
                          def_act_var2=def_act_proposed_hidden,
                          diff_def_act_var2=diff_def_act_proposed_hidden,
                          mult_output=m_proposed_hidden_through_1_minus_z_gate) 

        m_z_gate = -1*m_1_minus_z #this will be incremented later on

        #hidden_at_tm1_through_z_gate = z_gate*hidden_at_tm1
        #Therefore:
        #m_hidden_at_tm1 is going to get incremented later on, a lot
        incr_m_z_gate, m_hidden_at_tm1 =\
                         distribute_over_product(
                          def_act_var1=def_act_z_gate,
                          diff_def_act_var1=diff_def_act_z_gate,
                          def_act_var2=def_act_hidden_tm1,
                          diff_def_act_var2=diff_def_act_hidden_tm1,
                          mult_output=m_hidden_at_tm1_through_z_gate)
        m_z_gate += incr_m_z_gate
        
        #proposed_hidden = self.hidden_state_activation(
        #                       hidden_input_from_x + hidden_input_from_h)
        #Therefore:
        (m_hidden_input_from_x, m_hidden_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_hidden_input_from_x,
                                        diff_def_act_hidden_input_from_h],
          diff_def_act_output=diff_def_act_proposed_hidden,
          mult_output=m_proposed_hidden)

        #hidden_input_from_x = B.dot(x_at_t, self.weights_on_x_for_h)\
        #                                              + self.bias_for_h
        #Therefore:
        m_x_at_t = B.dot(m_hidden_input_from_x, self.weights_on_x_for_h.T) 

        #hidden_input_from_h = B.dot(hidden_at_tm1_through_reset_gate,
        #                            self.weights_on_h_for_h)
        #Therefore:
        m_hidden_at_tm1_through_reset_gate =\
         B.dot(m_hidden_input_from_h , self.weights_on_h_for_h.T)
        
        #hidden_at_tm1_through_reset_gate = r_gate*hidden_at_tm1
        #Therefore:
        m_r_gate, incr_m_hidden_at_tm1 =\
                         distribute_over_product(
                          def_act_var1=def_act_r_gate,
                          diff_def_act_var1=diff_def_act_r_gate,
                          def_act_var2=def_act_hidden_tm1,
                          diff_def_act_var2=diff_def_act_hidden_tm1,
                          mult_output=m_hidden_at_tm1_through_reset_gate)
        m_hidden_at_tm1 += incr_m_hidden_at_tm1

        #r_gate = self.gate_activation(r_input_from_x
        #                              + r_input_from_h
        #                              + self.bias_for_r)
        #Therefore:
        (m_r_input_from_x, m_r_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_r_input_from_x,
                                        diff_def_act_r_input_from_h],
          diff_def_act_output=diff_def_act_r_gate,
          mult_output=m_r_gate)
        
        #r_input_from_x = B.dot(x_at_t, self.weights_on_x_for_r)
        #Therefore:
        #+= is because m_x_at_t has been initialized before
        m_x_at_t += B.dot(m_r_input_from_x, self.weights_on_x_for_r.T)
        
        #r_input_from_h = B.dot(hidden_at_tm1,
        #                       self.weights_on_h_for_r
        #Therefore:
        m_hidden_at_tm1 += B.dot(m_r_input_from_h,
                                  self.weights_on_h_for_r.T)

        #z_gate = self.gate_activation(z_input_from_x
        #                              + z_input_from_h 
        #                              + self.bias_for_z)
        (m_z_input_from_x, m_z_input_from_h) =\
         compute_mult_for_sum_then_transform(
          diff_def_act_input_vars_list=[diff_def_act_z_input_from_x,
                                        diff_def_act_z_input_from_h],
          diff_def_act_output=diff_def_act_z_gate,
          mult_output=m_z_gate)

        #z_input_from_x = B.dot(x_at_t, self.weights_on_x_for_z) 
        #Therefore:
        m_x_at_t += B.dot(m_z_input_from_x, self.weights_on_x_for_z.T)

        #z_input_from_h = B.dot(hidden_at_tm1,
        #                       self.weights_on_h_for_z)
        #Therefore:
        m_hidden_at_tm1 += B.dot(m_z_input_from_h, self.weights_on_h_for_z.T) 

        return [m_hidden_at_tm1, m_h_at_t, m_x_at_t]
