import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base
from tensorflow.python.framework import function
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from tensorflow.python.ops import math_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

@function.Defun()
def derivative(x, current_gradient):
    grad = 1. - (tf.nn.tanh(x) * tf.nn.tanh(x))
    return current_gradient * grad

@function.Defun(grad_func=derivative)
def StepFunction(x):
    return tf.nn.relu(tf.sign(x))

def sigmoid_grad(x):
    return math_ops.sigmoid(x) * (1 - math_ops.sigmoid(x))
def tanh_grad(x):
    return 1 - math_ops.tanh(x) * math_ops.tanh(x)
def identity_grad(x):
    return tf.ones_like(x)
def relu_grad(x):
    return tf.where(x >= 0, tf.ones_like(x), tf.zeros_like(x))

_SNUELayerTuple = collections.namedtuple("SNUELayerStateTuple", ("Vm", "h", "dg_t", "dh_t", "ew_in", "ew_in_wo", "ew_rec", "ew_rec_wo", "eb", "eb_wo","ed", "ed_wo"))
@tf_export("nn.rnn_cell.SNUELayerStateTuple")
class SNUELayerStateTuple(_SNUELayerTuple):
  __slots__ = ()
  @property
  def dtype(self):
    (Vm, Vm_wog, h, overVth_woh, ew_in, ew_in_wo, ew_rec, ew_rec_wo, eb, eb_wo, ed, ed_wo) = self
    if Vm.dtype != h.dtype or Vm.dtype != Vm_wog.dtype or Vm.dtype != overVth_woh.dtype or Vm.dtype != ew_in.dtype or Vm.dtype != ew_rec.dtype or Vm.dtype != eb.dtype or Vm.dtype != ew_in_wo.dtype or Vm.dtype != ew_rec_wo.dtype or Vm.dtype != eb_wo.dtype:
       raise TypeError("Inconsistent internal state")
    return Vm.dtype

@tf_export("nn.rnn_cell.SNUELayer")
class SNUELayer(LayerRNNCell):
    def __init__(self, num_units, num_units_prev, activation=StepFunction, g = tf.nn.relu, reuse=None, name=None,
                 decay=0.8, trainableDecay=False, initVth=1.0, recurrent=True, initW=None, initH=None):
        super(SNUELayer, self).__init__(_reuse=reuse, name=name)
        self.input_spec = base.InputSpec(ndim=2) 
        self._num_units = num_units
        self._num_units_prev = num_units_prev
        self._activation = activation
        if activation == math_ops.tanh:
            self.h_d = tanh_grad
        elif activation == math_ops.sigmoid:
            self.h_d = sigmoid_grad
        elif activation == tf.identity:
            self.h_d = identity_grad
        elif activation == tf.nn.relu:
            self.h_d = relu_grad
        elif activation == StepFunction:
            self.h_d = lambda x: 1. - (tf.nn.tanh(x) * tf.nn.tanh(x))
        else:
            print('Only tanh, sigmoid and identity function are implemented!')
            self.h_d = None
        
        self.decay = decay
        self.initVth = initVth
        self.trainableDecay = trainableDecay
        self.recurrent = recurrent
        self.g = g
        
        if g == math_ops.tanh:
            self.g_d = tanh_grad
        elif g == math_ops.sigmoid:
            self.g_d = sigmoid_grad
        elif g == tf.identity:
            self.g_d = identity_grad
        elif g == tf.nn.relu:
            self.g_d = relu_grad
        else:
            print('Only tanh, sigmoid and identity function are implemented!')
            self.g_d = None
        
        self.initW = initW
        self.initH = initH

        self._state_size = SNUELayerStateTuple(self._num_units, self._num_units, (self._num_units, self._num_units), (self._num_units, self._num_units), 
                                               (self._num_units, self._num_units_prev, self._num_units), (self._num_units, self._num_units_prev, self._num_units),
                                               (self._num_units, self._num_units, self._num_units), (self._num_units, self._num_units, self._num_units),
                                               (self._num_units, self._num_units), (self._num_units, self._num_units),(self._num_units, self._num_units),(self._num_units, self._num_units)) #Vm, h, ew, eb
        
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units
    
    def zero_state(self, batch_size, dtype,ini_state=None):
        return SNUELayerStateTuple(Vm = tf.zeros((batch_size, self._num_units), dtype=dtype, name='SNUE_Vm'),
                                   h = tf.zeros((batch_size, self._num_units), dtype=dtype, name='SNUE_h'),
                                   dg_t = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_dg'),
                                   dh_t = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_dh'),
                                   ew_in = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='SNUE_ew_in'),
                                   ew_in_wo = tf.zeros(((batch_size, self._num_units, self._num_units_prev, self._num_units)), dtype=dtype, name='SNUE_ew_in_wo'),
                                   ew_rec = tf.zeros(((batch_size, self._num_units, self._num_units, self._num_units)), dtype=dtype, name='SNUE_ew_rec'),
                                   ew_rec_wo = tf.zeros(((batch_size, self._num_units, self._num_units, self._num_units)), dtype=dtype, name='SNUE_ew_rec_wo'),
                                   eb = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_eb'),
                                   eb_wo = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_eb_wo'),
                                   ed = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_ed'),
                                   ed_wo = tf.zeros((batch_size, self._num_units, self._num_units), dtype=dtype, name='SNUE_ed_wo'),
                                   )
        
    def grad(self, grad, state, inp, out, optimizer, apply): 
        op_list = []
        
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key or self._decay.name in key:
                m_grad = tf.einsum('bj,bjl->bl', grad, el)
            else:
                m_grad = tf.einsum('bj,bjkl->bkl', grad, el)
            
            m_grad = tf.reduce_sum(m_grad, 0)
            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)

            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], 
                        tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self.recurrent and self._recurrent_kernel.name in key and self._recurrent_kernel.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._recurrent_kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], 
                                       tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], 
                                       m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._bias]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], 
                                       tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
        
                if self._decay.name in key and self._decay.trainable:                   
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([mod_grad], [self._decay]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], 
                                       tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))

        return op_list
    
    def grad_v(self, grad, vars, state, optimizer, apply): 
        op_list = []
        return op_list
    
    def get_grad(self, grad, state, inp, out, apply):
        return_list = []
        var_list = []
        op_list = []
        return return_list, var_list, op_list

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_weights = inputs_shape[1].value
        
        self.eligibility_trace_dict = {}
        self.eligibility_trace_storage_dict = {}
        
        add_name = ''
        
        if self.initW is None:
            self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name, shape=[input_weights, self._num_units])
        else:
            self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name, shape=[input_weights, self._num_units], initializer=tf.constant_initializer(self.initW))
        
        self.eligibility_trace_dict.update({self._kernel.name: 'ew_in'})
        self.el_kernel_storage = self.add_variable(_WEIGHTS_VARIABLE_NAME + add_name + '_storage', shape=[input_weights, self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'ew_in': self.el_kernel_storage})
        
        if self.recurrent:
            if self.initH is None:
                self._recurrent_kernel = self.add_variable('kernel_h' + add_name, shape=[self._num_units, self._num_units])
            else:
                self._recurrent_kernel = self.add_variable('kernel_h' + add_name, shape=[self._num_units, self._num_units], initializer=tf.constant_initializer(self.initH))
            self.eligibility_trace_dict.update({self._recurrent_kernel.name: 'ew_rec'})
            self.el_rec_kernel_storage = self.add_variable('kernel_h' + add_name + '_storage', shape=[self._num_units, self._num_units], initializer=tf.zeros_initializer, trainable=False)
            self.eligibility_trace_storage_dict.update({'ew_rec': self.el_rec_kernel_storage})
    
        self._bias = self.add_variable(_BIAS_VARIABLE_NAME + add_name, shape=[self._num_units], initializer=tf.constant_initializer(self.initVth, dtype=self.dtype))
        self.eligibility_trace_dict.update({self._bias.name: 'eb'})
        self.el_bias_storage = self.add_variable(_BIAS_VARIABLE_NAME + add_name + '_storage', shape=[self._num_units], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'eb': self.el_bias_storage})
        
        if self.trainableDecay:
            self._decay = self.add_variable("decay" + add_name, shape=[self._num_units],initializer=tf.constant_initializer(self.decay, dtype=self.dtype))
            self.eligibility_trace_dict.update({self._decay.name: 'ed'})
            self.el_decay_storage = self.add_variable("decay" + add_name + '_storage', shape=[self._num_units], initializer=tf.zeros_initializer, trainable=False)
            self.eligibility_trace_storage_dict.update({'ed': self.el_decay_storage})
        else:
            self._decay = self.add_variable("decay" + add_name, shape=[self._num_units], initializer=tf.constant_initializer(self.decay, dtype=tf.float32), trainable=False)
    
        self.built = True

    def call(self, inputs, state, target):
        (Vm, h, dg_t, dh_t, ew_in, ew_in_wo, ew_rec, ew_rec_wo, eb, eb_wo,ed,ed_wo) = state
        h_t = h 
        Vm_t = Vm 
        Vm = tf.multiply(Vm, 1.0-h) 
        Vm = tf.multiply(Vm, self._decay) 
        Vm = tf.add(tf.matmul(inputs, self._kernel), Vm)  
        
        if self.recurrent:
            Vm = tf.add(tf.matmul(h, self._recurrent_kernel), Vm) 
            
        Vm_wog = Vm
        Vm = self.g(Vm_wog) 

        if self.g_d != None:
            dg = tf.stop_gradient(tf.matrix_diag(self.g_d(Vm_wog)))
        else:
            dg = tf.stop_gradient(batch_jacobian(Vm, Vm_wog)) 
        
        overVth = Vm - self._bias 
        out = self._activation(overVth) 
        out.set_shape(overVth.shape)
        
        if self.h_d != None:
            dh = tf.stop_gradient(tf.matrix_diag(self.h_d(overVth)))
        else:
            dh = tf.stop_gradient(batch_jacobian(out, overVth)) 
        
        if self.recurrent:            
            #Bias
            fpart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, (tf.einsum('bjk,kl->blj', dh_t, self._recurrent_kernel) +\
                    (tf.expand_dims(self._decay * (1 - h_t), 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0)
                      - tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t)))) 
            
            spart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, (tf.einsum('bjk,kl->blj', -dh_t, self._recurrent_kernel) +\
                     (tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t)))
                                                                 )
            new_eb_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', fpart, eb_wo) + spart) 
            new_eb = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dh, new_eb_wo) - dh) 
            
            #Input weights
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) * tf.expand_dims(tf.expand_dims(inputs, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_in_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_in_wo) + spart)
            new_ew_in = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_in_wo))
            
            #Recurrent weights
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) * tf.expand_dims(tf.expand_dims(h_t, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_rec_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_rec_wo) + spart)
            new_ew_rec = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_rec_wo))
                      
        else:
            fpart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, ((tf.expand_dims(self._decay * (1 - h_t), 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) - tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            
            # bias
            spart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, ((tf.expand_dims(self._decay * Vm_t, 2) * tf.expand_dims(tf.eye(self._num_units, self._num_units), 0) * dh_t))))
            new_eb_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', fpart, eb_wo) + spart)
            new_eb = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dh, new_eb_wo) - dh)
            
            # decay
            spart = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dg, tf.matrix_diag(Vm_t*(1.0-h_t))))
            new_ed_wo = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', fpart, ed_wo) + spart)
            new_ed = tf.stop_gradient(tf.einsum('bjk,bkl->bjl', dh, new_ed_wo))

            #Input weights
            a = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tf.eye(self._num_units, self._num_units), 1), 0) *tf.expand_dims(tf.expand_dims(inputs, 1), 3))
            spart = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dg, a))
            
            new_ew_in_wo = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', fpart, ew_in_wo) + spart) 
            new_ew_in = tf.stop_gradient(tf.einsum('bjk,bklm->bjlm', dh, new_ew_in_wo))
            
            #Recurrent weights
            new_ew_rec_wo = tf.stop_gradient(ew_rec_wo)
            new_ew_rec = tf.stop_gradient(ew_rec)

        new_state = SNUELayerStateTuple(Vm,out,dg,dh,new_ew_in,new_ew_in_wo,new_ew_rec,new_ew_rec_wo,new_eb,new_eb_wo,new_ed,new_ed_wo)
        return out, new_state
    
    

_GPTuple = collections.namedtuple("GPTuple", ("M","Q","C","alpha","variance","count","flag","ew","eb")) 
@tf_export("nn.rnn_cell.GPTuple")
class GPTuple(_GPTuple):
  __slots__ = ()
  @property
  def dtype(self):
    (ew,eb) = self
    if ew.dtype != eb.dtype:
       raise TypeError("Inconsistent internal state: %s vs %s vs %s vs %s vs %s" % (str(ew.dtype), str(eb.dtype)))
    return ew.dtype

class GPLayer(LayerRNNCell): 
    '''
    Define GP process as a RNN layer with state as the: 
    previous layers' output, which is the mapped features Z 
    Kernel: horizontal scale M
    Bias: vertical scale of the function sigma_f 
    '''
    def __init__(self,length_scale,initW,epsilon,sigma_0,units=1, reuse=None, name=None,initB=None):
        super(GPLayer, self).__init__(_reuse=reuse, name=name)
        self._num_units = units
        self._state_size =  ((1, None), (None, None),(None, None),(1, None),(None,1))
        self.length_scale = tf.convert_to_tensor(length_scale,dtype=tf.float32)
        self.initW = initW
        self.initB = initB 
        self.sigma_0 = sigma_0**2 
        self.epsilon = epsilon
        self.lr_inc_kernel = 0.0
        self.lr_inc_bias = 0.0
    
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_units
    
    def zero_state(self, batch_size,dtype=tf.float32,ini_state=None):
        state = []
        name_list = ["M","Q","C","alpha","variance","count","flag","ew","eb"]
        for i in range(len(ini_state)):
            state.append(tf.convert_to_tensor(ini_state[i],dtype = dtype,name=name_list[i]))
        return GPTuple(*state)

    def grad(self, grad, state, inp, out, optimizer, apply): 
        op_list = []
        for key in self.eligibility_trace_dict:
            el = getattr(state, self.eligibility_trace_dict[key])
            if self._bias.name in key:
                m_grad = tf.einsum('bj,bjk->bk', grad, el) 
            else: 
                m_grad = tf.einsum('bj,bjk->bk', grad, el) 
                
            m_grad = tf.reduce_sum(m_grad, 0)
            if apply:
                mod_grad = m_grad + self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]]
            else:
                mod_grad = tf.zeros_like(m_grad)
                
            with tf.control_dependencies([mod_grad]):
                if self._kernel.name in key and self._kernel.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([self.lr_inc_kernel*mod_grad], [self._kernel]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
                
                if self._bias.name in key and self._bias.trainable:
                    
                    if apply:
                        op_list.append(optimizer.apply_gradients(zip([self.lr_inc_bias*mod_grad], [self._bias]), finish=False))
                        op_list.append(tf.assign(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], tf.zeros_like(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]])))
                    else:
                        op_list.append(tf.assign_add(self.eligibility_trace_storage_dict[self.eligibility_trace_dict[key]], m_grad))
        
        return op_list
    
    def grad_v(self, grad, vars, state, optimizer, apply): 
        op_list = []
        return op_list
    
    def get_grad(self, grad, state, inp, out, apply): 
        return_list = []
        var_list = []
        op_list = []
        return return_list, var_list, op_list

    def build(self, inputs_shape):
        self.eligibility_trace_dict = {}
        self.eligibility_trace_storage_dict = {}
        self.output_storage_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        add_name = ''
        
        if type(self.initW) != np.ndarray:
            self._kernel = self.add_variable('GP/kernel' + add_name, shape=[1], initializer=tf.constant_initializer(self.initW))
        else:
            self._kernel = self.add_variable('GP/kernel' + add_name, shape=[1], initializer=tf.constant_initializer(self.initW))
        self.eligibility_trace_dict.update({self._kernel.name: 'ew'})
        self.el_kernel_storage = self.add_variable('GP/kernel_storage' + add_name, shape=[1], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'ew': self.el_kernel_storage})
        
        if type(self.initB) != np.ndarray:
            self._bias = self.add_variable('GP/bias' + add_name, shape=[1], initializer=tf.constant_initializer(np.random.normal(size=[1])))
        else:
            self._bias = self.add_variable('GP/bias' + add_name, shape=[1], initializer=tf.constant_initializer(self.initB))
        self.eligibility_trace_dict.update({self._bias.name: 'eb'})
        self.el_bias_storage = self.add_variable('GP/bias_storage' + add_name, shape=[1], initializer=tf.zeros_initializer, trainable=False)
        self.eligibility_trace_storage_dict.update({'eb': self.el_bias_storage})
            
        self.built = True

    def K(self,x1,x2,kernel,bias): 
        # prior kernel function rbf kernel is implemented
        square_x1 = tf.reduce_sum(tf.einsum('bi,bi->bi',tf.einsum('bi,i->bi',x1, kernel),x1), axis=-1) 
        square_x2 = tf.reduce_sum(tf.einsum('bi,bi->bi',tf.einsum('bi,i->bi',x2,kernel),x2), axis=-1) 
        ex = tf.expand_dims(square_x1, axis=-1)
        ey = tf.expand_dims(square_x2, axis=-2)
        xy = tf.einsum('ij,kj->ik', tf.einsum('bi,i->bi',x1,  kernel ), x2)
        Re_dis = tf.square(bias)*tf.exp(-0.5*(ex - 2.0 * xy + ey))
        return Re_dis

    def extended(self,alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag): 
        C_kx = tf.einsum('bij,jk->bik',C,k_x)
        s = tf.concat([C_kx,tf.ones([1,1,C_kx.shape[2]])],1)
        alpha = tf.concat([alpha,tf.zeros([1,1,1])],1) + q * s
        
        C = C + r * tf.einsum('bij,bkj->bik',C_kx,C_kx)
        C = tf.concat([C,r*C_kx],2)
        C_i = tf.transpose(tf.concat([r*C_kx , r*tf.ones([1,1,1])],1),perm=[0,2,1])
        C = tf.concat([C,C_i],1)
        
        inv_gamma = tf.math.pow(gamma_x,-1)
        Q = Q + inv_gamma* (tf.einsum('bij,bkj->bik',e_x,e_x))
        Q = tf.concat([Q,-inv_gamma*e_x],2)
        Q_i = tf.transpose(tf.concat([-inv_gamma*e_x,inv_gamma*tf.ones([1,1,1])],1),perm=[0,2,1])
        Q = tf.concat([Q,Q_i],1)
        M = tf.concat([M,tf.expand_dims(input,0)],1)
        count = count + 1
        flag = tf.zeros_like(flag) 
        return M, Q, C, alpha,count,flag

    def iniupd(self,alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag): 
        alpha = alpha + q
        C = C + r 
        Q = Q + gamma_x**(-1)
        count = count 
        flag = tf.zeros_like(flag) 
        return M, Q, C, alpha,count,flag

    def reduced(self,alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag): 
        C_kx = tf.einsum('bij,jk->bik',C,k_x)
        s = C_kx + e_x
        alpha = alpha + q * s
        C = C + r * tf.einsum('bij,bkj->bik',s,s)
        count = count
        flag = tf.zeros_like(flag) 
        return M, Q, C, alpha, count, flag

    def clip_state(self,M,Q,C,alpha,count,flag):
        epsilon = tf.math.divide(tf.abs(alpha),tf.diag_part(Q[-1]))
        idx = tf.argmin(epsilon)[0][0]
        alpha_s = alpha[0,idx,0] 
        c_s = C[0,idx,idx] 
        q_s = Q[0,idx,idx]
        C_s = tf.concat([C[:,:idx,idx:idx+1],C[:,idx+1:,idx:idx+1]],1)
        Q_s = tf.concat([Q[:,:idx,idx:idx+1],Q[:,idx+1:,idx:idx+1]],1)

        alpha_t = tf.concat([alpha[:,:idx,0:1],alpha[:,idx+1:,0:1]],1)
        C_t_l = tf.concat([C[:,:idx,:idx],C[:,idx+1:,:idx]],1)
        C_t_r = tf.concat([C[:,:idx,idx+1:],C[:,idx+1:,idx+1:]],1)
        C_t = tf.concat([C_t_l,C_t_r],2)

        Q_t_l = tf.concat([Q[:,:idx,:idx],Q[:,idx+1:,:idx]],1)
        Q_t_r = tf.concat([Q[:,:idx,idx+1:],Q[:,idx+1:,idx+1:]],1)
        Q_t = tf.concat([Q_t_l,Q_t_r],2)
        
        alpha = alpha_t - alpha_s* Q_s / q_s 
        C = C_t + c_s*tf.einsum('bij,bkj->bik',Q_s,Q_s) / (q_s*q_s) - (tf.einsum('bij,bkj->bik',Q_s,C_s) + tf.einsum('bij,bkj->bik',C_s,Q_s))/q_s
        Q = Q_t - tf.einsum('bij,bkj->bik',Q_s,Q_s) / q_s
        M = tf.concat([M[:,:idx,:],M[:,idx+1:,:]],1)
        count = count-1
        return (tf.stop_gradient(M),tf.stop_gradient(Q),tf.stop_gradient(C),tf.stop_gradient(alpha),count,flag)

    def pass_state(self,M,Q,C,alpha,count,flag):
        return (tf.stop_gradient(M),tf.stop_gradient(Q),tf.stop_gradient(C),tf.stop_gradient(alpha),count,flag)

    def call(self, input, state, target): 
        M,Q,C,alpha,variance,count,flag,ew,eb = state

        k_x_s = self.K(input,input,self._kernel,self._bias)
        k_x = tf.transpose(self.K(input, tf.squeeze(M,axis=0),self._kernel,self._bias))
        ew = jacobian(k_x, self._bias) 
        variance_x = self.sigma_0 + k_x_s + tf.squeeze(tf.einsum('bak,kj->baj',tf.einsum('ja,bjk->bak',k_x,C),k_x),axis=0)
        f_x = tf.squeeze(tf.einsum('bji,jk->bik',alpha,k_x),axis=1)
        
        e_x = tf.einsum('bij,jk->bik',Q,k_x)
        gamma_x = tf.stop_gradient(k_x_s - tf.squeeze(tf.einsum('bak,kj->baj',tf.einsum('ja,bjk->bak',k_x,Q),k_x),axis=0))  

        q = tf.stop_gradient((target-f_x)/(variance_x+1e-10))
        r = tf.stop_gradient(-1/(variance_x+1e-10))
        flag = flag + 1

        M,Q,C,alpha,count,flag = tf.cond(tf.reduce_all(tf.equal(alpha,tf.zeros_like(alpha))),
                                            lambda:self.iniupd(alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag),
                                                lambda:tf.cond(tf.greater(tf.reshape(gamma_x,[]),self.epsilon),
                                                    lambda: self.extended(alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag),
                                                    lambda: self.reduced(alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag)))
        
        (M,Q,C,alpha,count,flag) = tf.cond(tf.greater(tf.reshape(count,[]),self.length_scale),  
                        lambda:self.clip_state(M,Q,C,alpha,count,flag), 
                        lambda:self.pass_state(M,Q,C,alpha,count,flag))
        
        variance = tf.concat([variance,variance_x],0)
        new_ew = jacobian(f_x,self._kernel) 
        new_eb = jacobian(f_x, self._bias) 

        return f_x,GPTuple(M,Q,C,alpha,variance,count,flag,new_ew,new_eb) 
    