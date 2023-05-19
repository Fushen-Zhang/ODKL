import collections
from os import stat
from pkgutil import get_data
from joblib import PrintTime
import tensorflow as tf
import numpy as np
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.layers import base
from tensorflow.python.framework import function
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from tensorflow.python.ops.gen_array_ops import matrix_diag_part
from tensorflow.python.ops import math_ops

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
    Define GP process as a RNN layer to be applied with OSTL
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
                                                    lambda: self.reduced(alpha,C,Q,M,k_x,e_x,q,r,gamma_x,input,count,flag)
                                               ))
        
        (M,Q,C,alpha,count,flag) = tf.cond(tf.greater(tf.reshape(count,[]),self.length_scale),  
                         lambda:self.clip_state(M,Q,C,alpha,count,flag), 
                         lambda:self.pass_state(M,Q,C,alpha,count,flag))
        
        variance = tf.concat([variance,variance_x],0)
        new_ew = jacobian(f_x,self._kernel) 
        new_eb = jacobian(f_x, self._bias) 

        return f_x,GPTuple(M,Q,C,alpha,variance,count,flag,new_ew,new_eb) 
    