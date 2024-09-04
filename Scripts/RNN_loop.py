from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.ops.rnn import _should_cache, _transpose_batch_time, _best_effort_input_batch_size, _maybe_tensor_shape_from_tensor, _infer_state_dtype

import sys
sys.path.append(".")
from Scripts.Neuron_models import SNUELayerStateTuple,GPTuple
_concat = rnn_cell_impl._concat

CELLNUM = 4

def _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, call_cell, state_size, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.

    Returns an (output, state) pair conditioned on `sequence_length`.
    When skip_conditionals=False, the pseudocode is something like:

    if t >= max_sequence_length:
      return (zero_output, state)
    if t < min_sequence_length:
      return call_cell()

    # Selectively output zeros or output, old state or new state depending
    # on whether we've finished calculating each row.
    new_output, new_state = call_cell()
    final_output = np.vstack([
      zero_output if time >= sequence_length[r] else new_output_r
      for r, new_output_r in enumerate(new_output)
    ])
    final_state = np.vstack([
      state[r] if time >= sequence_length[r] else new_state_r
      for r, new_state_r in enumerate(new_state)
    ])
    return (final_output, final_state)

    Args:
      time: int32 `Tensor` scalar.
      sequence_length: int32 `Tensor` vector of size [batch_size].
      min_sequence_length: int32 `Tensor` scalar, min of sequence_length.
      max_sequence_length: int32 `Tensor` scalar, max of sequence_length.
      zero_output: `Tensor` vector of shape [output_size].
      state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
        or a list/tuple of such tensors.
      call_cell: lambda returning tuple of (new_output, new_state) where
        new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
        new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
      state_size: The `cell.state_size` associated with the state.
      skip_conditionals: Python bool, whether to skip using the conditional
        calculations.  This is useful for `dynamic_rnn`, where the input tensor
        matches `max_sequence_length`, and using conditionals just slows
        everything down.

    Returns:
      A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
        final_output is a `Tensor` matrix of shape [batch_size, output_size]
        final_state is either a single `Tensor` matrix, or a tuple of such
          matrices (matching length and shapes of input `state`).

    Raises:
      ValueError: If the cell returns a state tuple whose length does not match
        that returned by `state_size`.
    """

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    # Vector describing which batch entries are finished.
    copy_cond = time >= sequence_length

    def _copy_one_through(output, new_output):
        # TensorArray and scalar get passed through.
        if isinstance(output, tensor_array_ops.TensorArray):
            return new_output
        if output.shape.ndims == 0:
            return new_output
        # Otherwise propagate the old or the new value.
        with ops.colocate_with(new_output):
            
            #If the state is not dependent on the batch_size, then just copy the content through
            if copy_cond.shape[0].value == output.shape[0].value:
                return array_ops.where(copy_cond, output, new_output)
            else:
                return new_output
            #return tf.cond(tf.equal(tf.shape(copy_cond)[0], tf.shape(output)[0]),
            #               lambda: array_ops.where(copy_cond, output, new_output),
            #               lambda: new_output)
            
            #return array_ops.where(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return control_flow_ops.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        #new_output, new_state = call_cell()
        new_output, new_state, new_outputs, old_state, old_outputs, inp = call_cell()
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        new_outputs = nest.flatten(new_outputs)
        old_state = nest.flatten(old_state)
        old_outputs = nest.flatten(old_outputs)
        inp = nest.flatten(inp)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        def empty_update(): return flat_zero_output + flat_state
        final_output_and_state = control_flow_ops.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        if not isinstance(substate, tensor_array_ops.TensorArray):
            substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    #return final_output, final_state
    return final_output, final_state, new_outputs, old_state, old_outputs, inp


def dynamic_rnn(cell, inputs, target_outputs, loss_function, gradient_function, train, store_LE_signal, use_BP, update_gradient_every, optimizer, 
                 global_grad=-1, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    Performs fully dynamic unrolling of `inputs`.

    Example:

    ```python
    # create a BasicRNNCell
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

    # defining initial state
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    # 'state' is a tensor of shape [batch_size, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                       initial_state=initial_state,
                                       dtype=tf.float32)
    ```

    ```python
    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, 256]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=data,
                                       dtype=tf.float32)
    ```
    Args:
      cell: An instance of RNNCell.
      inputs: The RNN inputs.
        If `time_major == False` (default), this must be a `Tensor` of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such
          elements.
        If `time_major == True`, this must be a `Tensor` of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such
          elements.
        This may also be a (possibly nested) tuple of Tensors satisfying
        this property.  The first two dimensions must match across all the inputs,
        but otherwise the ranks and other shape components may differ.
        In this case, input to `cell` at each time-step will replicate the
        structure of these tuples, except for the time dimension (from which the
        time is taken).
        The input to `cell` at each time step will be a `Tensor` or (possibly
        nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
      sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
        Used to copy-through state and zero-out outputs when past a batch
        element's sequence length.  So it's more for performance than correctness.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to "rnn".

    Returns:
      A pair (outputs, state) where:

      outputs: The RNN output `Tensor`.

        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.

        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.

        Note, if `cell.output_size` is a (possibly nested) tuple of integers
        or `TensorShape` objects, then `outputs` will be a tuple having the
        same structure as `cell.output_size`, containing Tensors having shapes
        corresponding to the shape data in `cell.output_size`.

      state: The final state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
        If it is a (possibly nested) tuple of ints or `TensorShape`, this will
        be a tuple having the corresponding shapes. If cells are `LSTMCells`
        `state` will be a tuple containing a `LSTMStateTuple` for each cell.

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If inputs is None or an empty list.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)

    with vs.variable_scope(scope or "rnn") as varscope:
        # Create a new scope in which the caching device is either
        # determined by the parent scope, or is set to place the cached
        # Variable using the same placement as for the rest of the RNN.
        if _should_cache():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        # By default, time_major==False and inputs are batch-major: shaped
        #   [batch, time, depth]
        # For internal calculations, we transpose to [time, batch, depth]
        flat_input = nest.flatten(inputs)
        flat_target_output = nest.flatten(target_outputs)

        if not time_major:
            # (B,T,D) => (T,B,D)
            flat_input = [ops.convert_to_tensor(
                input_) for input_ in flat_input]
            flat_input = tuple(_transpose_batch_time(input_)
                               for input_ in flat_input)

            flat_target_output = [ops.convert_to_tensor(
                target_) for target_ in flat_target_output]
            flat_target_output = tuple(_transpose_batch_time(
                target_) for target_ in flat_target_output)

        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size, "
                    "but saw shape: %s" % sequence_length.get_shape())
            sequence_length = array_ops.identity(  # Just to find it in the graph.
                sequence_length, name="sequence_length")

        batch_size = _best_effort_input_batch_size(flat_input)
        '''
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    "If there is no initial_state, you must give a dtype.")
        '''
        state = cell.zero_state(batch_size = batch_size, dtype = dtype,ini_state = initial_state)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if not context.executing_eagerly() and sequence_length is not None:
            # Perform some shape validation
            with ops.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(
            structure=inputs, flat_sequence=flat_input)
        target_outputs = nest.pack_sequence_as(
            structure=target_outputs, flat_sequence=flat_target_output)

        (outputs, final_state) = _dynamic_rnn_loop(
            cell,
            inputs,
            target_outputs,
            loss_function,
            gradient_function,
            train,
            store_LE_signal, #1
            use_BP, #0
            update_gradient_every, #-1
            optimizer,
            global_grad,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            outputs = nest.map_structure(_transpose_batch_time, outputs)

        return (outputs, final_state)

def _dynamic_rnn_loop(cell,
                      inputs,
                      target_outputs,
                      loss_function,
                      gradient_function,
                      train,
                      store_LE_signal,
                      use_BP,
                      update_gradient_every,
                      optimizer,
                      global_grad, #-1
                      initial_state, # cell.zero_state(batch_size, dtype)
                      parallel_iterations, #None
                      swap_memory, # False  
                      sequence_length=None,
                      dtype=None):
    """Internal implementation of Dynamic RNN.

    Args:
      cell: An instance of RNNCell.
      inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
        tuple of such elements.
      initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
        `cell.state_size` is a tuple, then this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      parallel_iterations: Positive Python int.
      swap_memory: A Python boolean
      sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
      dtype: (optional) Expected dtype of output. If not specified, inferred from
        initial_state.

    Returns:
      Tuple `(final_outputs, final_state)`.
      final_outputs:
        A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
        `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
        objects, then this returns a (possibly nested) tuple of Tensors matching
        the corresponding shapes.
      final_state:
        A `Tensor`, or possibly nested tuple of Tensors, matching in length
        and shapes to `initial_state`.

    Raises:
      ValueError: If the input depth cannot be inferred via shape inference
        from the inputs.
    """
    state = initial_state
    assert isinstance(parallel_iterations,
                      int), "parallel_iterations must be int"

    state_size = cell.state_size
    flat_input = nest.flatten(inputs) 
    flat_target_output = nest.flatten(target_outputs) 
    flat_output_size = nest.flatten(cell.output_size) 
    flat_target_output_size = nest.flatten(target_outputs.shape)
    # Construct an initial output
    input_shape = array_ops.shape(flat_input[0])
    time_steps = input_shape[0]

    batch_size = _best_effort_input_batch_size(flat_input)

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)
    target_output_got_shape = tuple(target_output_.get_shape().with_rank_at_least(3)
                                    for target_output_ in flat_target_output)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0].value
        got_batch_size = shape[1].value
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = _concat(batch_size, size)
        return array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)

    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = math_ops.reduce_min(sequence_length)
        max_sequence_length = math_ops.reduce_max(sequence_length)
    else:
        max_sequence_length = time_steps

    time = array_ops.constant(0, dtype=dtypes.int32, name="time")

    with ops.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, element_shape, dtype):
        return tensor_array_ops.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            element_shape=element_shape,
                                            tensor_array_name=base_name + name,
                                            clear_after_read=False)

    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode:
        output_ta = tuple(
            _create_ta(
                "output_%d" % i,
                element_shape=(tensor_shape.TensorShape([const_batch_size])
                               .concatenate(
                                   _maybe_tensor_shape_from_tensor(out_size))),
                dtype=_infer_state_dtype(dtype, state))
            for i, out_size in enumerate(flat_output_size))
        input_ta = tuple(
            _create_ta(
                "input_%d" % i,
                element_shape=flat_input_i.shape[1:],
                dtype=flat_input_i.dtype)
            for i, flat_input_i in enumerate(flat_input))
        input_ta = tuple(ta.unstack(input_)
                         for ta, input_ in zip(input_ta, flat_input))
        target_output_ta = tuple(
            _create_ta(
                "target_output_%d" % i,
                element_shape=flat_target_output_i.shape[1:],
                dtype=flat_target_output_i.dtype)
            for i, flat_target_output_i in enumerate(flat_target_output))
        target_output_ta = tuple(ta.unstack(target_output_)
                                 for ta, target_output_ in zip(target_output_ta, flat_target_output))
    else:
        output_ta = tuple([0 for _ in range(time_steps.numpy())]
                          for i in range(len(flat_output_size)))
        input_ta = flat_input
        target_output_ta = flat_target_output

    #update_cnt = tf.constant(0, tf.int32)
    # def _time_step(time, output_ta_t, state, update_cnt):
    # def _time_step(time, output_ta_t, state, cum_loss):
    def _time_step(time, output_ta_t, state):
        """Take a time step of the dynamic RNN.

        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.

        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        if in_graph_mode:
            input_t = tuple(ta.read(time) for ta in input_ta) # (Tensorarray在某一索引下的值)
            # Restore some shape information
            for input_, shape in zip(input_t, inputs_got_shape):
                input_.set_shape(shape[1:])

            target_output_t = tuple(ta.read(time) for ta in target_output_ta)
            # Restore some shape information
            for target_output_, shape in zip(target_output_t, target_output_got_shape):
                target_output_.set_shape(shape[1:])

        else:
            input_t = tuple(ta[time.numpy()] for ta in input_ta)
            target_output_t = tuple(ta[time.numpy()]
                                    for ta in target_output_ta)

        input_t = nest.pack_sequence_as(
            structure=inputs, flat_sequence=input_t)
        target_output_t = nest.pack_sequence_as(
            structure=target_outputs, flat_sequence=target_output_t)
         
        def call_cell():  
            return cell(input_t,target_output_t,state)

        if sequence_length is not None:
            (output, new_state, new_outputs, old_state, old_outputs, inp) = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)

        else:
            (output, new_state, new_outputs, old_state, old_outputs, inp) = call_cell()
        # Pack state if using state tuples
        # output has time x batch x output
        output = nest.flatten(output)
        old_outputs = nest.flatten(old_outputs)
        inp = nest.flatten(inp)

        # Compute the gradients for each timestep
        def update_trainable_cells_eligibility_traces(apply, time):
            # Compute loss function and gradients
            #aa = new_state[-1][4][-1:] 
            #loss = loss_function(target_output_t, output[0], aa, time, loop_bound - 1)
            loss = loss_function(target_output_t, output[0], time, loop_bound - 1)
            '''
            #grads_vars_list = gradient_function(loss, output[0])
            # grads = [gradient_function(loss, output[0]),
            #         gradient_function(loss, output[0])]
            #grads = [gradient_function(loss, out) for out in new_outputs]
            # grads = [gradient_function(loss, old_outputs[0]),
            #         gradient_function(loss, new_outputs[1])]
            grads = gradient_function(loss, output[0])

            #call_cell_grad = lambda: cell.grad(grads_vars_list, new_state, apply)
            def call_cell_grad(): return cell.grad(grads, new_state, apply)
            #call_cell_grad = lambda: cell.grad(grads_vars_list, state, apply)
            op_list = call_cell_grad()
            '''

            def apply_wo_scaling():
                op_list = []
                for cell_ind, c in enumerate(cell._cells):
                    '''
                    #print("The cell id is:{}".format(cell_ind))
                    if cell_ind==CELLNUM-1:
                        g1 = gradient_function(loss, new_outputs[cell_ind], time, loop_bound - 1, cell_ind) # loss对cell[cell_ind]的输出求梯度
                        #g2 = gradient_function(loss, aa, time, loop_bound - 1, cell_ind) # loss对cell[cell_ind]的输出求梯度
                        g2 = 0
                        #inp式每一层的输入，new_state是每一层的输出状态，new_outputs是每一层的输出
                        call_cell_grad = lambda: cell.grad_c([g1,g2], new_state, inp, new_outputs, cell_ind, optimizer, apply) 
                        #call_cell_grad = lambda: cell.grad_c([g1,0], new_state, inp, new_outputs, cell_ind, optimizer, apply) 
                    else:
                    '''
                    g = gradient_function(loss, new_outputs[cell_ind], time, loop_bound - 1, cell_ind) # loss对cell[cell_ind]的输出求梯度
                    #inp式每一层的输入，new_state是每一层的输出状态，new_outputs是每一层的输出
                    call_cell_grad = lambda: cell.grad_c(g, new_state, inp, new_outputs, cell_ind, optimizer, apply) 
                    op_list.extend(call_cell_grad())
                return tf.group(op_list)
                #return op_list

            def apply_w_scaling():
                op_list = []
                g_list = []
                vars = []
                
                for cell_ind, c in enumerate(cell._cells):
                    g = gradient_function(loss, new_outputs[cell_ind], time, loop_bound - 1, cell_ind)
                    
                    get_cell_grad = lambda: cell.get_grad_c(g, new_state, inp, new_outputs, cell_ind)
                    c_g, c_v = get_cell_grad()
                    g_list.extend(c_g)
                    vars.extend(c_v)
                    
                '''Scale the gradients'''
                grads, global_norm = tf.clip_by_global_norm(g_list, clip_norm=global_grad)
                    
                '''Distribute scaled gradients'''
                for cell_ind, c in enumerate(cell._cells):
                    call_cell_grad = lambda: cell.grad_c_v(grads, vars, new_state, cell_ind, optimizer, apply)
                    op_list.extend(call_cell_grad())
                    
                return tf.group(op_list)
            
            op_list = cond(tf.greater(tf.constant(global_grad, tf.float32), 0.),
                           lambda: apply_w_scaling(),
                           lambda: apply_wo_scaling()) # global_grad==-1,执行这个

            with tf.control_dependencies([op_list]):
                op = tf.cond(tf.equal(apply, tf.constant(True, tf.bool)),
                             lambda: optimizer._finish([tf.no_op()], 'NoOp'),
                             lambda: tf.no_op())
                
                with tf.control_dependencies([op]):
                    return tf.identity(tf.ones(1, dtype=tf.bool))

        # Compute the gradients for each timestep
        def update_trainable_cells_BP(apply, time):

            # Compute loss function and gradients
            
            #loss = loss_function(target_output_t, output[0], time, loop_bound - 1)
            #grads_vars_list = gradient_function(loss, output[0], time, loop_bound - 1)
            
            #op_list = [optimizer.apply_gradients(grads_vars_list)]

            # with tf.control_dependencies(op_list):
            #    return tf.no_op()
            return tf.no_op()

        op = tf.no_op()

        with tf.control_dependencies([op]):
            # Decision tree
            op = tf.cond(tf.logical_or(tf.equal(train, tf.constant(1, tf.int32)),tf.equal(train, tf.constant(2, tf.int32))),
                         lambda: tf.cond(tf.equal(tf.constant(store_LE_signal, tf.int32), tf.constant(1, tf.int32)),
                                         lambda: tf.cond(tf.equal(tf.constant(use_BP, tf.int32), tf.constant(1, tf.int32)),
                                                         lambda: tf.cond(tf.logical_or(
                                                             tf.logical_and(
                                                                 tf.equal(update_gradient_every, -1), tf.equal(time, loop_bound - 1)),
                                                             tf.logical_and(tf.not_equal(update_gradient_every, -1), tf.logical_or(tf.equal(tf.mod(time, update_gradient_every), tf.constant(0, tf.int32)), tf.equal(time, loop_bound - 1)))),
                                             lambda: update_trainable_cells_BP(
                                                 True, time),
                                             lambda: update_trainable_cells_BP(False, time)),
                             lambda: tf.cond(tf.logical_or(  #### 从这里执行
                                 tf.logical_and(
                                     tf.equal(update_gradient_every, -1), tf.equal(time, loop_bound - 1)), 
                                 tf.logical_and(tf.not_equal(update_gradient_every, -1), tf.logical_or(tf.equal(tf.mod(time, update_gradient_every), tf.constant(0, tf.int32)), tf.equal(time, loop_bound - 1)))),
                                             lambda: update_trainable_cells_eligibility_traces(  ###
                                                 True, time),
                                             lambda: update_trainable_cells_eligibility_traces(False, time))),  ###
                lambda: tf.cond(tf.equal(tf.constant(use_BP, tf.int32), tf.constant(1, tf.int32)),
                                lambda: tf.cond(tf.logical_or(
                                    tf.logical_and(
                                        tf.equal(update_gradient_every, -1), tf.equal(time, loop_bound - 1)),
                                    tf.logical_and(tf.not_equal(update_gradient_every, -1), tf.logical_or(tf.equal(tf.mod(time, update_gradient_every), tf.constant(0, tf.int32)), tf.equal(time, loop_bound - 1)))),
                    lambda: update_trainable_cells_BP(True, time),
                    lambda: tf.no_op()),
                             lambda: tf.cond(tf.logical_or( 
                                 tf.logical_and( 
                                     tf.equal(update_gradient_every, -1), tf.equal(time, loop_bound - 1)),
                                 tf.logical_and(tf.not_equal(update_gradient_every, -1), tf.logical_or(tf.equal(tf.mod(time, update_gradient_every), tf.constant(0, tf.int32)), tf.equal(time, loop_bound - 1)))), ### 在线预测满足这条
                    lambda: update_trainable_cells_eligibility_traces(
                        True, time),
                    lambda: tf.no_op()))),
                lambda: tf.no_op()) #如果不能训练（train=0），来到这条
            
            with tf.control_dependencies([op]):
    
                if in_graph_mode:
                    output_ta_t = tuple(
                        ta.write(time, out) for ta, out in zip(output_ta_t, output))
                else:
                    for ta, out in zip(output_ta_t, output):
                        ta[time.numpy()] = out
                
                return (time + 1, output_ta_t, new_state)

    if in_graph_mode:
        # Make sure that we run at least 1 step, if necessary, to ensure
        # the TensorArrays pick up the dynamic shape.
        loop_bound = math_ops.minimum(
            time_steps, math_ops.maximum(1, max_sequence_length))
    else:
        # Using max_sequence_length isn't currently supported in the Eager
        # branch.
        loop_bound = time_steps

    #cum_loss = tf.constant(0, shape=(200, 1600), dtype=tf.float32)
    #cum_loss = tf.constant(0, dtype=tf.float32)

    '''
    def body_outter(t, output_ta, state):
        
        t, o, s = control_flow_ops.while_loop(
            cond=lambda time, *_: time < t+1,
            body=_time_step,
            #loop_vars=(time, output_ta, state, update_cnt),
            #loop_vars=(time, output_ta, state, cum_loss),
            loop_vars=(t, output_ta, state),
            parallel_iterations=1,  # parallel_iterations,
            maximum_iterations=time_steps,
            swap_memory=swap_memory,
            back_prop=True,
            #back_prop=False,
            )
        
        #Move the application of the learning and gradient computation to here
        #In order that one can make use of the output tensor arrays properly
        
        return t, o, s
    
    _, output_final_ta, final_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < loop_bound,
        body=body_outter,
        #loop_vars=(time, output_ta, state, update_cnt),
        #loop_vars=(time, output_ta, state, cum_loss),
        loop_vars=(time, output_ta, state),
        parallel_iterations=1,  # parallel_iterations,
        maximum_iterations=time_steps,
        swap_memory=swap_memory,
        back_prop=True,
        #back_prop=False,
    )
    '''
    #with tf.Session() as sess:  print(time_steps.eval()) 
    #with tf.Session() as sess:  print(loop_bound.eval()) 
    #print(tf.TensorShape([None, state[2][0].shape[1]])) 
    '''
    def SNUELayer_neststructure(layer):
        return SNUELayerStateTuple(
        tf.TensorShape(state[layer][0].shape),tf.TensorShape(state[layer][1].shape), tf.TensorShape(state[layer][2].shape), 
        tf.TensorShape(state[layer][3].shape), tf.TensorShape(state[layer][4].shape), tf.TensorShape(state[layer][5].shape), 
        tf.TensorShape(state[layer][6].shape), tf.TensorShape(state[layer][7].shape), tf.TensorShape(state[layer][8].shape), 
        tf.TensorShape(state[layer][9].shape),tf.TensorShape(state[layer][10].shape),tf.TensorShape(state[layer][11].shape))
    '''

    def SNUELayer_neststructure(layer):
        a = []
        for i in range(len(state[layer])):
            a.append(tf.TensorShape(state[layer][i].shape))
        return SNUELayerStateTuple(*a)

    def DenseLayer_neststructure(layer):
        a = []
        for i in range(len(state[layer])):
            a.append(tf.TensorShape(state[layer][i].shape))
        return DenseTuple(*a)
    
    '''
    def DenseLayer_neststructure(layer):
        return DenseTuple( 
                tf.TensorShape(state[layer][0].shape), tf.TensorShape(state[layer][1].shape), 
                tf.TensorShape(state[layer][2].shape), tf.TensorShape(state[layer][3].shape))
    '''

    GPLayer_neststructure = GPTuple(tf.TensorShape([1,None,state[-1][0].shape[2]]), 
        tf.TensorShape([1,None,None]), tf.TensorShape([1,None,None]), tf.TensorShape([1,None,None]),
        tf.TensorShape([None,1]),tf.TensorShape([1]), tf.TensorShape(None),tf.TensorShape([1,1,1])
        ,tf.TensorShape([1,1,1]))

    def Shape_invariants(L):
        shape_invariants = []

        for i in range(len(L)):
            if L[i] == 'Dense':
                shape_invariants.append(DenseLayer_neststructure(i))
            elif L[i] == 'SNU':
                shape_invariants.append(SNUELayer_neststructure(i))
            else:
                raise TypeError("The type of layer you specified is not implemented.")  

        shape_invariants.append(GPLayer_neststructure)
        return tuple(shape_invariants)

    _, output_final_ta, final_state = control_flow_ops.while_loop(
        cond=lambda time, *_: time < loop_bound,
        body=_time_step,
        #loop_vars=(time, output_ta, state, update_cnt),
        #loop_vars=(time, output_ta, state, cum_loss),
        loop_vars=(time, output_ta, state),
        parallel_iterations=1,  # parallel_iterations,
        maximum_iterations=time_steps,
        swap_memory=swap_memory,
        #back_prop=True,
        back_prop=False, #Space only: Forces tf.gradients to compute spatial gradients only
        shape_invariants=(time.get_shape(),  (tf.TensorShape(None),),Shape_invariants(['SNU','SNU','SNU'])) 
         #里面必须都用TensorShape,So in general you need to use get_shape() instead of tf.shape(). DenseTuple,SNUELayerStateTuple
    )
    if in_graph_mode:
        final_outputs = tuple(ta.stack() for ta in output_final_ta)
        # Restore some shape information
        for output, output_size in zip(final_outputs, flat_output_size):
            shape = _concat(
                [const_time_steps, const_batch_size], output_size, static=True)
            output.set_shape(shape)
    else:
        final_outputs = output_final_ta

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)
    if not in_graph_mode:
        final_outputs = nest.map_structure_up_to(
            cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)

    op_list = [tf.no_op()]
    #out = nest.map_structure(_transpose_batch_time, final_outputs)
    #tar_out = nest.map_structure(_transpose_batch_time, target_outputs)
    #out = nest.map_structure(_transpose_batch_time, final_outputs)
    #tar_out = nest.map_structure(_transpose_batch_time, target_outputs)
    #loss = loss_function(tar_out, out)
    #grads_vars_list = gradient_function(loss)
    #grads, vars = zip(*grads_vars_list)
    #call_cell_grad = lambda: cell.grad(grads_vars_list, final_state, True)
    #op_list, remaining_grad = call_cell_grad()
    # Apply any remaining gradients
    # if len(remaining_grad) > 0:
    #    op_list = tf.group([op_list, optimizer.apply_gradients(remaining_grad)])
    # else:
    #    op_list = tf.group(op_list)
    #op_list = [optimizer.apply_gradients(zip(grads, vars))]

    with tf.control_dependencies(op_list):
            # Unpack final output if not using output tuples.
        if in_graph_mode:
            final_outputs = tuple(ta.stack() for ta in output_final_ta)
            # Restore some shape information
            for output, output_size in zip(final_outputs, flat_output_size):
                shape = _concat(
                    [const_time_steps, const_batch_size], output_size, static=True)
                output.set_shape(shape)
        else:
            final_outputs = output_final_ta

        final_outputs = nest.pack_sequence_as(
            structure=cell.output_size, flat_sequence=final_outputs)
        if not in_graph_mode:
            final_outputs = nest.map_structure_up_to(
                cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)
        
        return (final_outputs, final_state)
