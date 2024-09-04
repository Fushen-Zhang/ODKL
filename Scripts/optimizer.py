from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.compat.v1.train import Optimizer as OptimizerBase
from tensorflow.python.training.optimizer import _RefVariableProcessor, _DenseResourceVariableProcessor

def _get_processor(v):
  """The processor of v."""
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  
    return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)

class Optimizer(OptimizerBase):

  def __init__(self, use_locking, name):
      super(Optimizer, self).__init__(use_locking, name)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, finish=True):

    if distribute_ctx.get_cross_replica_context():
      raise RuntimeError("Use `_distributed_apply()` instead of "
                         "`apply_gradients()` in a cross-replica context.")
    if distribute_ctx.has_distribution_strategy():
      grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
      return distribute_ctx.get_replica_context().merge_call(
          self._distributed_apply, args=(grads_and_vars, global_step, name))

    grads_and_vars = tuple(grads_and_vars)  
    if not grads_and_vars:
      raise ValueError("No variables provided.")
    converted_grads_and_vars = []
    for g, v in grads_and_vars:
      if g is not None:
        try:
          g = ops.convert_to_tensor_or_indexed_slices(g)
        except TypeError:
          raise TypeError(
              "Gradient must be convertible to a Tensor"
              " or IndexedSlices, or None: %s" % g)
        if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
          raise TypeError(
              "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
      p = _get_processor(v)
      converted_grads_and_vars.append((g, v, p))

    converted_grads_and_vars = tuple(converted_grads_and_vars)
    var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
    if not var_list:
      raise ValueError("No gradients provided for any variable: %s." %
                       ([str(v) for _, v, _ in converted_grads_and_vars],))
    with ops.init_scope():
      self._create_slots(var_list)
    update_ops = []
    with ops.name_scope(name, self._name) as name:
      self._prepare()
      for grad, var, processor in converted_grads_and_vars:
        if grad is None:
          continue
        if context.executing_eagerly() or isinstance(
            var,
            resource_variable_ops.ResourceVariable) and not var._in_graph_mode:
          scope_name = ""
        else:
          scope_name = var.op.name
        with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
          update_ops.append(processor.update_op(self, grad))
          
      if finish:
          if global_step is None:
            apply_updates = self._finish(update_ops, name)
          else:
            with ops.control_dependencies([self._finish(update_ops, "update")]):
              with ops.colocate_with(global_step):
                if isinstance(global_step, resource_variable_ops.ResourceVariable):
                  apply_updates = resource_variable_ops.assign_add_variable_op(
                      global_step.handle,
                      ops.convert_to_tensor(1, dtype=global_step.dtype),
                      name=name)
                else:
                  apply_updates = state_ops.assign_add(global_step, 1, name=name)
    
          if not context.executing_eagerly():
            if isinstance(apply_updates, ops.Tensor):
              apply_updates = apply_updates.op
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
              train_op.append(apply_updates)
          return apply_updates
      else:
          with ops.control_dependencies(update_ops):
              return control_flow_ops.no_op()