import jax
from jax._src.api import AxisName
from typing import TypeVar, Callable, Sequence, Any
# from jax.experimental.shard_map import shard_map
from jax._src.shard_map import smap
from jax.tree_util import tree_flatten
from jax._src.api_util import flatten_axes

__all__ = [
    "map_vmap",
    "pmap_vmap",
    "smap_vmap",
]

FUN_TYPE = TypeVar("FUN_TYPE", bound=Callable)

def map_vmap(fun: FUN_TYPE, batchsize: int) -> FUN_TYPE:
    def mapped_fun(*args):
        return jax.lax.map(fun, args, batch_size=batchsize)
    return mapped_fun # type: ignore
    

# vectorises a function with respect to in_axes and out_axes
# (semantics of returned function is equivalent to pmap)
# internally each axis specified in in_axes and out_axes is split in two.
# pmap is performed over the first of the two axes
def pmap_vmap(fun: FUN_TYPE, 
    axis_name: AxisName,
    batch_size: int,
    *,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0) -> FUN_TYPE:
        
    
    vfun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
    pfun = jax.pmap(vfun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes)
    
    
    def mapped_fun(*args):
        in_leaves, in_tree = tree_flatten(args)
        in_axes_flat = flatten_axes("pmap_vmap in_axes", in_tree, in_axes)
        # print("in_axes_flat:", in_axes_flat)
        
        # the dimension of each specified axis must agree (as in vmap)
        leaf_batch_axis_sizes = ([leaf.shape[axis] for axis, leaf in zip(in_axes_flat, in_leaves) if axis is not None])
        batch_axis_size = leaf_batch_axis_sizes[0]
        assert all(size == batch_axis_size for size in leaf_batch_axis_sizes)
        
        # compute number of batches
        num_batches, remainder = divmod(batch_axis_size, batch_size)
        assert remainder == 0
        # print(f"{batch_size=} {batch_axis_size=} {num_batches=}")
        
        # split axis with size (num_batches*batch_size) into two with shape (num_batches, batch_size) for each leaf that is mapped over
        batched_args = tuple(
            leaf if axis is None else
            leaf.reshape(leaf.shape[:axis] + (num_batches, batch_size) + leaf.shape[axis+1:])  
            for axis, leaf in zip(in_axes_flat, in_leaves))
        # print("batched_args:", jax.tree.map(lambda v: v.shape, batched_args))
        
        # run pmapped function
        batched_out = pfun(*in_tree.unflatten(batched_args))
        
        # print("batched_out:", jax.tree.map(lambda v: v.shape, batched_out))
        
        out_leaves, out_tree = tree_flatten(batched_out)
        out_axes_flat = flatten_axes("pmap_vmap out_axes", out_tree, out_axes)
        
        # print("out_axes_flat", out_axes_flat)
        # combine split axes in output with shape (num_batches, batch_size) to single axis with size (num_batches*batch_size)
        unbatched_leaves = tuple(
            leaf if axis is None else
            leaf.reshape(leaf.shape[:axis] + (num_batches * batch_size, ) + leaf.shape[axis+2:])  
            for axis, leaf in zip(out_axes_flat, out_leaves))

        out = out_tree.unflatten(unbatched_leaves)
        # print("out:", jax.tree.map(lambda v: v.shape, out))

        return out
        
        
    
    return mapped_fun # type:ignore
        

def smap_vmap(fun: FUN_TYPE,
    *,
    axis_name: AxisName,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0) -> FUN_TYPE:
    
    vfun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
    return smap(vfun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes) # type: ignore


    