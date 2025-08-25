import jax
from jax._src.api import AxisName
from typing import TypeVar, Callable, Sequence, Any, Tuple
# from jax.experimental.shard_map import shard_map
from jax._src.shard_map import smap
from jax.tree_util import tree_flatten
from jax._src.api_util import flatten_axes
from dccxjax.utils import log_warn

__all__ = [
    "batched_vmap",
    "pmap_vmap",
    "smap_vmap",
]

FUN_TYPE = TypeVar("FUN_TYPE", bound=Callable)

# def map_vmap(fun: FUN_TYPE, batchsize: int) -> FUN_TYPE:
#     def mapped_fun(*args):
#         return jax.lax.map(fun, args, batch_size=batchsize)
#     return mapped_fun # type: ignore
    
def batch_func_args(args, in_axes, batch_size: int):
    in_leaves, in_tree = tree_flatten(args)
    in_axes_flat = flatten_axes("pmap_vmap in_axes", in_tree, in_axes)
    # print("in_axes_flat:", in_axes_flat)
    
    # the dimension of each specified axis must agree (as in vmap)
    leaf_batch_axis_sizes = ([leaf.shape[axis] for axis, leaf in zip(in_axes_flat, in_leaves) if axis is not None])
    batch_axis_size = leaf_batch_axis_sizes[0]
    assert all(size == batch_axis_size for size in leaf_batch_axis_sizes)
    
    # compute number of batches
    num_batches, remainder = divmod(batch_axis_size, batch_size)
    batch_elems = int(num_batches * batch_size)

    # if remainder != 0 and num_batches > 0:
    # assert remainder == 0
    # print(f"{batch_size=} {batch_axis_size=} {num_batches=}")
    
    # split axis with size (num_batches*batch_size) into two with shape (num_batches, batch_size) for each leaf that is mapped over
    if batch_elems > 0:
        batched_args = tuple(
            leaf if axis is None else
            jax.lax.slice_in_dim(leaf, 0, batch_elems, 1, axis).reshape(leaf.shape[:axis] + (num_batches, batch_size) + leaf.shape[axis+1:])  
            for axis, leaf in zip(in_axes_flat, in_leaves)
        )
        batched_args = in_tree.unflatten(batched_args)
    else:
        batched_args = None
    
    if remainder:
        remainder_args = tuple(
            leaf if axis is None else
            jax.lax.slice_in_dim(leaf, batch_elems, None, 1, axis)
            for axis, leaf in zip(in_axes_flat, in_leaves)
        )
        remainder_args = in_tree.unflatten(remainder_args)
    else:
        remainder_args = None
        
    return batched_args, remainder_args, int(num_batches), int(remainder)

def unbatch_output(batched_out, out_axes, batch_size: int, num_batches: int):
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

def concatentate_output(x, y, dimension: int, out_axes):
    x_leaves, out_tree = tree_flatten(x)
    y_leaves, out_tree_2 = tree_flatten(y)
    assert out_tree == out_tree_2
    out_axes_flat = flatten_axes("concatentate_output out_axes", out_tree, out_axes)
    concatenated_leaves = [x if axis is None else jax.lax.concatenate((x, y), dimension) for x, y, axis in zip(x_leaves, y_leaves, out_axes_flat)]
    return out_tree.unflatten(concatenated_leaves)
    

def swap_axis_perm(arr: jax.Array, axis1: int, axis2: int):
    all_axes = list(range(arr.ndim))
    all_axes[axis1], all_axes[axis2] = all_axes[axis2], all_axes[axis1]
    return all_axes

def make_batch_axis_leading(args, batch_axes, num_batches: int):
    leaves, tree = tree_flatten(args)
    batch_axes_flat = flatten_axes("make_batch_axis_leading in_axes", tree, batch_axes)
    
    transposed_args = tuple(
        jax.lax.broadcast(leaf, (num_batches,)) if axis is None else
        jax.lax.transpose(leaf, swap_axis_perm(leaf, 0, axis))
        for axis, leaf in zip(batch_axes_flat, leaves) 
    )
    
    return tree.unflatten(transposed_args)

def put_batch_axis_back(args, batch_axes):
    leaves, tree = tree_flatten(args)
    batch_axes_flat = flatten_axes("put_batch_axis_back in_axes", tree, batch_axes)
    
    transposed_args = tuple(
        leaf[0,...] if axis is None else
        jax.lax.transpose(leaf, swap_axis_perm(leaf, 0, axis))
        for axis, leaf in zip(batch_axes_flat, leaves) 
    )
    
    return tree.unflatten(transposed_args)

# like jax.lax.map but with in_axes and out_axes, batch_size must divided batch_axis dim
def batched_vmap(fun: FUN_TYPE, batch_size: int, in_axes: int | None | Sequence[Any] = 0, out_axes: Any = 0) -> FUN_TYPE:
    vfun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
    def mapped_fun(*args):
        batched_args, remainder_args, num_batches, remainder = batch_func_args(args, in_axes, batch_size)
        if (num_batches > 0 and remainder > 0):
            log_warn(f"Warning: batching vmap with num_batches={num_batches} and remainder={remainder}")
        # print(f"batched_args {jax.tree.map(lambda v: v.shape, batched_args)}", num_batches)
        # print(f"remainder_args {jax.tree.map(lambda v: v.shape, remainder_args)}")
        # TODO: replace in future if this gets implemented: https://github.com/jax-ml/jax/issues/30528
        if remainder_args is not None:
            remainder_out = vfun(*remainder_args)
            # print(f"remainder_out {jax.tree.map(lambda v: v.shape, remainder_out)}")
        else:
            remainder_out = None
            
        if batched_args is not None:
            batched_args = make_batch_axis_leading(batched_args, in_axes, num_batches)
            _, batched_out = jax.lax.scan(lambda _, x: ((), vfun(*x)), (), batched_args)
            # print(f"batched_out {jax.tree.map(lambda v: v.shape, batched_out)}")
            batched_out = put_batch_axis_back(batched_out, out_axes)
            unbatched_out = unbatch_output(batched_out, out_axes, batch_size, num_batches)
            
            if remainder_out is not None:
                return concatentate_output(unbatched_out, remainder_out, 0, out_axes)
            else:
                return unbatched_out
            
        else:
            assert remainder_out is not None
            return remainder_out
            
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
    pfun = jax.pmap(fun, in_axes=in_axes, out_axes=out_axes)
    pvfun = jax.pmap(vfun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes)
    
    def mapped_fun(*args):
        batched_args, remainder_args, num_batches, remainder = batch_func_args(args, in_axes, batch_size)
        if (remainder > 0):
            log_warn(f"Warning: pmap vmap with num_batches={num_batches} and remainder={remainder}")
        
        if remainder_args is not None:
            remainder_out = pfun(*remainder_args)
        else:
            remainder_out = None
            
        if batched_args is not None:
            batched_out = pvfun(*batched_args)
            unbatched_out = unbatch_output(batched_out, out_axes, batch_size, num_batches)
            
            if remainder_out is not None:
                return concatentate_output(unbatched_out, remainder_out, 0, out_axes)
            else:
                return unbatched_out
        else:
            assert remainder_out is not None
            return remainder_out
        
    return mapped_fun # type:ignore
        

def smap_vmap(fun: FUN_TYPE,
    *,
    axis_name: AxisName,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0) -> FUN_TYPE:
    
    vfun = jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)
    return smap(vfun, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes) # type: ignore


    