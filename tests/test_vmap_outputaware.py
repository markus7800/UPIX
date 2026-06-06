import jax
import jax.numpy as jnp

def get_dependency_matrix(func, dummy_args):
    """
    Traces a multi-output function and returns a boolean matrix 
    of shape (num_inputs, num_outputs) indicating dependencies.
    """
    jaxpr = jax.make_jaxpr(func)(*dummy_args).jaxpr
    num_inputs = len(jaxpr.invars)
    num_outputs = len(jaxpr.outvars)
    
    dependency_matrix = [[False] * num_outputs for _ in range(num_inputs)]
    
    for out_idx, outvar in enumerate(jaxpr.outvars):
        # Track variable memory identities to avoid equality override issues
        active_ids = {id(outvar)}
        
        for eqn in reversed(jaxpr.eqns):
            if any(id(v) in active_ids for v in eqn.outvars):
                for invar in eqn.invars:
                    active_ids.add(id(invar))
                    
        for in_idx, invar in enumerate(jaxpr.invars):
            if id(invar) in active_ids:
                dependency_matrix[in_idx][out_idx] = True
                
    return dependency_matrix

def create_smart_grid_function(func, dependency_matrix):
    num_inputs = len(dependency_matrix)
    num_outputs = len(dependency_matrix[0])
    
    mapped_f = func
    
    # Iterate backwards so outermost vmap maps the first input (a)
    for i in reversed(range(num_inputs)):
        # Map exactly ONE input variable at this level
        in_axes: list = [None] * num_inputs
        in_axes[i] = 0
        
        # Determine which outputs should inherit this mapped dimension
        out_axes = []
        for j in range(num_outputs):
            if dependency_matrix[i][j]:
                out_axes.append(0)    # Add dimension to this output
            else:
                out_axes.append(None) # Output ignores this dimension
        
        mapped_f = jax.vmap(
            mapped_f, 
            in_axes=tuple(in_axes), 
            out_axes=tuple(out_axes)
        )
        
    # JIT compile to enforce XLA shared-computation hoisting
    return jax.jit(mapped_f)

# --- 1. Define a complex function ---
def f(a, b, c, d, e):
    # Shared intermediate (depends on a, b)
    shared = a + b              
    
    out1 = shared               # Depends on: a, b
    out2 = jax.lax.cond(        # Control flow! Depends on: c, d, e
        c > 0, 
        lambda: d * e, 
        lambda: d + e
    )
    out3 = a * c * e            # Depends on: a, c, e
    out4 = shared + out2        # Depends on: ALL (a, b, c, d, e)
    
    return out1, out2, out3, out4

# --- 2. Define 1D ranges for our 5 inputs ---
# (Different sizes to prove the dimensions map correctly)
ranges = [
    jnp.arange(10), # a (size 10)
    jnp.arange(20), # b (size 20)
    jnp.arange(30), # c (size 30)
    jnp.arange(40), # d (size 40)
    jnp.arange(50), # e (size 50)
]
dummy_args = [r[0] for r in ranges]

# --- 3. Build and execute ---
dep_matrix = get_dependency_matrix(f, dummy_args)
smart_f = create_smart_grid_function(f, dep_matrix)

# ONE forward pass. No memory blowup.
outputs = smart_f(*ranges)

# --- 4. Verify Shapes ---
print(f"Out1 shape (Expected 10, 20): {outputs[0].shape}")
print(f"Out2 shape (Expected 30, 40, 50): {outputs[1].shape}")
print(f"Out3 shape (Expected 10, 30, 50): {outputs[2].shape}")
print(f"Out4 shape (Expected 10, 20, 30, 40, 50): {outputs[3].shape}")