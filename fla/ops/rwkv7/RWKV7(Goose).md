# RWKV7 (Goose) Mechanism: Mathematical Derivation

Zhiyuan Li

## Introduction to RWKV-7 Architecture

RWKV-7 employs **Dynamic State Evolution** that transcends the fundamental TC0 expressivity limitations of attention/linear attention paradigms. RWKV-7 possesses NC1 expressivity, allowing it to solve many problems that attention mechanisms cannot.

In simple terms, traditional attention mechanisms (like Transformer's QKV-softmax-attention) store multiple {k,v} (key and value vector pairs), matching queries (q alias named r in RWKV) against keys to retrieve corresponding values.

RWKV-7 takes a different approach - rather than directly storing {k,v} pairs, it dynamically updates a state by learning relationships between keys and values from context. This updated state then processes new input queries (q, or r in RWKV terminology) to produce outputs[^1].

[^1]: For a more detailed explanation of this approach, see the original article by the RWKV author: https://mp.weixin.qq.com/s/kC_Z3vuQ5B4PiRwZVeIvHQ

Specifically, RWKV-7 maintains an internal model $v≈kS^⊤$. It aims to fit a simple objective: for given vector sequences {kt} and {vt}, use state S to transform ki into vi, making the output v as close as possible to the target v.

To achieve this, during inference with an L2 loss function $L=½‖v−kS^⊤‖²$, RWKV-7 automatically simulates dynamic gradient descent to continuously train its internal model $v≈kS^⊤$.

The gradient is: **$∂L/∂S = S_k^T k - v^T k$**

Therefore, the gradient descent update (with weight decay factors $d_t = \exp(-\exp(w_t))$ and learning rate parameters) is: $$S_t = S_{t-1} \cdot \text{Diag}(d_t) - \eta_t \cdot (k_t^T k_t S_{t-1} - k_t^T v_t)$$ This simplifies to:

$$S_t = S_{t-1} \cdot \text{Diag}(d_t) - \eta_t \cdot k_t^T k_t \cdot S_{t-1} + \eta_t \cdot k_t^T v_t$$ 

$$S_t = S_{t-1} \cdot (\text{Diag}(d_t) - \eta_t \cdot k_t^T k_t) + \eta_t \cdot k_t^T v_t$$

In the full RWKV-7 implementation, this gradient descent update is generalized by replacing the terms as follows:

- $\text{Diag}(d_t)$ becomes $D_t$ (the diagonal decay matrix)
- The term $-\eta_t \cdot k_t^T k_t$ is generalized to $\alpha_t \beta_t^T$, where:
  - $\alpha_t$ can be initialized as $-\eta_t \cdot k_t$ 
  - $\beta_t$ can be initialized as $k_t$
- The term $\eta_t \cdot k_t^T v_t$ becomes $v_t k_t^T$ with appropriate scaling of $k_t$

This leads to the final recurrence equation[^2]:

[^2]: For a more detailed explanation of, see the triton codes: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L94

$$S_t = S_{t-1} \cdot D_t + S_{t-1} \cdot \alpha_t \beta_t^T + v_t k_t^T \in \mathbb{R}^{d_v \times d_k}$$

This formulation allows more flexibility in how the state evolves while maintaining the core gradient descent learning dynamics.

The output at each timestep is computed as:

$$o_t = q_t^T \cdot S_t$$ Where $q_t \in \mathbb{R}^{d_k}$ is the query vector (named $r$ in RWKV terminology), typically scaled by a factor of $\frac{1}{\sqrt{d_k}}$. This formulation allows RWKV-7 to continuously adapt its internal representation based on context, transcending the limitations of traditional attention mechanisms.

## 1. Forward Pass Recurrence Equation

The RWKV7 recurrence equation is defined as:

$$S_t = S_{t-1} \cdot D_t + S_{t-1} \cdot \alpha_t \beta_t^T + v_t k_t^T \in \mathbb{R}^{d_v \times d_k}$$

Where:

- $S_t$ is the state matrix at time $t$ with dimensions $d_v \times d_k$
- $D_t = \text{Diag}(d_t)$ is a diagonal matrix containing decay factors
- $\alpha_t, \beta_t, k_t \in \mathbb{R}^{d_k}$ are key-related vectors produced from input at time $t$
- $v_t \in \mathbb{R}^{d_v}$ is the value vector produced from input at time $t$
- $d_t = \exp(-\exp(w_t))$ where $w_t$ is produced from input at time $t$

The output at each timestep is computed as:

$$o_t = q_t^T \cdot S_t$$

Where $q_t \in \mathbb{R}^{d_k}$ is the query vector, typically scaled by a factor of $\frac{1}{\sqrt{d_k}}$.

## 2. Backward Pass Derivation

Let $L$ be the loss function. We need to compute gradients with respect to all inputs by working backwards through time.

### 2.1 Gradient of Loss w.r.t. State $S_t$

Let's denote $\frac{\partial L}{\partial S_t}$ as $dS_t$. For any timestep $t$:

$$dS_t = \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial S_t} + \frac{\partial L}{\partial S_{t+1}} \frac{\partial S_{t+1}}{\partial S_t}$$

The first term comes from the output computation: $\frac{\partial o_t}{\partial S_t} = q_t^T$, so:

$$\frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial S_t} = do_t \cdot q_t^T$$

The second term accounts for how $S_t$ affects future states and is propagated backward through time.

### 2.2 Gradient of Loss w.r.t. Query $q_t$

$$\frac{\partial L}{\partial q_t} = \frac{\partial L}{\partial o_t} \frac{\partial o_t}{\partial q_t} = do_t \cdot S_t^T \cdot \text{scale}$$

The scale factor is included because the query is typically scaled in attention mechanisms.

### 2.3 Gradient Backpropagation Through Time

For time step $t$, we compute gradients with respect to all parameters and pass the gradient to the previous state.

#### 2.3.1 Gradient w.r.t. Previous State $S_{t-1}$

The state update can be broken down as: $$S_t = S_{t-1} \cdot D_t + S_{t-1} \cdot \alpha_t \beta_t^T + v_t k_t^T$$

Therefore: $$\frac{\partial S_t}{\partial S_{t-1}} = D_t + \alpha_t \beta_t^T$$

And: $$\frac{\partial L}{\partial S_{t-1}} = dS_t \cdot (D_t + \alpha_t \beta_t^T)$$

This includes:

- Contribution from decay term: $dS_t \cdot D_t$
- Contribution from state update modulation: $dS_t \cdot (\alpha_t \beta_t^T)$

#### 2.3.2 Gradient w.r.t. Decay Parameter $w_t$

For $d_t = \exp(-\exp(w_t))$, we have: $$\frac{\partial d_t}{\partial w_t} = -\exp(w_t) \cdot \exp(-\exp(w_t)) = -\exp(w_t) \cdot d_t$$

Then: $$\frac{\partial L}{\partial w_t} = \frac{\partial L}{\partial S_t} \frac{\partial S_t}{\partial D_t} \frac{\partial D_t}{\partial d_t} \frac{\partial d_t}{\partial w_t} = \sum_{i,j} \left(dS_t[i,j] \cdot S_{t-1}[i,j] \cdot (-\exp(w_t)) \cdot d_t[j]\right)$$

#### 2.3.3 Gradient w.r.t. α Parameter

$$\frac{\partial L}{\partial \alpha_t} = \frac{\partial L}{\partial S_t} \frac{\partial S_t}{\partial \alpha_t} = dS_t \cdot S_{t-1} \cdot \beta_t$$

#### 2.3.4 Gradient w.r.t. β Parameter

$$\frac{\partial L}{\partial \beta_t} = \frac{\partial L}{\partial S_t} \frac{\partial S_t}{\partial \beta_t} = (dS_t)^T \cdot S_{t-1}^T \cdot \alpha_t$$

#### 2.3.5 Gradient w.r.t. Key Vector $k_t$

$$\frac{\partial L}{\partial k_t} = \frac{\partial L}{\partial S_t} \frac{\partial S_t}{\partial k_t} = (dS_t)^T \cdot v_t$$

#### 2.3.6 Gradient w.r.t. Value Vector $v_t$

$$\frac{\partial L}{\partial v_t} = \frac{\partial L}{\partial S_t} \frac{\partial S_t}{\partial v_t} = dS_t \cdot k_t$$

## 3. Tensor Implementation Notes

In practice, the operations are vectorized for batches, multiple heads, and sequence lengths:

1. For state update:
   - $S_{t-1} \cdot D_t$ is implemented as `state * torch.exp(-torch.exp(w))`
   - $S_{t-1} \cdot \alpha_t \beta_t^T$ is implemented as `torch.einsum('bhik,bhk,bhj->bhij', state, a_t, b_t)`
   - $v_t k_t^T$ is implemented as `torch.einsum('bhi,bhj->bhij', v_t, k_t)`
2. For output computation:
   - $q_t^T \cdot S_t$ is implemented as `torch.einsum('bhj,bhij->bhi', q_t, state)`
3. For gradient computation:
   - $\frac{\partial L}{\partial q_t}$: `torch.einsum('bhi,bhij->bhj', doutput[:,:,t], state) * scale`
   - $\frac{\partial L}{\partial S_t}$ from output: `torch.einsum('bhi,bhj->bhij', doutput[:,:,t], q_t)`
   - $\frac{\partial L}{\partial w_t}$: `torch.sum(dstate * prev_state, dim=(-2,-1)) * w_t_grad`
   - $\frac{\partial L}{\partial \alpha_t}$: `torch.einsum('bhij,bhik,bhj->bhk', dstate, prev_state, b_t)`
   - $\frac{\partial L}{\partial \beta_t}$: `torch.einsum('bhji,bhki,bhk->bhj', dstate, prev_state, a_t)`
   - $\frac{\partial L}{\partial k_t}$: `torch.einsum('bhji,bhi->bhj', dstate, v_t)`
   - $\frac{\partial L}{\partial v_t}$: `torch.einsum('bhij,bhj->bhi', dstate, k_t)`
   - $\frac{\partial L}{\partial S_{t-1}}$: `torch.einsum('bhij,bhj->bhi', dstate, d_t) + torch.einsum('bhij,bhj,bhk->bhik', dstate, a_t, b_t)`

## 4. Memory Optimization Strategies

### 4.1 Checkpointing

To optimize memory usage during backpropagation with long sequences:

1. During forward pass:
   - Store state at regular intervals (every `state_ckpt_interval` steps)
   - This reduces memory requirements compared to storing all intermediate states
2. During backward pass:
   - Start from the nearest checkpoint
   - Recompute intermediate states as needed
   - Compute gradients working backward through time

### 4.2 Efficient Element-wise Implementation

For educational purposes and validation, an element-wise implementation can be used that explicitly computes each operation. While slower, this provides a clear reference for understanding the mathematical operations and validating vectorized implementations.

This approach balances computational efficiency and memory usage, making it possible to train on long sequences with limited memory resources.

```python
import torch
from typing import Optional, Tuple
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

def elementwise_rwkv7_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """
    ElementWise implementation of RWKV7 (Goose) attention mechanism.
    
    Args:
        q: Query tensor of shape [B, H, L, N]
        k: Key tensor of shape [B, H, L, N]
        v: Value tensor of shape [B, H, L, V]
        w: Time decay weights of shape [B, H, L, N]
        a: Dynamic learning rate modulator of shape [B, H, L, N]
        b: State update modulator of shape [B, H, L, N]
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state
        
    Returns:
        output: Attention output of shape [B, H, L, V]
        final_state: Final state if requested
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float32] else torch.float32
    orig_dtype = q.dtype
    
    B, H, L, N = q.shape
    V = v.shape[-1]
    
    # Convert all inputs to the computation dtype
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    
    # Initialize state and output
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    output = torch.zeros(B, H, L, V, dtype=torch_dtype, device=q.device)
    
    # Apply scaling to query
    if scale == -1.0:
        scale = N ** -0.5
    q = q * scale
    
    # Use initial state if provided
    if initial_state is not None:
        state = state + initial_state.to(dtype=torch_dtype)
    
    # Process each timestep
    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                # Get current timestep values
                q_t = q[bi, hi, t]  # [N]
                k_t = k[bi, hi, t]  # [N]
                v_t = v[bi, hi, t]  # [V]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))  # [N]
                a_t = a[bi, hi, t]  # [N]
                b_t = b[bi, hi, t]  # [N]
                
                # Current state
                s_t = state[bi, hi]  # [N, V]
                
                # Compute state update components
                # 1. Decay the current state: S_{t-1} * D_t
                decayed_state = s_t * w_t.unsqueeze(-1)  # [N, V]
                
                # 2. Compute S_{t-1} * a_t
                sa = torch.zeros(V, dtype=torch_dtype, device=q.device)
                for n in range(N):
                    for v_idx in range(V):
                        sa[v_idx] += s_t[n, v_idx] * a_t[n]
                
                # 3. Compute (S_{t-1} * a_t) * b_t^T
                sab = torch.zeros_like(s_t)
                for n in range(N):
                    for v_idx in range(V):
                        sab[n, v_idx] = sa[v_idx] * b_t[n]
                
                # 4. Compute v_t * k_t^T
                kv = torch.zeros_like(s_t)
                for n in range(N):
                    for v_idx in range(V):
                        kv[n, v_idx] = k_t[n] * v_t[v_idx]
                
                # 5. Update state: S_t = S_{t-1} * D_t + sab + v_t * k_t^T
                state[bi, hi] = decayed_state + sab + kv
                
                # 6. Compute output: o_t = q_t^T * S_t
                o_t = torch.zeros(V, dtype=torch_dtype, device=q.device)
                for n in range(N):
                    for v_idx in range(V):
                        o_t[v_idx] += q_t[n] * state[bi, hi, n, v_idx]
                
                output[bi, hi, t] = o_t
    
    # Return output and final state if requested
    if output_final_state:
        return output.to(orig_dtype), state.to(orig_dtype)
    else:
        return output.to(orig_dtype), None



def elementwise_rwkv7_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    scale: float = 1.0
):
    """
    ElementWise implementation of RWKV7 backward pass.
    
    Args:
        q, k, v, w, a, b: Forward pass inputs
        doutput: Gradient of the loss with respect to the output
        dh_t: Gradient of the loss with respect to the final state
        scale: Scaling factor used in the forward pass
        
    Returns:
        dq, dk, dv, dw, da, db: Gradients with respect to inputs
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float32] else torch.float32
    B, H, L, N = q.shape
    V = v.shape[-1]
    
    # Convert all inputs and gradients to computation dtype
    q, k, v, w, a, b, doutput = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b, doutput))
    
    # Initialize gradient tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dw = torch.zeros_like(w)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    
    # Apply scaling to query
    if scale == -1.0:
        scale = N ** -0.5
    
    # Initialize state and state gradient
    states = []
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    dstate = torch.zeros_like(state)
    
    # If gradient with respect to final state is provided, use it
    if dh_t is not None:
        dstate = dstate + dh_t.to(dtype=torch_dtype)
    
    # Forward pass to store all intermediate states
    for t in range(L):
        states.append(state.clone())  # Store the state BEFORE update
        
        for bi in range(B):
            for hi in range(H):
                # Get current timestep values
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                
                # Current state before update
                s_t = state[bi, hi]
                
                # Compute intermediate values
                sa = torch.zeros(V, dtype=torch_dtype, device=q.device)
                for n in range(N):
                    for v_idx in range(V):
                        sa[v_idx] += s_t[n, v_idx] * a_t[n]
                
                # Update state
                for n in range(N):
                    for v_idx in range(V):
                        state[bi, hi, n, v_idx] = (
                            s_t[n, v_idx] * w_t[n] +
                            sa[v_idx] * b_t[n] +
                            k_t[n] * v_t[v_idx]
                        )
    
    # Save the final state for output computation
    final_states = state.clone()
    
    # Backward pass
    for t in range(L-1, -1, -1):
        for bi in range(B):
            for hi in range(H):
                # Get current timestep values
                q_t = q[bi, hi, t]  # Don't remove scaling yet
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                w_t_param = w[bi, hi, t]
                w_t = torch.exp(-torch.exp(w_t_param))
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                
                # Get the state before update at time t
                prev_state = states[t][bi, hi]
                
                # Get the state after update at time t (for output computation)
                if t < L-1:
                    current_state = states[t+1][bi, hi]
                else:
                    current_state = final_states[bi, hi]
                
                dout_t = doutput[bi, hi, t]
                
                # Gradient with respect to query
                for n in range(N):
                    for v_idx in range(V):
                        # Use current state (after update) for output computation
                        dq[bi, hi, t, n] += dout_t[v_idx] * current_state[n, v_idx] * scale
                
                # Gradient with respect to state from output
                for n in range(N):
                    for v_idx in range(V):
                        dstate[bi, hi, n, v_idx] += dout_t[v_idx] * q_t[n] * scale
                
                # Compute intermediate gradients
                for n in range(N):
                    for v_idx in range(V):
                        # Gradient with respect to key
                        dk[bi, hi, t, n] += dstate[bi, hi, n, v_idx] * v_t[v_idx]
                        
                        # Gradient with respect to value
                        dv[bi, hi, t, v_idx] += dstate[bi, hi, n, v_idx] * k_t[n]
                
                # Gradient with respect to decay parameter
                w_t_grad = torch.exp(w_t_param) * (-w_t)
                for n in range(N):
                    for v_idx in range(V):
                        dw[bi, hi, t, n] += dstate[bi, hi, n, v_idx] * prev_state[n, v_idx] * w_t_grad[n]
                
                # Compute sa for backprop (S_{t-1} * a_t)
                sa = torch.zeros(V, dtype=torch_dtype, device=q.device)
                for n in range(N):
                    for v_idx in range(V):
                        sa[v_idx] += prev_state[n, v_idx] * a_t[n]
                
                # Gradient with respect to a
                for n in range(N):
                    sum_grad = 0.0
                    for v_idx in range(V):
                        for k_idx in range(N):
                            sum_grad += dstate[bi, hi, k_idx, v_idx] * prev_state[n, v_idx] * b_t[k_idx]
                    da[bi, hi, t, n] += sum_grad
                
                # Gradient with respect to b
                for n in range(N):
                    for v_idx in range(V):
                        db[bi, hi, t, n] += dstate[bi, hi, n, v_idx] * sa[v_idx]
                
                # Gradient for previous state
                dprev_state = torch.zeros_like(prev_state)
                for n in range(N):
                    for v_idx in range(V):
                        # From decay term
                        dprev_state[n, v_idx] += dstate[bi, hi, n, v_idx] * w_t[n]
                
                # From (S_{t-1} * a_t) * b_t^T term
                for n in range(N):
                    for v_idx in range(V):
                        # First, compute ∂sa/∂S_{t-1}[n,v_idx] = a_t[n]
                        sa_grad = a_t[n]
                        
                        # Then, for each dimension k_idx of the state:
                        # multiply by b_t[k_idx] and dstate[bi,hi,k_idx,v_idx]
                        for k_idx in range(N):
                            dprev_state[n, v_idx] += dstate[bi, hi, k_idx, v_idx] * sa_grad * b_t[k_idx]
                
                # Update dstate for next iteration (previous timestep)
                dstate[bi, hi] = dprev_state
    
    return dq, dk, dv, dw, da, db, dstate

class ElementwiseRWKV7Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, a, b, scale=1.0, initial_state=None):
        ctx.save_for_backward(q, k, v, w, a, b)
        ctx.scale = scale
        output, final_state = elementwise_rwkv7_forward(
            q, k, v, w, a, b, scale, initial_state, output_final_state=True
        )
        return output, final_state
    
    @staticmethod
    def backward(ctx, doutput, dfinal_state):
        q, k, v, w, a, b = ctx.saved_tensors
        scale = ctx.scale
        
        dq, dk, dv, dw, da, db, _ = elementwise_rwkv7_backward(
            q, k, v, w, a, b, doutput, dfinal_state, scale
        )
        
        return dq, dk, dv, dw, da, db, None, None


def elementwise_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
):
    """
    Differentiable ElementWise implementation of RWKV7 attention.
    
    Args:
        q: Query tensor of shape [B, H, L, N]
        k: Key tensor of shape [B, H, L, N]
        v: Value tensor of shape [B, H, L, V]
        w: Time decay weights of shape [B, H, L, N]
        a: Dynamic learning rate modulator of shape [B, H, L, N]
        b: State update modulator of shape [B, H, L, N]
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        
    Returns:
        output: Attention output of shape [B, H, L, V]
        final_state: Final state
    """
    return ElementwiseRWKV7Function.apply(q, k, v, w, a, b, scale, initial_state)

# Original naive implementation for comparison
def naive_recurrent_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    for t in range(L):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        a_t = a[:, :, t]
        b_t = b[:, :, t]

        sab = torch.einsum('bhik,bhk,bhj->bhij', state, a_t, b_t)
        state = state * torch.exp(-torch.exp(w[:, :, t, None, :])) + sab + torch.einsum('bhj,bhi->bhij', k_t, v_t)
        o[:, :, t] = torch.einsum('bhj,bhij->bhi', q_t, state)

    if not output_final_state:
        ht = None
    elif initial_state is not None:
        ht = state.to(initial_state.dtype)
    else:
        ht = state.to(orig_dtype)

    return o.to(orig_dtype), ht


def test_forward_equivalence():
    """Test that elementwise and vectorized implementations give same results"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define test dimensions
    B, H, L, N, V = 2, 2, 3, 4, 5
    
    # Create random test inputs
    q = torch.randn(B, H, L, N, requires_grad=True)
    k = torch.randn(B, H, L, N, requires_grad=True)
    v = torch.randn(B, H, L, V, requires_grad=True)
    w = torch.randn(B, H, L, N, requires_grad=True)
    a = torch.randn(B, H, L, N, requires_grad=True)
    b = torch.randn(B, H, L, N, requires_grad=True)
    
    # Create initial state
    initial_state = torch.randn(B, H, N, V)
    
    # Run both implementations
    output1, state1 = naive_recurrent_rwkv7(q, k, v, w, a, b, initial_state=initial_state)
    output2, state2 = elementwise_rwkv7_forward(q, k, v, w, a, b, initial_state=initial_state)
    
    # Check if outputs are close
    output_diff = torch.max(torch.abs(output1 - output2)).item()
    state_diff = torch.max(torch.abs(state1 - state2)).item()
    
    print(f"Forward pass test:")
    print(f"  Max output difference: {output_diff:.6e}")
    print(f"  Max state difference: {state_diff:.6e}")
    
    # Define what's considered close enough (adjust as needed)
    tolerance = 1e-5
    
    if output_diff < tolerance and state_diff < tolerance:
        print("  ✓ Forward pass test PASSED: Implementations are equivalent")
    else:
        print("  ✗ Forward pass test FAILED: Implementations are different")
    
    return output_diff < tolerance and state_diff < tolerance


def test_backward_against_autograd():
    """Test elementwise backward implementation against PyTorch autograd"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define test dimensions (smaller for faster testing)
    B, H, L, N, V = 1, 1, 16, 64, 64
    
    # Create random test inputs
    q = torch.randn(B, H, L, N, requires_grad=True)
    k = torch.randn(B, H, L, N, requires_grad=True)
    v = torch.randn(B, H, L, V, requires_grad=True)
    w = torch.randn(B, H, L, N, requires_grad=True)
    a = torch.randn(B, H, L, N, requires_grad=True)
    b = torch.randn(B, H, L, N, requires_grad=True)
    
    # Create initial state
    initial_state = torch.randn(B, H, N, V)
    
    # For autograd version, ensure inputs require gradients
    q_auto = q.clone().detach().requires_grad_(True)
    k_auto = k.clone().detach().requires_grad_(True)
    v_auto = v.clone().detach().requires_grad_(True)
    w_auto = w.clone().detach().requires_grad_(True)
    a_auto = a.clone().detach().requires_grad_(True)
    b_auto = b.clone().detach().requires_grad_(True)
    
    # Run forward pass with autograd tracking
    output_auto, state_auto = naive_recurrent_rwkv7(
        q_auto, k_auto, v_auto, w_auto, a_auto, b_auto, initial_state=initial_state
    )
    
    # Create random gradients for output and final state
    grad_output = torch.randn_like(output_auto)
    grad_state = torch.randn_like(state_auto)
    
    # Compute gradients using autograd
    output_auto.backward(grad_output, retain_graph=True)
    state_auto.backward(grad_state)
    
    # Get autograd gradients
    dq_auto = q_auto.grad.clone()
    dk_auto = k_auto.grad.clone()
    dv_auto = v_auto.grad.clone()
    dw_auto = w_auto.grad.clone()
    da_auto = a_auto.grad.clone()
    db_auto = b_auto.grad.clone()
    
    # Reset gradients for manual backward test
    q_auto.grad.zero_()
    k_auto.grad.zero_()
    v_auto.grad.zero_()
    w_auto.grad.zero_()
    a_auto.grad.zero_()
    b_auto.grad.zero_()
    
    # Run manual backward pass
    dq, dk, dv, dw, da, db, _ = elementwise_rwkv7_backward(
        q, k, v, w, a, b, grad_output, grad_state
    )
    
    # Compare gradients
    differences = {
        'dq': torch.max(torch.abs(dq - dq_auto)).item(),
        'dk': torch.max(torch.abs(dk - dk_auto)).item(),
        'dv': torch.max(torch.abs(dv - dv_auto)).item(),
        'dw': torch.max(torch.abs(dw - dw_auto)).item(),
        'da': torch.max(torch.abs(da - da_auto)).item(),
        'db': torch.max(torch.abs(db - db_auto)).item(),
    }
    
    print(f"\nBackward pass test:")
    for param, diff in differences.items():
        print(f"  Max {param} difference: {diff:.6e}")
    
    # Define what's considered close enough
    tolerance = 1e-5
    all_close = all(diff < tolerance for diff in differences.values())
    
    if all_close:
        print("  ✓ Backward pass test PASSED: Manual backward matches autograd")
    else:
        print("  ✗ Backward pass test FAILED: Manual backward differs from autograd")
    
    return all_close


def test_autograd_function():
    """Test the custom autograd function implementation"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define test dimensions
    B, H, L, N, V = 2, 2, 3, 5, 5
    
    # Create random test inputs
    q = torch.randn(B, H, L, N, requires_grad=True)
    k = torch.randn(B, H, L, N, requires_grad=True)
    v = torch.randn(B, H, L, V, requires_grad=True)
    w = torch.randn(B, H, L, N, requires_grad=True)
    a = torch.randn(B, H, L, N, requires_grad=True)
    b = torch.randn(B, H, L, N, requires_grad=True)
    
    # Create initial state
    initial_state = torch.zeros(B, H, N, V)
    
       # Clone inputs for the two paths we're testing
    q1, k1, v1, w1, a1, b1 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True), w.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True), b.clone().detach().requires_grad_(True)
    q2, k2, v2, w2, a2, b2 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(True), w.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True), b.clone().detach().requires_grad_(True)
    
    
    # Path 1: Using naive implementation with autograd
    
    output1, state1 = naive_recurrent_rwkv7(q1, k1, v1, w1, a1, b1, initial_state=initial_state)
    

    
    output2, state2 = ElementwiseRWKV7Function.apply(q2, k2, v2, w2, a2, b2, 1.0, initial_state)
    
    # Check forward pass equivalence
    output_diff = torch.max(torch.abs(output1 - output2)).item()
    state_diff = torch.max(torch.abs(state1 - state2.transpose(-1, -2))).item()
    
    print(f"\nAutograd Function test (forward):")
    print(f"  Max output difference: {output_diff:.6e}")
    print(f"  Max state difference: {state_diff:.6e}")
    
    # Create loss function to test backward pass
    def compute_loss(output, state):
        return output.sum() #+ state.sum()
    
    # Compute loss and gradients for both paths
    loss1 = compute_loss(output1, state1)
    loss1.backward()
    
    loss2 = compute_loss(output2, state2)
    loss2.backward()
    
    # Compare gradients
    grad_diffs = {
        'q': torch.max(torch.abs(q1.grad - q2.grad)).item(),
        'k': torch.max(torch.abs(k1.grad - k2.grad)).item(),
        'v': torch.max(torch.abs(v1.grad - v2.grad)).item(),
        'w': torch.max(torch.abs(w1.grad - w2.grad)).item(),
        'a': torch.max(torch.abs(a1.grad - a2.grad)).item(),
        'b': torch.max(torch.abs(b1.grad - b2.grad)).item(),
    }
    
    print(f"\nAutograd Function test (backward):")
    for param, diff in grad_diffs.items():
        print(f"  Max {param} gradient difference: {diff:.6e}")
    
    # Define what's considered close enough
    tolerance = 1e-5
    forward_ok = output_diff < tolerance and state_diff < tolerance
    backward_ok = all(diff < tolerance for diff in grad_diffs.values())
    
    if forward_ok and backward_ok:
        print("  ✓ Autograd Function test PASSED: Custom implementation matches PyTorch autograd")
    else:
        if not forward_ok:
            print("  ✗ Forward pass failed")
        if not backward_ok:
            print("  ✗ Backward pass failed")
    
    return forward_ok and backward_ok

test_autograd_function()
```

