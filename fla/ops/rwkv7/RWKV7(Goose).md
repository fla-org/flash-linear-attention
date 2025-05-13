# RWKV7 (Goose) Mechanism: Mathematical Derivation

Zhiyuan Li

>Special thanks to [Sonta](https://github.com/sustcsonglin) and [Beortust](https://github.com/Beortext), Sonta pointed out the correct notation for the outer product in the formulas, and Beortust corrected a considerable number of typos and also helped to improve the formatting.

## Introduction to RWKV-7 Architecture

RWKV-7 employs **Dynamic State Evolution** that transcends the fundamental TC0 expressivity limitations of attention/linear attention paradigms. RWKV-7 possesses NC1 expressivity, allowing it to solve many problems that attention mechanisms cannot.

In simple terms, traditional attention mechanisms (like Transformer's QKV-softmax-attention) store multiple $\{k,v\}$ (key and value vector pairs), matching queries ($q$ alias named $r$ in RWKV) against keys to retrieve corresponding values.

RWKV-7 takes a different approach - rather than directly storing $\{k,v\}$ pairs, it dynamically updates a state by learning relationships between keys and values from context. This updated state then processes new input queries ($q$, or $r$ in RWKV terminology) to produce outputs[^1].

[^1]: For a more detailed explanation of this approach, see the original article by the RWKV author: https://mp.weixin.qq.com/s/kC_Z3vuQ5B4PiRwZVeIvHQ

Specifically, RWKV-7 maintains an internal model $v \approx k^{\top} S$. It aims to fit a simple objective: for given vector sequences $\{k\}$ and $\{v\}$, use state $S$ to transform $k_i$ into $v_i$, making the output $v$ as close as possible to the target $v$.

For clarity on dimensions:

$S_t \in \mathbb{R}^{d_v \times d_k}$ is the state matrix

$k_t \in \mathbb{R}^{d_k}$ is the key vector

$v_t \in \mathbb{R}^{d_v}$ is the value vector

$q_t \in \mathbb{R}^{d_k}$ is the query vector (named $r$ in RWKV terminology)

To achieve this, during inference with an L2 loss function $L=\frac{1}{2}\left \| v−k^{\top} S \right \|^2$, RWKV-7 automatically simulates dynamic gradient descent to continuously train its internal model $v \approx k^{\top} S$.

The gradient of the L2 loss function with respect to the state matrix $S$ is: $\frac{\partial L}{\partial S}$

Applying stochastic gradient descent (SGD) with this gradient yields a recurrent update formula that forms the foundation of RWKV-7's mechanism. In standard SGD, we would update the parameters by subtracting the gradient scaled by a learning rate:

<svg xmlns="http://www.w3.org/2000/svg" width="25.306ex" height="5.31ex" viewBox="0 -1391 11185.4 2346.9" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" style=""><defs><path id="MJX-5-TEX-I-1D446" d="M308 24Q367 24 416 76T466 197Q466 260 414 284Q308 311 278 321T236 341Q176 383 176 462Q176 523 208 573T273 648Q302 673 343 688T407 704H418H425Q521 704 564 640Q565 640 577 653T603 682T623 704Q624 704 627 704T632 705Q645 705 645 698T617 577T585 459T569 456Q549 456 549 465Q549 471 550 475Q550 478 551 494T553 520Q553 554 544 579T526 616T501 641Q465 662 419 662Q362 662 313 616T263 510Q263 480 278 458T319 427Q323 425 389 408T456 390Q490 379 522 342T554 242Q554 216 546 186Q541 164 528 137T492 78T426 18T332 -20Q320 -22 298 -22Q199 -22 144 33L134 44L106 13Q83 -14 78 -18T65 -22Q52 -22 52 -14Q52 -11 110 221Q112 227 130 227H143Q149 221 149 216Q149 214 148 207T144 186T142 153Q144 114 160 87T203 47T255 29T308 24Z"></path><path id="MJX-5-TEX-I-1D461" d="M26 385Q19 392 19 395Q19 399 22 411T27 425Q29 430 36 430T87 431H140L159 511Q162 522 166 540T173 566T179 586T187 603T197 615T211 624T229 626Q247 625 254 615T261 596Q261 589 252 549T232 470L222 433Q222 431 272 431H323Q330 424 330 420Q330 398 317 385H210L174 240Q135 80 135 68Q135 26 162 26Q197 26 230 60T283 144Q285 150 288 151T303 153H307Q322 153 322 145Q322 142 319 133Q314 117 301 95T267 48T216 6T155 -11Q125 -11 98 4T59 56Q57 64 57 83V101L92 241Q127 382 128 383Q128 385 77 385H26Z"></path><path id="MJX-5-TEX-N-3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path><path id="MJX-5-TEX-N-2212" d="M84 237T84 250T98 270H679Q694 262 694 250T679 230H98Q84 237 84 250Z"></path><path id="MJX-5-TEX-N-31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path><path id="MJX-5-TEX-I-1D702" d="M21 287Q22 290 23 295T28 317T38 348T53 381T73 411T99 433T132 442Q156 442 175 435T205 417T221 395T229 376L231 369Q231 367 232 367L243 378Q304 442 382 442Q436 442 469 415T503 336V326Q503 302 439 53Q381 -182 377 -189Q364 -216 332 -216Q319 -216 310 -208T299 -186Q299 -177 358 57L420 307Q423 322 423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 114 189T154 366Q154 405 128 405Q107 405 92 377T68 316T57 280Q55 278 41 278H27Q21 284 21 287Z"></path><path id="MJX-5-TEX-N-22C5" d="M78 250Q78 274 95 292T138 310Q162 310 180 294T199 251Q199 226 182 208T139 190T96 207T78 250Z"></path><path id="MJX-5-TEX-N-2202" d="M202 508Q179 508 169 520T158 547Q158 557 164 577T185 624T230 675T301 710L333 715H345Q378 715 384 714Q447 703 489 661T549 568T566 457Q566 362 519 240T402 53Q321 -22 223 -22Q123 -22 73 56Q42 102 42 148V159Q42 276 129 370T322 465Q383 465 414 434T455 367L458 378Q478 461 478 515Q478 603 437 639T344 676Q266 676 223 612Q264 606 264 572Q264 547 246 528T202 508ZM430 306Q430 372 401 400T333 428Q270 428 222 382Q197 354 183 323T150 221Q132 149 132 116Q132 21 232 21Q244 21 250 22Q327 35 374 112Q389 137 409 196T430 306Z"></path><path id="MJX-5-TEX-I-1D43F" d="M228 637Q194 637 192 641Q191 643 191 649Q191 673 202 682Q204 683 217 683Q271 680 344 680Q485 680 506 683H518Q524 677 524 674T522 656Q517 641 513 637H475Q406 636 394 628Q387 624 380 600T313 336Q297 271 279 198T252 88L243 52Q243 48 252 48T311 46H328Q360 46 379 47T428 54T478 72T522 106T564 161Q580 191 594 228T611 270Q616 273 628 273H641Q647 264 647 262T627 203T583 83T557 9Q555 4 553 3T537 0T494 -1Q483 -1 418 -1T294 0H116Q32 0 32 10Q32 17 34 24Q39 43 44 45Q48 46 59 46H65Q92 46 125 49Q139 52 144 61Q147 65 216 339T285 628Q285 635 228 637Z"></path><path id="MJX-5-TEX-S4-2223" d="M139 -249H137Q125 -249 119 -235V251L120 737Q130 750 139 750Q152 750 159 735V-235Q151 -249 141 -249H139Z"></path></defs><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="msub"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D446"></use></g><g data-mml-node="mi" transform="translate(613, -150) scale(0.707)"><use xlink:href="#MJX-5-TEX-I-1D461"></use></g></g><g data-mml-node="mo" transform="translate(1196, 0)"><use xlink:href="#MJX-5-TEX-N-3D"></use></g><g data-mml-node="msub" transform="translate(2251.8, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D446"></use></g><g data-mml-node="TeXAtom" transform="translate(613, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D461"></use></g><g data-mml-node="mo" transform="translate(361, 0)"><use xlink:href="#MJX-5-TEX-N-2212"></use></g><g data-mml-node="mn" transform="translate(1139, 0)"><use xlink:href="#MJX-5-TEX-N-31"></use></g></g></g><g data-mml-node="mo" transform="translate(4296, 0)"><use xlink:href="#MJX-5-TEX-N-2212"></use></g><g data-mml-node="msub" transform="translate(5296.2, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D702"></use></g><g data-mml-node="mi" transform="translate(497, -150) scale(0.707)"><use xlink:href="#MJX-5-TEX-I-1D461"></use></g></g><g data-mml-node="mo" transform="translate(6320.7, 0)"><use xlink:href="#MJX-5-TEX-N-22C5"></use></g><g data-mml-node="mfrac" transform="translate(6820.9, 0)"><g data-mml-node="mrow" transform="translate(220, 676)"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-N-2202"></use></g><g data-mml-node="mi" transform="translate(566, 0)"><use xlink:href="#MJX-5-TEX-I-1D43F"></use></g></g><g data-mml-node="mrow" transform="translate(238, -686)"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-N-2202"></use></g><g data-mml-node="mi" transform="translate(566, 0)"><use xlink:href="#MJX-5-TEX-I-1D446"></use></g></g><rect width="1447" height="60" x="120" y="220"></rect></g><g data-mml-node="msub" transform="translate(8507.9, 0)"><g data-mml-node="TeXAtom" data-mjx-texclass="ORD"><g data-mml-node="mo"><svg width="278" height="2047" y="-773.5" x="27.5" viewBox="0 -253.6 278 2047"><use xlink:href="#MJX-5-TEX-S4-2223" transform="scale(1, 3.074)"></use></svg></g></g><g data-mml-node="TeXAtom" transform="translate(333, -808.9) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D446"></use></g><g data-mml-node="mo" transform="translate(645, 0)"><use xlink:href="#MJX-5-TEX-N-3D"></use></g><g data-mml-node="msub" transform="translate(1423, 0)"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D446"></use></g><g data-mml-node="TeXAtom" transform="translate(613, -150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><use xlink:href="#MJX-5-TEX-I-1D461"></use></g><g data-mml-node="mo" transform="translate(361, 0)"><use xlink:href="#MJX-5-TEX-N-2212"></use></g><g data-mml-node="mn" transform="translate(1139, 0)"><use xlink:href="#MJX-5-TEX-N-31"></use></g></g></g></g></g></g></g></svg>

Incorporating weight decay factors $d_t = \exp(-\exp(w_t))$ as a form of time-dependent regularization and learning rate $\eta_t$, the gradient descent update becomes:

$$S_t = S_{t-1} \text{Diag}(d_t) - \eta_t \cdot (S_{t-1} k_t k_t^{\top} - v_t k_t^{\top})$$

This can be expanded and rearranged as follows:

$$S_t = S_{t-1} \text{Diag}(d_t) - \eta_t \cdot S_{t-1} k_t k_t^{\top} + \eta_t \cdot v_t k_t^{\top}$$

For notational simplicity, we denote $\text{Diag}(d_t)$ as $D_t$ (the diagonal decay matrix):

$$S_t = S_{t-1} D_t - \eta_t \cdot S_{t-1} k_t k_t^{\top} + \eta_t \cdot v_t k_t^{\top}$$

In the full RWKV-7 implementation, this update rule is generalized through several key transformations:

1. The diagonal decay term $D_t$ remains as a component-wise multiplication with $S_{t-1}$

2. The term $-\eta_t \cdot k_t k_t^{\top}$ is generalized to $\alpha_t \beta_t^{\top}$, where:

   - $\alpha_t$ can be initialized as $-k_t$
   - $\beta_t$ can be initialized as $\eta_t \cdot k_t$

3. The term $-\eta_t \cdot S_{t-1} k_t k_t^{\top}$ can be factorized and computed efficiently:

   - First compute $u_t = S_{t-1} k_t$ (matrix-vector product)
   - Then compute $-\eta_t \cdot u_t k_t^{\top}$ (scaled outer product)

4. The term $\eta_t \cdot v_t k_t^{\top}$ is directly implemented as the outer product between the value vector $v_t$ and key vector $k_t$, resulting in a rank-1 update matrix

This leads to the final recurrence equation[^2]:

$$
S_t = S_{t-1} D_t + S_{t-1} \alpha_t \beta_t^{\top} + v_t k_t^{\top} \in \mathbb{R}^{d_v \times d_k}
$$

The output at each timestep is computed as:
$o_t = S_t r_t$

Where $r_t \in \mathbb{R}^{d_k}$ is the query vector (named $r$ in RWKV terminology), typically scaled by a factor of $\frac{1}{\sqrt{d_k}}$. This formulation allows RWKV-7 to continuously adapt its internal representation based on context, transcending the limitations of traditional attention mechanisms.

[^2]: For a more detailed explanation, see the triton codes. Note: In the optimized Triton implementation, `w` is already the log of the decay factor, so there's only one exponential operation needed. https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/rwkv7/fused_recurrent.py#L94

This formulation allows more flexibility in how the state evolves while maintaining the core gradient descent learning dynamics.

## 1. Forward Pass Recurrence Equation

In the implementation, the state update is defined as:

For each batch (bi) and head (hi), at time step t:

```python
w_t = torch.exp(-torch.exp(w[bi, hi, t]))  # shape [K]
sa = (state[bi, hi] * a_t[None, :]).sum(dim=1)  # shape [V]
state[bi, hi] = w_t[None, :] * state[bi, hi] + sa[:, None] * b_t[None, :] + k_t[None, :] * v_t[:, None]
```

Where state[bi, hi] has shape [V, K], representing a state matrix that maps from K-dimensional keys to V-dimensional values.

## 2. Backward Pass Derivation

### 2.1 Gradient of Loss w.r.t. State

For time step t, if L is the loss function, dstate_curr = ∂L/∂state[bi, hi, t+1] is the gradient of the current state:

```
dstate_curr = dstate[bi, hi] + q_t[None, :] * doutput[bi, hi, t][:, None]
```

This includes gradients propagated from future time steps dstate[bi, hi] and gradients from the current output.

### 2.2 Gradient w.r.t. Query q_t

```
dq[bi, hi, t] = torch.matmul(doutput[bi, hi, t], curr_state) * scale
```

### 2.3 Gradient w.r.t. Decay Parameter w_t

For the gradient of w_t, we need to consider how it affects the state update:

1. For the `w_t[None, :] * state[bi, hi]` component of the state update:

First, compute the derivative of L with respect to w_t:

```
∂L/∂w_t[k] = ∑_v (dstate_curr[v,k] * prev_state[v,k])
```

This equation sums over the v dimension for each position k, resulting in a vector of shape [K].

Then, compute the derivative of w_t with respect to w:

```
∂w_t[k]/∂w[k] = -exp(w[k]) * exp(-exp(w[k])) = -exp(w[k]) * w_t[k]
```

Finally, apply the chain rule:

```
∂L/∂w[k] = ∂L/∂w_t[k] * ∂w_t[k]/∂w[k]
         = (∑_v dstate_curr[v,k] * prev_state[v,k]) * (-exp(w[k]) * w_t[k])
```

In code, this is expressed as:

```python
dw[bi, hi, t] += -torch.sum(dstate_curr * prev_state, dim=0) * torch.exp(w[bi, hi, t]) * w_t
```

Or equivalently:

```python
dw[bi, hi, t] += -torch.sum(dstate_curr * prev_state, dim=0) * torch.exp(w[bi, hi, t]) * torch.exp(-torch.exp(w[bi, hi, t]))
```

### 2.4 Gradient w.r.t. k_t and v_t

For the `k_t[None, :] * v_t[:, None]` component:

```python
dk[bi, hi, t] += torch.sum(dstate_curr * v_t[:, None], dim=0)
dv[bi, hi, t] += torch.sum(dstate_curr * k_t[None, :], dim=1)
```

### 2.5 Gradient w.r.t. α_t and β_t (a_t and b_t in code)

For the `sa[:, None] * b_t[None, :]` component, where `sa = (state[bi, hi] * a_t[None, :]).sum(dim=1)`:

```python
db[bi, hi, t] += torch.sum(dstate_curr * sa[:, None], dim=0)
dsa = torch.sum(dstate_curr * b_t[None, :], dim=1)
da[bi, hi, t] += torch.sum(prev_state * dsa[:, None], dim=0)
```

### 2.6 Gradient w.r.t. Previous State S\_{t-1}

Finally, we compute the gradient of the previous state for backpropagation:

```python
dstate_from_sa = a_t[None, :] * dsa[:, None]
dstate_from_decay = dstate_curr * w_t[None, :]
dstate[bi, hi] = dstate_from_sa + dstate_from_decay
```

```python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch

from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def naive_recurrent_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.
    Modified from bo's code.
    https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py#L170

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, V, N, dtype=torch_dtype, device=q.device)
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

        # from bo's code
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


def naive_recurrent_rwkv7_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, V, N, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))

                # h: [V, K], a_t [K] -> [1, K]
                # sa: [V]
                sa = (state[bi, hi] * a_t[None, :]).sum(dim=1)

                state[bi, hi] = w_t[None, :] * state[bi, hi] + sa[:, None] * b_t[None, :] + k_t[None, :] * v_t[:, None]
                y = (state[bi, hi] * q_t[None, :]).sum(dim=1)

                o[bi, hi, t] = y

    ht = state if output_final_state else None
    return o.to(orig_dtype), ht


@torch.no_grad()
def naive_recurrent_rwkv7_2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None
):
    """
    Backward pass for the naive_recurrent_rwkv7_2 implementation.

    Args:
        q, k, v, w, a, b: Original forward pass inputs
        doutput: Gradient of the loss with respect to the output
        dh_t: Gradient of the loss with respect to the final state (if any)
        scale: Scaling factor used in the forward pass
        dtype: Optional dtype for computation

    Returns:
        Gradients with respect to all inputs
    """
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    q, k, v, w, a, b, doutput = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b, doutput))
    if dh_t is not None:
        dh_t = dh_t.to(dtype=torch_dtype)

    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]

    # Initialize gradients
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dw = torch.empty_like(w)
    da = torch.empty_like(a)
    db = torch.empty_like(b)

    # Initialize state gradients
    dstate = torch.zeros(B, H, V, N, dtype=torch_dtype, device=q.device)
    if dh_t is not None:
        dstate += dh_t

    if scale == -1.0:
        scale = N ** -0.5

    # First rebuild all states from forward pass
    states = []
    state = torch.zeros(B, H, V, N, dtype=torch_dtype, device=q.device)
    states.append(state.clone())

    # In practice, we don't recompute all states from the beginning.
    # Instead, we use checkpointing: we save states at regular intervals (e.g., every 16 tokens)
    # during the forward pass, then reconstruct intermediate states during the backward pass
    # by working backwards from the nearest checkpoint.
    #
    # For example, to get state[t-1] from state[t]:
    # state[t-1] = (state[t] - (sa * b_t + k_t * v_t)) / w_t
    #
    # This approach balances memory usage and computational efficiency:
    # - Reduces memory by not storing every state
    # - Maintains numerical stability by limiting the number of backward steps from each checkpoint
    # - Allows efficient gradient computation without recomputing the entire sequence
    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))

                sa = (state[bi, hi] * a_t[None, :]).sum(dim=1)

                state[bi, hi] = w_t[None, :] * state[bi, hi] + sa[:, None] * b_t[None, :] + k_t[None, :] * v_t[:, None]
        states.append(state.clone())

    # Backward pass through time
    for t in range(L-1, -1, -1):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_scalar = w[bi, hi, t]
                w_exp = torch.exp(w_scalar)
                w_t = torch.exp(-w_exp)

                curr_state = states[t+1][bi, hi]  # State after update [V, K]
                prev_state = states[t][bi, hi]    # State before update [V, K]

                dq[bi, hi, t] += torch.matmul(doutput[bi, hi, t], curr_state) * scale

                dstate_from_out = q_t[None, :] * doutput[bi, hi, t][:, None]  # [V, K]

                dstate_curr = dstate[bi, hi] + dstate_from_out

                sa = (prev_state * a_t[None, :]).sum(dim=1)  # [V]

                # state[bi, hi] = w_t[None, :] * prev_state + ...
                dw[bi, hi, t] = -torch.sum(dstate_curr * prev_state, dim=0) * \
                    w_t * w_exp

                # k_t[None, :] * v_t[:, None] -> [V, K]
                dk[bi, hi, t] = torch.sum(dstate_curr * v_t[:, None], dim=0)
                dv[bi, hi, t] = torch.sum(dstate_curr * k_t[None, :], dim=1)

                # sa[:, None] * b_t[None, :] -> [V, K]
                db[bi, hi, t] = torch.sum(dstate_curr * sa[:, None], dim=0)
                dsa = torch.sum(dstate_curr * b_t[None, :], dim=1)  # [V]

                # sa = (prev_state * a_t[None, :]).sum(dim=1)
                da[bi, hi, t] = torch.sum(prev_state * dsa[:, None], dim=0)
                dstate_from_sa = a_t[None, :] * dsa[:, None]  # [V, K]

                # w_t[None, :] * prev_state
                dstate_from_decay = dstate_curr * w_t[None, :]  # [V, K]

                dstate[bi, hi] = dstate_from_sa + dstate_from_decay

    return dq, dk, dv, dw, da, db, dstate


class NativeRecurrentRWKV7Function(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, initial_state,
                training: bool = True, dtype: Optional[torch.dtype] = None,
                state_ckpt_interval: int = 16):
        o, ht = naive_recurrent_rwkv7_2(q, k, v, w, a, b, scale=scale, initial_state=initial_state)
        if training:
            ctx.save_for_backward(q, k, v, w, a, b)
            ctx.scale = scale
            ctx.dtype = dtype
            ctx.ckpt_interval = state_ckpt_interval
            ctx.use_initial_state = initial_state is not None
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, a, b = ctx.saved_tensors
        dq, dk, dv, dw, da, db, dh = naive_recurrent_rwkv7_2_bwd(
            q, k, v, w, a, b, do, dht, ctx.scale, dtype=ctx.dtype)
        dh = dh if ctx.use_initial_state else None
        return dq, dk, dv, dw, da, db, None, dh, None, None


def recurrent_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    assert cu_seqlens is None
    assert head_first is True
    assert w is not None
    if scale == -1.0:
        scale = q.shape[-1] ** -0.5
    o, final_state = NativeRecurrentRWKV7Function.apply(q, k, v, w, a, b, scale, initial_state)

    return o, final_state


def test_autograd_function():
    """Test the custom autograd function implementation"""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test dimensions
    B, H, T, D = 1, 1, 128, 64
    V = N = D
    device = 'cpu'
    dtype = torch.float64

    # Create random test inputs
    q = torch.empty(B, H, T, D, device=device).uniform_(-8, 8).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, H, T, D, device=device).uniform_(-8, 8).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, H, T, D, device=device).uniform_(-8, 8).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, H, T, D, device=device).uniform_(-8, -6).to(dtype=dtype).requires_grad_(True)

    kk = torch.empty(B, H, T, D, device=device).uniform_(-8, 8)
    kk = torch.nn.functional.normalize(kk, dim=-1).to(dtype=dtype)

    a = -kk.clone().requires_grad_(True)  # -kk
    a_scale = torch.empty(B, H, T, D, device=device).uniform_(0, 0.1).to(dtype=dtype)
    b = (kk * a_scale).requires_grad_(True)  # kk*a

    # Create initial state
    initial_state = torch.zeros(B, H, V, N).to(torch.float64)

    # Clone inputs for the two paths we're testing
    q1, k1, v1, w1, a1, b1 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(
        True), w.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True), b.clone().detach().requires_grad_(True)
    q2, k2, v2, w2, a2, b2 = q.clone().detach().requires_grad_(True), k.clone().detach().requires_grad_(True), v.clone().detach().requires_grad_(
        True), w.clone().detach().requires_grad_(True), a.clone().detach().requires_grad_(True), b.clone().detach().requires_grad_(True)

    # Path 1: Using naive implementation with autograd

    output1, state1 = naive_recurrent_rwkv7(q1, k1, v1, w1, a1, b1, initial_state=initial_state.clone())

    output2, state2 = recurrent_rwkv7(q2, k2, v2, w2, a2, b2, 1.0, initial_state.clone())

    # Check forward pass equivalence
    output_diff = torch.max(torch.abs(output1 - output2)).item()
    state_diff = torch.max(torch.abs(state1 - state2)).item()

    print(f"\nAutograd Function test (forward):")
    print(f"  Max output difference: {output_diff:.6e}")
    print(f"  Max state difference: {state_diff:.6e}")

    # Create loss function to test backward pass
    def compute_loss(output, state):
        return output.sum()  # + state.sum()

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


test_autograd_function()
```
