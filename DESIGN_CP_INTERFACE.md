# KDA Context Parallel 接口设计方案

## 当前实现的问题

1. **全局状态管理**：通过 `_GDN_CP_CONTEXT` 全局变量和 `set_gdn_cp_context()`/`get_gdn_cp_context()` 传递上下文
2. **耦合度高**：`chunk_kda` 函数没有直接的 context 参数，依赖全局状态
3. **策略不灵活**：没有统一的参数来控制使用哪种 CP 策略（true cp, cp2tp, ring cp）
4. **层级混乱**：Layer 层和 Op 层的职责不清晰

## 设计哲学

1. **显性优于隐性**：context 应作为参数显式传递，不依赖全局状态
2. **策略可配置**：统一参数控制 CP 策略，kernel 支持不同的 level
3. **低耦合**：Layer 层负责准备 context，Op 层负责使用 context
4. **向后兼容**：保持现有接口不变或最小改动

## 核心设计

### 1. CP 策略枚举

```python
from enum import Enum
from typing import Optional
import torch.distributed as dist

class CPStrategy(Enum):
    """Context Parallel 策略"""
    NONE = "none"              # 不使用 CP
    TRUE_CP = "true_cp"        # True CP: 用自定义 Triton kernel 实现
    RING_CP = "ring_cp"        # Ring CP: 环形通信（预留）
    CP2TP_HEADS = "cp2tp_heads" # CP2TP: 在 attention heads 上做 tensor parallel
    CP2TP_HIDDEN = "cp2tp_hidden" # CP2TP: 在 hidden dim 上做 tensor parallel
```

### 2. Context 数据结构

```python
from dataclasses import dataclass

@dataclass
class CPContext:
    """Context Parallel 上下文"""
    group: Optional[dist.ProcessGroup] = None
    strategy: CPStrategy = CPStrategy.NONE
    rank: int = 0
    world_size: int = 1

    # True CP specific
    cu_seqlens: Optional[torch.Tensor] = None
    is_last_rank: bool = True
    is_first_rank: bool = True
    pre_num_ranks: int = 0
    post_num_ranks: int = 0
    pre_num_conv_tokens: int = 0
    kernel_size: Optional[int] = None

    # CP2TP specific
    cp_size: int = 1
    cp_rank: int = 0

    def is_valid(self) -> bool:
        """检查 context 是否有效"""
        if self.strategy == CPStrategy.NONE:
            return True
        if self.group is None or self.world_size <= 1:
            return False
        return True

    def copy_for_backward(self) -> 'CPContext':
        """复制 context 用于 backward"""
        import torch
        return CPContext(
            group=self.group,
            strategy=self.strategy,
            rank=self.rank,
            world_size=self.world_size,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            is_last_rank=self.is_last_rank,
            is_first_rank=self.is_first_rank,
            pre_num_ranks=self.pre_num_ranks,
            post_num_ranks=self.post_num_ranks,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
            kernel_size=self.kernel_size,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
        )
```

### 3. Context 构建工具

```python
def build_cp_context(
    cu_seqlens: Optional[torch.Tensor],
    group: Optional[dist.ProcessGroup],
    strategy: CPStrategy = CPStrategy.TRUE_CP,
    kernel_size: Optional[int] = None,
) -> CPContext:
    """构建 CPContext

    Args:
        cu_seqlens: 原始 cu_seqlens (before partition)
        group: process group
        strategy: CP 策略
        kernel_size: conv kernel size (for True CP)

    Returns:
        CPContext
    """
    if group is None or dist.get_world_size(group) <= 1:
        return CPContext(strategy=CPStrategy.NONE)

    if strategy == CPStrategy.TRUE_CP:
        # 调用现有的 get_cp_cu_seqlens 逻辑
        from fla.ops.common.cp_chunk_delta_h import get_cp_cu_seqlens
        context = get_cp_cu_seqlens(
            cu_seqlens=cu_seqlens,
            group=group,
            kernel_size=kernel_size
        )
        # 转换为新的 CPContext
        return CPContext(
            group=context.group,
            strategy=strategy,
            rank=dist.get_rank(group),
            world_size=dist.get_world_size(group),
            cu_seqlens=context.cu_seqlens,
            is_last_rank=context.is_last_rank,
            is_first_rank=context.is_first_rank,
            pre_num_ranks=context.pre_num_ranks,
            post_num_ranks=context.post_num_ranks,
            pre_num_conv_tokens=context.pre_num_conv_tokens,
            kernel_size=context.kernel_size,
        )
    elif strategy in [CPStrategy.CP2TP_HEADS, CPStrategy.CP2TP_HIDDEN]:
        return CPContext(
            group=group,
            strategy=strategy,
            rank=dist.get_rank(group),
            world_size=dist.get_world_size(group),
            cp_size=dist.get_world_size(group),
            cp_rank=dist.get_rank(group),
        )
    else:
        return CPContext(strategy=CPStrategy.NONE)
```

### 4. Layer 层接口

```python
class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        ...
        # 新增参数
        cp_strategy: CPStrategy = CPStrategy.NONE,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        self.cp_strategy = cp_strategy
        self.cp_group = cp_group

    def forward(self, hidden_states, ...):
        # 准备 context
        cp_context = None
        if self.cp_strategy != CPStrategy.NONE and self.cp_group is not None:
            cu_seqlens = kwargs.get('cu_seqlens')
            if cu_seqlens is None:
                # 自动生成 cu_seqlens
                seqlen = hidden_states.shape[1]
                world_size = dist.get_world_size(self.cp_group)
                total_seqlen = seqlen * world_size
                cu_seqlens = torch.tensor([0, total_seqlen], dtype=torch.int32, device=hidden_states.device)

            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens,
                group=self.cp_group,
                strategy=self.cp_strategy,
                kernel_size=self.conv_size if self.use_short_conv else None,
            )

        # 如果使用 CP2TP，做 all-to-all
        if cp_context.strategy == CPStrategy.CP2TP_HEADS:
            q, k, v = self._apply_cp2tp_heads(q, k, v, cp_context)

        # 调用 kernel，传递 context
        o, state = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            cp_context=cp_context,  # 新增参数
            ...
        )

        # 如果使用了 CP2TP，还原 all-to-all
        if cp_context.strategy == CPStrategy.CP2TP_HEADS:
            o = self._restore_cp2tp_heads(o, cp_context)
```

### 5. Op 层接口

```python
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ...
    # 新增参数
    cp_context: Optional[CPContext] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA chunk kernel with CP support

    Args:
        ...
        cp_context: CP 上下文，如果为 None 则自动创建（在 world_size > 1 时）
    """
    # 如果 cp_context 为 None 但需要 CP
    if cp_context is None and
       torch.distributed.is_initialized() and
       torch.distributed.get_world_size() > 1:
        # 自动获取默认 group
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=torch.distributed.group.WORLD,
            strategy=CPStrategy.TRUE_CP,  # 默认使用 True CP
            kernel_size=kernel_size,
        )

    # 根据策略选择实现
    if cp_context.strategy == CPStrategy.TRUE_CP:
        # 使用 True CP 实现
        return _chunk_kda_true_cp(q, k, v, g, beta, cp_context, ...)
    elif cp_context.strategy in [CPStrategy.CP2TP_HEADS, CPStrategy.CP2TP_HIDDEN]:
        # CP2TP 已经在 Layer 层处理，这里不需要额外操作
        return _chunk_kda_no_cp(q, k, v, g, beta, ...)
    else:
        # 不使用 CP
        return _chunk_kda_no_cp(q, k, v, g, beta, ...)

def _chunk_kda_true_cp(q, k, v, g, beta, cp_context: CPContext, ...):
    """True CP 实现"""
    # 直接使用 cp_context 中的字段
    # 不再调用 get_gdn_cp_context()
    initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
        k=kg,
        w=w,
        u=u,
        gk=g,
        cu_seqlens=cp_context.cu_seqlens,
        initial_state=initial_state,
        context=cp_context,  # 直接传递 CPContext
        use_exp2=True,
    )
    ...

def _chunk_kda_no_cp(q, k, v, g, beta, ...):
    """不使用 CP 的实现（保持现有逻辑）"""
    ...
```

### 6. 简化后的 cp_chunk_delta_h.py

```python
from dataclasses import dataclass
import torch.distributed as dist

@dataclass
class CPContext:
    """简化后的 CPContext，取代原来的 Context"""
    group: Optional[dist.ProcessGroup] = None
    strategy: str = "none"
    # True CP fields
    cu_seqlens: Optional[torch.Tensor] = None
    is_last_rank: bool = True
    is_first_rank: bool = True
    pre_num_ranks: int = 0
    post_num_ranks: int = 0
    pre_num_conv_tokens: int = 0
    kernel_size: Optional[int] = None

    # CP2TP fields
    cp_size: int = 1
    cp_rank: int = 0

    def copy_for_backward(self):
        return CPContext(
            group=self.group,
            strategy=self.strategy,
            cu_seqlens=self.cu_seqlens.clone() if self.cu_seqlens is not None else None,
            is_last_rank=self.is_last_rank,
            is_first_rank=self.is_first_rank,
            pre_num_ranks=self.pre_num_ranks,
            post_num_ranks=self.post_num_ranks,
            pre_num_conv_tokens=self.pre_num_conv_tokens,
            kernel_size=self.kernel_size,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
        )

# 保留原有的 kernel 函数，但修改签名，接受 CPContext
def chunk_gated_delta_rule_fwd_h_pre_process(
    k, w, u, gk, cu_seqlens, initial_state,
    context: CPContext,  # 使用 CPContext
    use_exp2=True,
):
    if context.strategy != "true_cp" or context.group is None:
        return initial_state
    # 原有逻辑...
```

## 优势

1. **清晰的分层**：
   - Layer 层：决定使用哪种 CP 策略，准备 context
   - Op 层：使用 context 执行具体的 kernel

2. **显性依赖**：
   - Context 作为参数传递，不依赖全局变量
   - 代码更容易理解和调试

3. **灵活配置**：
   - 通过 `cp_strategy` 参数统一控制
   - 每个 kernel 可以声明支持哪些 strategy

4. **向后兼容**：
   - 默认 `cp_strategy=CPStrategy.NONE`，不影响现有代码
   - 可以逐步迁移到新的接口

5. **可扩展**：
   - 容易添加新的 CP 策略
   - 容易添加新的 kernel 实现

## 使用示例

### 1. 不使用 CP（向后兼容）

```python
layer = KimiDeltaAttention(...)
# 不使用 CP，完全向后兼容
o, _ = layer(hidden_states)
```

### 2. 使用 True CP

```python
# 方式 1：通过 Layer 参数配置
cp_group = dist.new_group()
layer = KimiDeltaAttention(
    ...,
    cp_strategy=CPStrategy.TRUE_CP,
    cp_group=cp_group,
)

# forward 时自动处理 context
o, _ = layer(hidden_states)
```

### 3. 使用 CP2TP

```python
# 方式 1：通过 Layer 参数配置
cp_group = dist.new_group()
layer = KimiDeltaAttention(
    ...,
    cp_strategy=CPStrategy.CP2TP_HEADS,
    cp_group=cp_group,
)

o, _ = layer(hidden_states)  # 自动做 all-to-all
```

### 4. 直接使用 Op

```python
from fla.ops.kda import chunk_kda
from fla.ops.common.cp import CPStrategy, CPContext

# 手动准备 context
cp_context = CPContext(
    group=cp_group,
    strategy=CPStrategy.TRUE_CP,
    cu_seqlens=cu_seqlens,
)

o, state = chunk_kda(
    q, k, v, g, beta,
    cp_context=cp_context,  # 显式传递
    ...
)
```

### 5. 自动获取（但明确指定）

```python
# 在需要时自动创建 context，但由用户明确指定策略
def chunk_kda(..., cp_strategy: CPStrategy = CPStrategy.NONE):
    cp_context = None
    if cp_strategy != CPStrategy.NONE:
        # 自动获取 group 和构建 context
        cp_context = build_cp_context(..., strategy=cp_strategy)
    ...
```
