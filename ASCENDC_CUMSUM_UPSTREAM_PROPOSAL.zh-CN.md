# [RFC] 为 chunk_local_cumsum 接入可选 AscendC 后端

## 摘要

提议将 `flash-linear-attention-npu` 的 `chunk_local_cumsum` AscendC 实现迁入 FLA，以现有 `fla.ops.utils.chunk_local_cumsum` 为统一入口，复用 backend registry 和 verifier，并通过显式构建开关把 AscendC host 库、kernel 二进制及 OPP 元数据打入 FLA wheel。

首阶段只接管 Ascend950 上的 dense 三维前向计算，默认关闭 AscendC；不符合支持范围的调用继续使用已有实现。现有 PoC 已在 Ascend950PR_9579 上完成构建、安装和专项精度验证。本提案请求评审集成方式和合入范围，不主张默认替换 Triton-Ascend，也不主张已有性能收益。

**状态：设计评审草案，PoC 可运行；合入前仍需完成下文的许可、发布、兼容性和回归验收。**

## 1. 背景与目标

FLA 已有公共算子接口、Triton-Ascend 实现和 backend 分发机制；fla-npu 中的 AscendC 算子另有 host/kernel、ACLNN 桥接及 OPP 打包体系。直接要求上层调用 fla-npu 私有接口，会将布局、运行时初始化和后端选择暴露给模型调用方。

本次迁移希望建立一个可实际验证的单算子接入方式：

- 公共 Python 接口、参数默认值、输出布局不变。
- AscendC 源码位于所属算子的 backend 目录，公共桥接代码按职责集中管理。
- 复用 FLA 的 pip/setuptools 入口，未启用 native 构建时不要求 CANN。
- 通过 verifier 限定接管范围，保留现有 Triton-Ascend 路径。
- 从安装后的公共入口验证实际路由、数值结果和 stream 顺序。

首阶段不包含全仓 AscendC 迁移、通用算子自动发现、默认后端切换、算法重写、反向 kernel、模型训练支持或性能优化。

## 2. 算子契约和影响范围

### 2.1 公共接口与语义

公共入口保持原签名：

```python
chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    output_dtype=torch.float,
    chunk_indices=None,
    use_graph=False,
    **kwargs,
)
```

对于 dense 输入 `g[B,T,H]`，每个 chunk 独立计算：

```text
start(t) = floor(t / chunk_size) * chunk_size
y[b,t,h] = scale * sum(g[b,i,h], i=start(t)..t)
```

`scale=None` 等效于 1；支持末尾不足一个 chunk。PoC 保留来源 host/kernel 的原始实现，通过 adapter 将 `[B,T,H]` 转为连续 `[B,H,T]`，调用原生 `head_first=True` 路径，再转回连续 `[B,T,H]`。FP16/BF16 输入在原实现的 FP32 累加路径处理，首阶段只返回 FP32。

### 2.2 首阶段接管矩阵

| 维度         | AscendC 接管条件                                          | 其他调用                                |
| ---------- | ----------------------------------------------------- | ----------------------------------- |
| 设备         | Ascend950；已验证 Ascend950PR_9579                        | 保留现有分发                              |
| 输入         | 非空 dense 三维 `[B,T,H]`，torch.strided；支持非连续输入           | 4D、空输入等不由本后端接管                      |
| 输入 dtype   | FP32、FP16、BF16                                        | 保留现有分发                              |
| 输出 dtype   | FP32，包括默认值                                            | `None`、FP16/BF16 等不接管               |
| chunk_size | 64、128，允许尾块                                           | 保留现有分发                              |
| 方向         | `reverse=False`                                       | reverse 继续走现有实现                     |
| scale      | None 或有限且绝对值不超过 FP32 上限的 Python 标量                    | verifier 拒绝，交回现有分发                  |
| 元数据        | 无 cu_seqlens、chunk_indices，use_graph=False，无额外 kwargs | 保留现有分发                              |
| 梯度         | 没有启用输入梯度计算的调用                                         | grad enabled 且 g.requires_grad 时不接管 |

“保留现有分发”仅表示不由 AscendC 执行，不扩大原接口对非法输入的承诺。不同累加顺序允许产生现有数值预算内的差异，不承诺 bitwise 相同。

**训练与 graph 边界需要单独验收。** 梯度 guard 只判断当前调用状态，不能证明所有模型训练都不会进入此后端：例如自定义 autograd Function 的 forward 内可能关闭 grad。`use_graph=False` 也不等价于检测所有外部 graph capture。当前 PoC 不提供训练、capture 或完整模型兼容性结论；合入前应补充相应路由测试，必要时进一步收紧 verifier。

### 2.3 下游影响

直接调用该公共算子的消费者包括 Gated Delta Rule、KDA、GDN2、GLA 和 common fused chunk 路径。最终是否接管取决于实际 tensor 维度、元数据、梯度状态及参数，不能仅按模型名称判断。

- 未设置 `FLA_ASCENDC=1`：原有后端选择保持不变。
- 显式启用且通过 verifier：计算切换到 AscendC，增加布局转换及 native 加载依赖。
- 显式启用但不满足条件：继续尝试 Triton-Ascend/default 路径。
- 首阶段不更改模型参数、checkpoint/state-dict 或公共符号。

目前专项算子测试不能替代上述消费者的前向、状态及梯度回归；相关覆盖列入合入验收。

## 3. 设计方案

### 3.1 源码组织

```text
fla/ops/utils/backends/
├── triton_ascend/                  # 保留现有实现
└── ascendc/
    ├── __init__.py                 # backend 注册与策略
    ├── cumsum.py                   # verifier 和布局 adapter
    ├── README.md                  # 支持范围、构建与运行说明
    └── csrc/chunk_local_cumsum/
        ├── stable_chunk_local_cumsum.cpp
        ├── op_host/               # opdef、tiling、infershape、ACLNN
        └── op_kernel/             # AscendC kernel 和 tiling 数据结构

csrc/ascendc/                       # 共享 Stable ABI 桥接、注册和 CMake
build_helpers/                     # setuptools 命令及启动初始化
fla/utils/ascendc/                  # 包内 native loader
```

修改点集中在 `setup.py`、`pyproject.toml`、`scripts/build_packages.py`、utils backend 导入和对应测试；新增 `MANIFEST.in`、来源声明及许可证文本。通用 dispatcher 和公共算子函数体不增加 AscendC 专用条件。

单算子的 host/kernel 保持来源内容；原仓的完整构建脚本不整体迁入。共享层只承载本次已需要的加载、执行和打包功能，后续算子再按实际需求扩展。

### 3.2 后端选择与回退

沿用 owner 已有的 `@dispatch('utils')` / `@register('utils')` 注册方式：

| 属性           | AscendC                                 | Triton-Ascend |
| ------------ | --------------------------------------- | ------------- |
| backend_type | ascendc                                 | triton_ascend |
| priority     | -1                                      | 0             |
| 默认启用         | 否，显式 FLA_ASCENDC=1                      | NPU 环境默认启用    |
| 可用性          | NPU 环境、目标 build metadata 和包内 native 库存在 | NPU 环境        |
| 逐次检查         | 第 2 节 verifier                          | 现有 verifier   |

```text
公共 chunk_local_cumsum
  → input_guard
  → registry 按 priority 排序
  → available + enabled + verifier
      → 接受：执行 AscendC adapter
      → 拒绝：继续尝试 Triton-Ascend
  → 没有后端接受：执行原默认实现
```

已选中的 AscendC 在加载、分配或 launch 时出错，异常直接传播。回退只用于能力选择，不用于隐藏运行错误。`FLA_DISABLE_BACKEND_DISPATCH=1` 仍按现有含义绕过全部 backend 分发。

### 3.3 pip 构建与包内 OPP

提议采用显式 opt-in 的源码构建：

```bash
source /path/to/ascend-toolkit/set_env.sh
FLA_BUILD_ASCENDC=1 FLA_ASCENDC_SOC=ascend950 MAX_JOBS=4 \
  python -m pip wheel . --no-build-isolation --no-deps --wheel-dir dist/ascendc
```

前提是当前环境已装好兼容的 torch、torch_npu、CANN 开发工具和构建依赖。`--no-deps` 仅用于依赖已准备好的环境；`--no-build-isolation` 使构建能使用这些已有依赖。

```text
pip → setuptools.build_meta → setup.py → 自定义 build_py / bdist_wheel
  → CMake 构建 opdef、tiling、ACLNN、launcher
  → op_build 生成 INI 和 proto → 编译 proto
  → CANN 生成 dynamic 编译入口及 dtype 编译脚本
  → opc / AscendC 编译器生成 device .o、JSON
  → 生成 binary 索引并校验引用完整性
  → 组装 package-local OPP、launcher、build.json
  → 平台 wheel → pip 安装
```

生成物置于 `fla/utils/ascendc/_native/`，包括 `libfla_ascendc.so` 和 vendor OPP 下的 opapi、proto、tiling、kernel、索引。CANN/PyTorch 动态库仍是外部依赖，不修改系统 OPP，不要求安装另一套 fla-npu Python 包。

不启用 `FLA_BUILD_ASCENDC` 时保持纯 Python wheel；sdist 包含可重建的源文件闭包，不包含机器相关的 `_native`。native 编译失败即构建失败，不退化为成功生成纯 Python wheel。

拆包脚本已将必要源码带入 `fla-core`，但 native core 分包尚未上板验收。native editable install 暂不支持。本次平台标签为 `py3-none-linux_x86_64`，不声明 manylinux 或跨 CANN/PyTorch 版本兼容性。

### 3.4 初始化与运行调用链

当前 PoC 在 native wheel 中安装标准库 bootstrap 和 `.pth`。进程以 `FLA_ASCENDC=1` 启动时，bootstrap 在 torch_npu 初始化前，将包内 vendor 目录加入 `ASCEND_CUSTOM_OPP_PATH`；没有开关时不修改该路径，也不导入 torch。

```text
启动 Python → .pth 配置包内 OPP
导入 FLA → 注册 backend
公共入口 → verifier → [B,T,H] 转 [B,H,T]
  → 懒加载包内 opapi / launcher，校验 source_hash
  → torch.ops.fla_ascendc.npu_chunk_local_cumsum
  → Stable ABI boxed adapter
  → aclnnChunkLocalCumsumGetWorkspaceSize
  → 分配并维护 workspace / executor 生命周期
  → aclnnChunkLocalCumsum，在当前 NPU stream 提交
  → CANN tiling / binary 索引 → 预编译 AscendC kernel
  → FP32 结果转回 [B,T,H]
```

AscendC kernel 在构建时编译，运行时不 JIT。桥接层获取当前设备 raw stream 并维持 torch_npu 队列顺序；当前实现涉及 torch_npu 私有接口，因此其兼容性不能由 Stable ABI 名称推导。

`.pth` 对 Python 进程有启动期影响，且 `ASCEND_CUSTOM_OPP_PATH` 是进程级环境变量。建议维护者明确是否接受该 opt-in 初始化机制；如果要求替换，应先验证替代方案在 torch_npu/CANN 初始化前生效，不能简单挪到首次调用内。

## 4. 备选方案与取舍

| 方案                        | 优点                          | 代价与本提案选择                                 |
| ------------------------- | --------------------------- | ---------------------------------------- |
| 当前方案：源代码随 FLA，按需编译并打包 OPP | 单一公共接口、源码与产物可追踪、安装后无需另装 OPP | 增加 native 构建、许可和版本矩阵维护；作为首阶段建议           |
| 独立 fla-npu 插件包            | FLA 主包保持轻量，原生发布可独立进行        | 跨包版本协同和部署更复杂；若主仓不能接受许可或 native 依赖，可采用此方向 |
| 首次调用 JIT 编译 AscendC       | 减少预编译分发组合                   | 运行环境需编译工具，首调用有成本，缓存/并发机制更复杂；首阶段不采用       |
| 重写 kernel 直接处理 BTH 布局     | 有机会减少复制开销                   | 扩大数值与性能评审范围；本次保留原 kernel，后续以实测决定         |

不建议将“按检测到的设备全局选择实现”替代逐算子 verifier；同一进程仍需要对不同输入分别选择后端。

## 5. PoC 证据

代码基线与验证身份：

- FLA 基线：`1fe52b5d05c53601885610fab3d39157647d836d`；候选为其上的 `ascendc-cumsum-poc` 工作区修改，尚无可供上游审阅的候选提交 SHA。
- 来源仓：[flashserve/flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)，版本 `066f47e4f1c41f8a2227142f3114bc60c4554dee`。
- 验证日期：2026-10-15；Linux x86_64、Ascend950PR_9579、CANN 9.2.0-beta.2、Python 3.12.3、torch 2.9.0+cpu、torch_npu 2.9.0.post2、triton-ascend 3.2.2。

| 验证                     | 实际结果                                      | 覆盖边界                                           |
| ---------------------- | ----------------------------------------- | ---------------------------------------------- |
| 完整仓 native wheel 构建、安装 | 通过；生成 9 个 device .o；593 个安装文件与最终 wheel 一致 | 单一工具链和设备环境                                     |
| 公共 API Demo            | 6/6 通过，逐例确认真实 native 调用                   | 3 dtype × 2 chunk_size                         |
| dense 数值测试             | 432/432 通过                                | 9 shape × 2 chunk × 3 dtype × 4 scale × 连续/非连续 |
| 回退测试                   | 5/5 通过                                    | 4D、reverse、varlen、FP16 输出、chunk_size=32        |
| stream 测试              | 1/1 通过                                    | 非默认 stream 上 producer → cumsum → consumer      |
| 专项合计                   | 438 passed，0 failed，0 skipped             | 另有 38 项原有测试未被本次选择                              |

reference 为输入转 CPU double 后逐 chunk cumsum，再缩放并转 FP32；沿用 `assert_close(..., 1e-3)`，另加 RMS ratio `< 1e-3` 的硬断言。没有放宽容差。6 个 Demo 的最大绝对误差为 `3.814697265625e-6`，最大 RMS ratio 为 `1.1774982056067529e-7`；这些数值不能扩展为全输入域保证。

```text
最终 wheel SHA256:
cd9d9820612e579ffb4bec5d406576a33ff09684a10381a81083c56756f04455
native 源码 hash:
8cfe31bb3e69e52b2830b3c9e206c43739fc328919f88e8379de8a56c49a8286
```

目前没有同硬件 Triton-Ascend/AscendC 延迟对比。BTH↔BHT 两次转换、首次加载和稳态执行成本必须分别测量；不能由正确性通过推断性能更好。

实际构建验证已经解决生成器源码布局、setuptools/wheel 命令兼容、proto 静态注册及显式链接依赖、raw stream 参数和 CANN Python 环境路径问题。评审实现应保留完整 kernel 索引检查、proto 严格链接和安装产物一致性检查，详细故障日志无需放入主 PR 描述。

## 6. 请求上游确认的决策

| 决策    | 本提案建议                                         | 需要确认的原因                                                                               |
| ----- | --------------------------------------------- | ------------------------------------------------------------------------------------- |
| 集成边界  | owner backend + 共享 native runtime；公共 API 不变   | 符合当前算子所有权，避免第二套接口                                                                     |
| 默认策略  | 默认关闭，显式启用后按 verifier 优先尝试                     | 让影响范围限于 opt-in 调用                                                                     |
| 源码许可  | 保留每个文件的原许可及 NOTICE，先确认主仓接收政策                  | host/kernel 使用 CANN Open Software License v2，桥接含 BSD-3-Clause 和 PyTorch 许可；不能统一改标 MIT |
| 初始化   | 首阶段采用仅 native wheel 安装、环境开关控制的 .pth bootstrap | 必须早于 CANN 初始化，但存在进程级影响，需要导入顺序和多 OPP 共存测试                                              |
| 工具链支持 | 首阶段只声明已验证的版本组合                                | 私有 stream/queue 接口及平台 wheel 标签不保证 ABI/版本通用性                                           |
| 发布方式  | 先支持 opt-in 源码 wheel 构建；预编译发布另行决定              | 当前通用 Linux 标签不区分 SoC/CANN，不能直接把多种 native 变体混发为同名产物                                    |
| 头文件维护 | 固定 PyTorch 头文件闭包、保留 manifest 和来源              | 需要明确升级、同步和回归责任                                                                        |
| 分包策略  | native 归属 fla-core，先补齐安装验收                    | 本次仅完整仓 wheel 已上板，不能据此宣称分包已通过                                                          |

许可这一项请求确认项目接收与分发政策，本提案不替代法律合规判断，也不宣称不同许可已经完成兼容性审查。

## 7. 合入计划与验收门槛

建议先完成 RFC 设计确认，再提交一个默认关闭、可构建可运行的单算子集成 PR。构建、运行时、verifier 和回归测试共同构成最小闭环；不同时引入其他算子或改变数值算法。若代码量超过仓库阈值，需按模板说明单一目的，并把第三方头文件闭包与自有代码分组评审。

| 阶段     | 验收内容                                                                 | 当前状态                  |
| ------ | -------------------------------------------------------------------- | --------------------- |
| 设计对齐   | 确认第 6 节的目录、许可、初始化和发布选择；关联既有或新建 RFC                                   | 待上游评审                 |
| 候选固定   | 形成可审阅提交 SHA，记录所有源码与验证产物对应关系                                          | 当前为工作区 PoC            |
| 包装闭环   | 纯 Python wheel/sdist 不依赖 CANN；native 源码重建和 wheel 安装；native core 分包验收 | 完整仓已验证，native core 待补 |
| 数值及路由  | 保留 438 项覆盖，补默认参数等价性、各拒绝边界、旧后端 parity                                 | 已有专项证据，完整矩阵待补         |
| 生命周期   | 新进程导入顺序、关闭/缺包、同机多 OPP、设备切换、队列及异步生命周期                                 | 单非默认 stream 已验证，其他待补  |
| 数值边界   | 生产 gate 分布与 scale、极值/溢出、NaN/Inf 的既有行为对照                              | 当前有限随机输入不足以覆盖，待补      |
| 下游兼容   | 受影响消费者前向、状态和梯度；确认自定义 autograd forward 与 graph capture 路由             | 未完成                   |
| 性能     | 同设备同输入对比 Triton-Ascend；计入布局转换，区分首次调用与稳态                              | 未测量                   |
| CI 与维护 | host 打包/路由检查、可用 NPU runner 的安装后专项测试、版本矩阵维护责任                         | 待确认及接入                |

验收不以“默认关闭”代替数值或构建检查；扩大 dtype、shape、SoC 或训练范围时必须增加对应证据。关闭 `FLA_ASCENDC` 并重新启动进程可恢复现有后端选择，无需修改模型代码；该回退方式应作为发布说明的一部分。

## 8. 评审材料与提交准备

本草案可作为设计 issue/RFC 正文。后续代码 PR 使用仓库现有模板，正文保留行为变化、影响范围、测试、性能和未完成项；完整文件清单及故障记录作为补充材料。

- 实现与验证细节：仓库内 `ASCENDC_CUMSUM_MIGRATION.zh-CN.md`。
- 已保存证据：本地 `build/ascendc-validation/result.json`、`logs/accuracy-cann-env.xml`、`logs/demo-cann-env.json` 及构建/安装日志。这些路径不是上游可访问链接，正式提交时需附去除机器路径等环境信息后的报告或 CI artifact。
- 重复工作检查：本次尝试查询上游 open PR/issue 的 `AscendC` 关键词，但本机 gh 未认证，未完成查询；发布前必须补查并关联现有讨论，不能据此声称没有重复工作。

本草案尚未发布到上游，未创建 issue/PR。
