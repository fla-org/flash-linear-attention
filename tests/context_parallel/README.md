# CP(Context Parallel)

1. CP2TP refers to Context Parallelism with Tensor Parallelism on attention heads. When CP is mentioned in FLA kernels, it specifically denotes TP (head parallelism), which requires two all-to-all collective operations—one before and one after the attention computation.

2. Ring CP introduces sequential dependencies between ranks. For example, in a CP2 configuration, Rank1 must wait for Rank0 to complete before it can compute; during backpropagation, Rank0 must conversely wait for Rank1.

3. True CP resembles the CP implementation in FLA(KDA and GDN first), enabling genuinely parallel computation across all ranks with minimal communication overhead. However, this approach is more complex and requires sophisticated mathematical optimizations.
