"""
model_runner.py — 模型执行器（Model Runner）

本模块负责 **模型的加载、推理执行和 GPU 资源管理**，是推理引擎中最底层的计算模块。

==================== 核心职责 ====================

1. **模型加载与初始化**：加载 Qwen3 模型权重，初始化 CUDA 环境
2. **KV Cache 显存分配**：根据 GPU 剩余显存计算并分配 KV Cache 张量
3. **CUDA Graph 捕获**：预录制 decode 阶段的计算图，消除每步的 kernel launch 开销
4. **推理执行**：准备输入张量，调用模型前向传播，执行采样
5. **多 GPU 张量并行通信**：通过 SharedMemory + NCCL 协调多个 GPU 的计算

==================== 关键概念 ====================

1. **CUDA Graph（CUDA 图）**
   - 在 decode 阶段，每个序列每步只处理 1 个 token，计算量很小
   - 但每个 CUDA kernel 的 launch 都有 ~5-10μs 的 CPU 开销
   - 一个 Transformer 层包含数十个 kernel，总 launch 开销可能比实际 GPU 计算时间还长
   - CUDA Graph 将一系列 kernel 录制为一个图，replay 时只需一次 launch，
     将 CPU 端开销降低 10-100 倍
   - 限制：图中的张量形状和地址必须固定，因此需要为不同 batch size 分别录制

2. **Pin Memory（锁页内存）**
   - 通过 pin_memory=True 创建的 CPU 张量，其物理内存页被锁定，不会被操作系统换出
   - 从锁页内存到 GPU 的 DMA（Direct Memory Access）传输可以使用异步拷贝
   - 对比普通内存：普通内存需要先拷贝到临时锁页缓冲区，再传输到 GPU，多一次拷贝

3. **Slot Mapping（槽位映射）**
   - 将每个 token 映射到 KV Cache 张量中的具体存储位置
   - slot = block_id * block_size + offset_within_block
   - 用于 Triton kernel（store_kvcache_kernel）将 K/V 写入正确的显存位置

4. **Warmup（预热）**
   - 首次推理时 PyTorch 的 torch.compile、cuBLAS 算法选择等会产生一次性开销
   - 预热通过执行一次最大配置的前向传播来触发这些一次性操作
   - 同时用于测量 peak memory，为后续 KV Cache 分配提供基准
"""
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """模型执行器——负责 GPU 上的模型推理和资源管理。

    在多 GPU 场景下：
      - Rank 0（主进程）：完整功能，包括调度集成、采样、结果收集
      - Rank 1~N（工作进程）：只运行模型前向传播，通过 NCCL 与 Rank 0 同步
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """初始化 ModelRunner。

        Args:
            config: 全局配置。
            rank: 当前 GPU 的编号（0 为主进程，1~N 为工作进程）。
            event: 进程间同步机制：
              - Rank 0: list[Event]，用于通知所有工作进程
              - Rank 1~N: 单个 Event，用于等待主进程的指令
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size    # 张量并行的 GPU 总数
        self.rank = rank
        self.event = event

        # ==================== 分布式通信初始化 ====================
        # 使用 NCCL（NVIDIA Collective Communication Library）作为后端
        # NCCL 专为 GPU 间通信优化，支持 NVLink、PCIe、InfiniBand 等互连
        # "tcp://localhost:2333" 是 rendezvous 地址，所有进程通过它发现彼此
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)   # 将当前进程绑定到对应的 GPU

        # ==================== 模型加载 ====================
        # 临时修改默认数据类型和设备，使模型参数直接在 GPU 上以模型指定的精度创建
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)   # 通常为 bfloat16 或 float16
        torch.set_default_device("cuda")
        # 【学习笔记】为何这里显式指定 Qwen3ForCausalLM？
        #   nano-vllm 是教学/简化版项目，只支持 Qwen3 一种模型架构，因此直接硬编码。
        #   真实的 vLLM 使用 **模型注册表（Model Registry）+ 动态分发** 机制适配所有模型：
        #     1. 维护 _MODELS 字典：{"LlamaForCausalLM": LlamaForCausalLM, "Qwen3ForCausalLM": ...}
        #     2. 从 HuggingFace config.json 的 "architectures" 字段读取架构名（如 "Qwen3ForCausalLM"）
        #     3. 在注册表中查找对应的模型类，动态实例化：model_cls = _MODELS[arch]; model = model_cls(config)
        #     4. 所有模型类遵循统一的 forward() 接口，model_runner 无需知道具体模型类型
        #   如需扩展 nano-vllm 支持其他模型（如 Llama），可以：
        #     a. 参照 nanovllm/models/qwen3.py 编写新模型文件
        #     b. 在此处添加简单的 if-else 或字典映射，根据 hf_config.architectures[0] 选择模型类
        self.model = Qwen3ForCausalLM(hf_config)         # 创建模型结构（参数随机初始化）
        load_model(self.model, config.model)              # 从 SafeTensors 文件加载预训练权重
        self.sampler = Sampler()                          # 初始化采样器（Gumbel-Max Trick）

        # ==================== 预热与 KV Cache 分配 ====================
        self.warmup_model()          # 预热：触发 JIT 编译，测量 peak memory
        self.allocate_kv_cache()     # 根据剩余显存分配 KV Cache 张量

        # ==================== CUDA Graph 捕获 ====================
        if not self.enforce_eager:
            self.capture_cudagraph()  # 为 decode 阶段录制不同 batch size 的 CUDA Graph

        # 恢复默认设备和数据类型
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ==================== 多 GPU 共享内存通信初始化 ====================
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 创建共享内存段（2^20 = 1MB），用于传递调用指令
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()   # 等待所有进程就绪
            else:
                dist.barrier()   # 等待 Rank 0 创建完共享内存
                # 工作进程打开已有的共享内存段
                self.shm = SharedMemory(name="nanovllm")
                # 进入事件循环，等待主进程的指令
                self.loop()

    def exit(self):
        """清理 GPU 资源和进程通信。

        步骤：
        1. 关闭共享内存
        2. 删除 CUDA Graph（释放 GPU 内存）
        3. 同步 CUDA 流，确保所有 GPU 操作完成
        4. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()          # 确保所有进程都已关闭共享内存
            if self.rank == 0:
                self.shm.unlink()   # Rank 0 负责删除共享内存段
        if not self.enforce_eager:
            del self.graphs, self.graph_pool   # 释放 CUDA Graph 占用的显存
        torch.cuda.synchronize()    # 等待所有 GPU 操作完成
        dist.destroy_process_group()

    # ==================== 多 GPU 通信机制 ====================

    def loop(self):
        """工作进程（Rank 1~N）的事件循环。

        不断从共享内存读取主进程发送的指令并执行，
        直到收到 "exit" 指令退出。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """工作进程从共享内存读取一条指令。

        共享内存布局：
          [0:4]  — 4 字节小端整数，表示数据长度 n
          [4:n+4] — pickle 序列化的数据 [method_name, arg1, arg2, ...]

        通过 Event.wait() 阻塞等待主进程的信号。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()      # 阻塞直到主进程 set() 该 Event
        # 读取数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 反序列化：得到 [method_name, *args]
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()     # 清除事件，准备接收下一条指令
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程（Rank 0）向共享内存写入一条指令，并通知所有工作进程。

        Args:
            method_name: 要调用的方法名（如 "run", "exit"）。
            *args: 方法参数。
        """
        assert self.world_size > 1 and self.rank == 0
        # 将 [method_name, *args] 序列化为字节流
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")   # 写入长度
        self.shm.buf[4:n+4] = data                     # 写入数据
        # 通知所有工作进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一的方法调用入口。

        Rank 0：先通过 write_shm 广播指令给所有工作进程，再在本地执行。
        Rank 1~N：直接执行（指令已通过 read_shm 读取）。

        这确保所有 GPU 同时执行相同的操作，NCCL 集合通信才能正确同步。

        Args:
            method_name: 要调用的方法名。
            *args: 方法参数。

        Returns:
            方法的返回值（只有 Rank 0 的返回值有意义）。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ==================== 模型预热与资源分配 ====================

    def warmup_model(self):
        """模型预热——触发一次性的 JIT 编译开销，并测量 peak memory。

        构造一个最大配置的虚拟输入（max_num_batched_tokens 个 token）进行一次 prefill，
        目的：
        1. 触发 torch.compile 的首次编译（后续推理不再需要编译）
        2. 触发 cuBLAS 的算法自动选择（AutoTuning）
        3. 测量模型推理的 peak memory，为后续 KV Cache 分配提供基准

        【学习笔记】为什么需要 warmup？——精确测量激活值峰值以最大化 KV Cache 分配

          GPU 显存布局：
            ┌─────────────────────────────────────────────────┐
            │              GPU 总显存 (e.g. 24GB)               │
            ├───────────┬────────────┬────────────────────────┤
            │ CUDA 上下文 │  模型权重   │        剩余空间         │
            │  (~300MB)  │ (常驻~8GB) │                        │
            ├───────────┴────────────┼──────────┬─────────────┤
            │       常驻占用          │ 激活值峰值 │  KV Cache   │
            │                        │  (临时)   │   (核心!)   │
            └────────────────────────┴──────────┴─────────────┘

          推理时模型前向传播会产生临时的中间激活值（attention scores、FFN 中间结果等），
          这些激活值的大小取决于输入的 token 数量，且是动态变化的。
          如果不知道激活值最大能占多少显存，就无法确定能分配多少给 KV Cache：
            - 分多了 → OOM 崩溃
            - 分少了 → 浪费显存，能处理的序列数变少

          Warmup 用最大规模的假输入跑一次推理，精确测量"推理时临时激活值最多占多少显存"，
          从而把剩余显存尽可能多地分配给 KV Cache。

        【学习笔记】warmup 同时完成的三件事：
          ① torch.compile JIT 编译：首次执行时将 Python 代码编译为优化的 CUDA kernel，
            编译过程可能需要几秒到几十秒，提前完成后续推理就无此延迟。
          ② cuBLAS AutoTuning：矩阵乘法在不同尺寸下最优算法不同，
            cuBLAS 首次遇到特定尺寸时会自动测试多种算法并选出最快的。
          ③ 记录显存峰值：PyTorch 的 memory_stats()["allocated_bytes.all.peak"]
            精确记录前向传播过程中显存分配的最高水位线。
        """
        # 清理现场，准备测量——将峰值统计计数器归零，像测量水箱容量前先把水位标记归零
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造"最坏情况"的虚拟输入：系统能遇到的最大输入规模
        # 例如 max_num_batched_tokens=8192, max_model_len=4096 → 2 个长度为 4096 的序列
        # 用全 0 的 token_ids 是因为我们不关心输出，只关心显存峰值
        # （显存消耗只取决于张量形状，不取决于具体 token 值）
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)    # 执行一次 prefill 推理（丢弃结果）
        # 释放临时显存：前向传播的中间激活值已自动释放（临时张量），
        # empty_cache() 把 PyTorch 缓存池中的空闲块也还给 CUDA，
        # 确保后续 allocate_kv_cache() 能拿到最大的连续空间
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配 KV Cache 显存——根据 GPU 剩余空间动态计算可分配的块数。

        计算公式：
          可用显存 = total * gpu_memory_utilization - (模型权重 + warmup 时的 peak 激活值)
          每块字节数 = 2(K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
          可分配块数 = 可用显存 / 每块字节数

        其中 "peak 激活值" 通过 warmup 阶段的 peak memory 统计获得。

        KV Cache 张量布局：
          shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
          - 维度 0: 2 表示 Key 和 Value
          - 维度 1: Transformer 层索引
          - 维度 2: 物理块索引
          - 维度 3: 块内 token 偏移
          - 维度 4: KV 注意力头索引
          - 维度 5: 每个头的维度
        """
        config = self.config
        hf_config = config.hf_config

        # 获取 GPU 显存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free    # 当前已使用的显存（包括模型权重等常驻数据）
        # peak 是 warmup 期间的最大显存分配量（包括临时激活值）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # current 是当前的显存分配量（warmup 后临时数据已释放）
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算每个块的字节大小
        num_kv_heads = hf_config.num_key_value_heads // self.world_size   # 张量并行后每个 GPU 的 KV 头数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每块字节 = 2(K+V) * 层数 * 每块token数 * KV头数 * 头维度 * 每个元素的字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

        # 可分配块数 = (总显存 * 利用率 - 常驻占用 - warmup peak + 当前占用) / 每块字节
        # (peak - current) 是 warmup 时临时激活值的最大额外开销
        # used 是非 PyTorch 管理的显存（如 CUDA 上下文、NCCL 缓冲区等）
        #
        # 【学习笔记】公式拆解：
        #   KV Cache 可用显存 = GPU总显存 × 利用率上限
        #                     - used（CUDA上下文等非PyTorch占用）
        #                     - (peak - current)（推理时临时激活值的最大额外开销）
        #   其中：
        #     peak    = warmup 测到的峰值显存（模型权重 + 最大激活值）
        #     current = warmup 后的当前显存（只有模型权重）
        #     peak - current = 纯粹的临时激活值峰值开销
        #   这样算出的 KV Cache 大小既不会 OOM（预留了最坏情况的激活值空间），
        #   又不会浪费（精确测量而非保守估计）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 分配 KV Cache 张量
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # 将 KV Cache 张量的切片绑定到每个 Attention 层
        # 每层的 k_cache 和 v_cache 是整体张量的一个 view（视图），不额外占用显存
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]   # shape: (num_blocks, block_size, num_kv_heads, head_dim)
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ==================== 输入准备 ====================

    def prepare_block_tables(self, seqs: list[Sequence]):
        """将所有序列的 block_table 拼接为一个二维张量，并传输到 GPU。

        不同序列的 block_table 长度可能不同，短的用 -1 填充（padding）。
        Flash Attention 的 paged attention 接口需要这个张量来查找每个序列的 KV Cache 块。

        Args:
            seqs: 序列列表。

        Returns:
            block_tables: shape (num_seqs, max_num_blocks), dtype int32, 在 GPU 上。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 用 -1 填充到等长（-1 表示无效块，Flash Attention 会忽略）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """准备 prefill 阶段的所有输入张量。

        Prefill 使用 Flash Attention 的 **varlen（变长）** 接口：
        多个不同长度的序列打包（pack）成一个一维张量，通过 cu_seqlens 记录每个序列的边界。

        关键张量：
        - input_ids: 所有序列的 token id 拼接（跳过已缓存的 token）
        - positions: 每个 token 的绝对位置（用于 RoPE 位置编码）
        - cu_seqlens_q: Query 的累积序列长度（Cumulative Sequence Lengths）
            例如 [0, 5, 12] 表示第一个序列 query 长 5，第二个序列 query 长 7
        - cu_seqlens_k: Key 的累积序列长度（包含缓存命中的部分）
        - slot_mapping: 每个需要写入 KV Cache 的 token 对应的存储槽位

        当有前缀缓存命中时，cu_seqlens_q ≠ cu_seqlens_k：
        - Q 的长度 = 需要实际计算的 token 数（跳过缓存）
        - K 的长度 = 序列完整长度（包含缓存中的 token，Attention 需要看到完整上下文）

        Args:
            seqs: 本次 prefill 的序列列表。

        Returns:
            (input_ids, positions): 在 GPU 上的输入张量。
        """
        input_ids = []       # 所有序列的 input token id（跳过缓存部分）
        positions = []       # 每个 token 的绝对位置
        cu_seqlens_q = [0]   # Query 累积长度前缀和，首元素为 0
        cu_seqlens_k = [0]   # Key 累积长度前缀和
        max_seqlen_q = 0     # 单个序列的最大 query 长度（Flash Attention 需要）
        max_seqlen_k = 0     # 单个序列的最大 key 长度
        slot_mapping = []    # token → KV Cache 存储位置的映射
        block_tables = None  # 前缀缓存命中时需要的块表

        for seq in seqs:
            seqlen = len(seq)
            # 跳过已缓存的 token，只取需要计算的部分作为输入
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 位置从 num_cached_tokens 开始（缓存的 token 已有位置编码）
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens   # 需要计算 Attention 的 query 长度
            seqlen_k = seqlen                            # 完整上下文长度（Attention 需要看到所有历史 token）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:   # warmup 阶段没有 block_table，跳过
                continue

            # 计算 slot_mapping：只为需要写入 KV Cache 的 token（非缓存部分）计算槽位
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size   # 该块在 KV Cache 张量中的起始偏移
                if i != seq.num_blocks - 1:
                    end = start + self.block_size               # 非最后一个块：完整填充
                else:
                    end = start + seq.last_block_num_tokens     # 最后一个块：可能不满
                slot_mapping.extend(list(range(start, end)))

        # 如果 K 的总长度 > Q 的总长度，说明有前缀缓存命中，
        # 需要提供 block_tables 让 Flash Attention 通过分页查找缓存的 KV
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 将所有 CPU 张量异步传输到 GPU
        # pin_memory=True + cuda(non_blocking=True) 实现 CPU→GPU 的异步 DMA 传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 将上下文信息设置到全局 Context 中，供 Attention 层读取
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """准备 decode 阶段的所有输入张量。

        Decode 阶段每个序列只处理 1 个 token（上一步生成的 token），
        但需要通过 Paged Attention 访问该序列所有历史 token 的 KV Cache。

        关键张量：
        - input_ids: 每个序列的最后一个 token（shape: [batch_size]）
        - positions: 每个 token 的绝对位置（= 序列长度 - 1）
        - slot_mapping: 每个 token 对应的 KV Cache 存储槽位
        - context_lens: 每个序列的上下文长度（用于 Flash Attention 的 mask 计算）
        - block_tables: 所有序列的块表（用于分页查找 KV Cache）

        Args:
            seqs: 本次 decode 的序列列表。

        Returns:
            (input_ids, positions): 在 GPU 上的输入张量。
        """
        input_ids = []       # 每个序列的最新 token
        positions = []       # 每个 token 的位置
        slot_mapping = []    # 每个 token 在 KV Cache 中的存储位置
        context_lens = []    # 每个序列的完整上下文长度

        for seq in seqs:
            input_ids.append(seq.last_token)                # 上一步生成的 token 作为本步输入
            positions.append(len(seq) - 1)                   # 位置 = 序列长度 - 1（从 0 开始）
            context_lens.append(len(seq))                    # 完整上下文长度（包括刚追加的 token）
            # 计算新 token 的 KV Cache 存储槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # 异步传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # 设置全局 Context（decode 模式）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样所需的温度张量。

        只在 Rank 0 上执行（采样只需在一个 GPU 上完成）。

        Args:
            seqs: 序列列表。

        Returns:
            temperatures: shape (batch_size,), 在 GPU 上。
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ==================== 模型推理 ====================

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行模型前向传播。

        根据不同阶段选择不同的执行方式：
        1. Prefill：直接执行（Eager Mode），因为每次 shape 不同，无法使用 CUDA Graph
        2. Decode + enforce_eager=True：直接执行
        3. Decode + enforce_eager=False：使用 CUDA Graph 加速
        4. Decode + batch_size > 512：直接执行（超过预录制的最大 batch size）

        使用 CUDA Graph 时，需要将实际输入拷贝到预分配的固定地址张量中，
        然后 replay 预录制的计算图。图的输出同样在固定地址张量中。

        Args:
            input_ids: 输入 token id 张量。
            positions: 位置张量。
            is_prefill: 是否为 prefill 阶段。

        Returns:
            logits: 模型输出的 logits 张量 (batch_size, vocab_size)。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行模式：模型前向传播 → 计算 logits
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 加速模式
            bs = input_ids.size(0)   # 实际 batch size
            context = get_context()

            # 从预录制的图中选择一个 batch size >= bs 的最小图
            # 例如 bs=5 会选择 bs=8 的图（graph_bs = [1, 2, 4, 8, 16, 32, ...])
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 将实际数据拷贝到 CUDA Graph 绑定的固定地址张量中
            # 注意：多余的位置用 -1/0 填充，这些位置的计算结果会被忽略
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)          # -1 表示不写入 KV Cache
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放预录制的 CUDA Graph——一次 kernel launch 执行所有算子
            graph.replay()

            # 从固定地址输出张量中取出有效部分的 hidden states，计算 logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """完整的推理 pipeline：输入准备 → 模型推理 → 采样。

        Args:
            seqs: 本次 step 的序列列表。
            is_prefill: 是否为 prefill 阶段。

        Returns:
            token_ids: 采样得到的 token id 列表（只在 Rank 0 上有有效值）。
        """
        # 1. 准备输入张量
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 2. 准备采样温度（只在 Rank 0 上）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 3. 模型前向传播，得到 logits
        logits = self.run_model(input_ids, positions, is_prefill)
        # 4. 采样（只在 Rank 0 上，因为 logits 已通过 AllGather 收集到 Rank 0）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 5. 清理全局 Context
        reset_context()
        return token_ids

    # ==================== CUDA Graph 捕获 ====================

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为 decode 阶段捕获多个 batch size 的 CUDA Graph。

        CUDA Graph 的工作原理：
        1. 创建固定地址的输入/输出张量（图中引用的张量地址必须固定）
        2. 对每个 batch size 执行一次 warmup（确保 lazy 初始化完成）
        3. 使用 torch.cuda.graph() 上下文管理器录制计算过程
        4. 录制完成后，通过 graph.replay() 一次性执行所有录制的 kernel

        预录制的 batch size 列表：[1, 2, 4, 8, 16, 32, ..., max_bs]
        实际推理时选择 >= 当前 bs 的最小预录制图。

        Graph Pool（图内存池）：
        所有图共享同一个内存池，避免为每个图分别分配临时缓冲区，节省显存。
        使用 reversed() 从大到小录制，确保最大图先占用内存池，
        小图复用已有空间。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)    # CUDA Graph 支持的最大 batch size
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 创建固定地址的输入/输出张量（最大尺寸，实际使用时取前 bs 个）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)   # 模型输出的 hidden states

        # 预录制的 batch size 列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}        # batch_size → CUDAGraph 的映射
        self.graph_pool = None  # 所有图共享的内存池

        # 从大到小录制（大图先分配内存池空间）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置 decode 上下文（使用前 bs 个元素的切片）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # Warmup: 确保所有 lazy 初始化完成（如 cuBLAS workspace 分配）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 录制 CUDA Graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 首个图创建后获取内存池句柄，后续图共享
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()   # 确保录制完成
            reset_context()

        # 保存固定地址张量的引用，run_model() 中通过这些引用修改输入
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
