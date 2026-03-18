"""
scheduler.py — 推理调度器（Scheduler）

本模块实现了 **Continuous Batching（连续批处理）** 的核心调度逻辑。

==================== 核心概念 ====================

1. **Continuous Batching（连续批处理）**
   传统的 Static Batching（静态批处理）要求一个 batch 内所有序列的长度相同，
   且必须等所有序列都完成后才能处理下一批。这导致短序列需要等待长序列，效率低下。

   Continuous Batching 允许：
   - 不同长度的序列可以在同一个 batch 中混合处理
   - 序列完成后立即释放资源，新序列可以随时加入
   - 每个 step 动态决定处理哪些序列

2. **Prefill 与 Decode 两阶段**
   - **Prefill（预填充）**：处理新请求的 prompt，一次性计算所有 prompt token 的 KV Cache。
     这是计算密集型（Compute-Bound）操作，主要受限于 GPU 算力。
   - **Decode（解码）**：逐 token 生成，每步只处理一个新 token，但需要读取所有历史 KV Cache。
     这是访存密集型（Memory-Bound）操作，主要受限于显存带宽。

   调度器优先处理 prefill（waiting 队列非空时），确保新请求尽快开始生成。

3. **Preemption（抢占）**
   当 KV Cache 显存不足以容纳所有 running 序列时，调度器会抢占（中断）部分序列：
   - 释放被抢占序列的 KV Cache
   - 将其放回 waiting 队列的头部（优先重新调度）
   - 被抢占的序列后续需要重新 prefill（重新计算 KV Cache）
   这是一种 **Recomputation（重计算）** 策略，用空间换时间保证系统可用性。

4. **调度策略**
   本调度器采用 **Prefill-First** 策略：
   - 每个 step 先尝试从 waiting 队列调度 prefill
   - 如果 waiting 队列为空，则调度 running 队列中的序列进行 decode
   - 同一个 step 不会同时包含 prefill 和 decode 序列
     （因为 prefill 和 decode 的计算模式不同，混合会降低效率）
"""
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """推理调度器，负责决定每个 step 处理哪些序列。

    Attributes:
        max_num_seqs: 单个 step 的最大序列数（decode batch size 上限）。
        max_num_batched_tokens: 单个 step 的最大 token 总数（prefill 阶段的显存上限）。
        eos: EOS token id，用于判断序列是否自然结束。
        block_manager: KV Cache 块管理器，负责物理块的分配和释放。
        waiting: 等待队列（FIFO），存放尚未开始 prefill 的新序列。
        running: 运行队列，存放已分配 KV Cache、正在生成中的序列。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()     # 等待 prefill 的序列队列
        self.running: deque[Sequence] = deque()     # 正在 decode 的序列队列

    def is_finished(self):
        """判断是否所有请求都已处理完毕（两个队列都为空）。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加一个新的推理请求到等待队列。

        Args:
            seq: 新创建的序列对象。
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """核心调度方法——决定本次 step 处理哪些序列。

        调度逻辑：
        1. 优先调度 prefill（waiting 队列非空时）
        2. 否则调度 decode（running 队列非空时）
        3. 同一个 step 不混合 prefill 和 decode

        Returns:
            (scheduled_seqs, is_prefill):
              - scheduled_seqs: 本次 step 要处理的序列列表
              - is_prefill: True 表示本次是 prefill 阶段，False 表示 decode 阶段
        """
        # ==================== Prefill 调度 ====================
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]    # 查看（但不弹出）队首序列

            # 检查约束：
            #   1. 加入该序列后总 token 数是否超过 max_num_batched_tokens: 单个 step 的最大 token 总数（prefill 阶段的显存上限）
            #   2. BlockManager 是否有足够的空闲块分配给该序列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break   # 资源不足，停止调度更多 prefill 序列

            num_seqs += 1
            # 分配 KV Cache 块（同时执行前缀缓存匹配）
            self.block_manager.allocate(seq)
            # 实际需要计算的 token 数 = 总 token 数 - 已缓存的 token 数（前缀缓存命中部分）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()         # 从等待队列弹出
            self.running.append(seq)       # 加入运行队列
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True    # 有 prefill 序列，返回 is_prefill=True

        # ==================== Decode 调度 ====================
        # 当 waiting 队列为空（或 prefill 资源不足），调度 running 队列中的序列进行 decode
        #
        # 【学习笔记】为何需要抢占（Preemption）？
        #   在 decode 阶段，每个序列每一步生成一个新 token，需要写入 KV Cache。
        #   当序列刚好跨入新 block 边界时（len(seq) % block_size == 1），需要分配新物理块。
        #   但 GPU 显存有限，物理块总数固定。当多个序列同时 decode 消耗块时，
        #   可能出现空闲块耗尽的情况。此时必须通过抢占释放一些序列的块，
        #   给更优先的序列腾出空间。这是一种"用重计算换空间"的策略，保证系统不会死锁。
        #
        # 【学习笔记】抢占的完整流程：
        #   ① 从 running 队列头部取出序列 seq（头部 = 最早开始的 = 优先级最高）
        #   ② 检查 can_append(seq)：当前块是否有空间，或是否有空闲块可分配
        #   ③ 若空间不足，从 running 队列尾部取出序列抢占（LIFO 策略——
        #      后来的序列已生成的 token 最少，重新计算代价最小）
        #   ④ 极端情况：只剩 seq 自己也不够，抢占自身，放回 waiting 队列
        #   ⑤ 空间足够后，正常将 seq 加入本次调度列表
        #
        # 【学习笔记】关于 while...else 语法：
        #   Python 的 while...else 中，else 块仅在 while 条件正常变为 False 时执行，
        #   若被 break 打断则不执行。因此只有 can_append 返回 True 时才会走到 else 分支。
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()   # ① 从运行队列头部取序列（FIFO，优先级最高）

            # ② 检查 BlockManager 是否有足够的空间追加一个 token
            while not self.block_manager.can_append(seq):
                # 空间不足：需要抢占其他序列来释放块
                if self.running:
                    # ③ 抢占运行队列尾部的序列（后来的序列优先被抢占，类似 LIFO 策略）
                    #    选择尾部的原因：后来的序列已生成的 token 最少，重新 prefill 代价最小，
                    #    优先保护先来的序列，减少整体延迟
                    self.preempt(self.running.pop())
                else:
                    # ④ 只剩最后一个序列也无法追加，抢占自身并跳出
                    #    这说明即使整个系统只服务这一个序列，可用块也不够了
                    self.preempt(seq)
                    break
            else:
                # ⑤ while 正常结束（can_append 返回 True），可以追加
                num_seqs += 1
                # 更新 BlockManager 状态（可能分配新块或注册已填满块的哈希）
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs   # 至少应有一个序列被调度（否则系统死锁）
        # 将调度的序列放回运行队列头部（保持 FIFO 顺序）
        # extendleft + reversed 保证原始顺序不变
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False   # decode 阶段

    def preempt(self, seq: Sequence):
        """抢占一个正在运行的序列：释放其 KV Cache，将其放回等待队列。

        被抢占的序列后续需要重新 prefill（重新计算 KV Cache）。
        放到等待队列的头部（appendleft）确保它被优先重新调度。

        这是一种 **Recomputation（重计算）** 抢占策略：
        相比 **Swap（交换到 CPU 内存）** 策略更简单，
        代价是需要重新 prefill，但避免了 CPU-GPU 数据传输的延迟。

        【学习笔记】抢占时 GPU 端 KV Cache 数据的处理：
          抢占时 deallocate() 只操作 CPU 端元数据（引用计数、空闲队列、块表），
          GPU 显存中的 KV 数据 **不会被清除**（类似 OS 释放内存页不清零）。
          这样做有两个好处：
          1. 避免不必要的 GPU 显存清零操作（节省带宽）
          2. 释放后的块的哈希信息保留在 hash_to_block_id 索引中，
             后续新请求（或被抢占序列重新 prefill 时）如果前缀匹配，
             可以直接复用 GPU 上残留的 KV 数据（Prefix Caching 跨请求生效）

        【学习笔记】两种抢占策略对比：
          - Recomputation（本项目采用）：释放 KV Cache → 重新 prefill。简单，无 CPU-GPU 传输。
          - Swap（vLLM 另一种策略）：将 KV Cache 交换到 CPU 内存 → 恢复时传回 GPU。
            保留已有计算结果，但增加了 CPU-GPU 数据传输延迟和 CPU 内存占用。

        Args:
            seq: 要抢占的序列。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)   # 放到等待队列头部，优先重新调度

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """后处理：将模型生成的 token 追加到各序列，并检查终止条件。

        在每个 step 的模型推理完成后调用。

        终止条件（满足其一即停止生成）：
        1. 模型输出了 EOS token 且 ignore_eos=False
        2. 已生成的 completion token 数达到 max_tokens

        序列终止后释放其 KV Cache 并从运行队列中移除。

        Args:
            seqs: 本次 step 处理的序列列表。
            token_ids: 模型为每个序列生成的 token id 列表（一一对应）。
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加新生成的 token 到序列
            seq.append_token(token_id)

            # 检查终止条件
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)   # 释放 KV Cache
                self.running.remove(seq)             # 从运行队列移除
