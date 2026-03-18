"""
block_manager.py — KV Cache 块管理器（含前缀缓存）

本模块实现了 **Paged Attention** 的核心内存管理机制和 **Prefix Caching** 优化。

==================== 核心概念 ====================

1. **Paged Attention（分页注意力）**
   借鉴操作系统虚拟内存的分页（Paging）思想：
   - 将 KV Cache 的显存空间划分为固定大小的 **物理块（Block）**，每块容纳 block_size 个 token 的 K/V
   - 每个序列维护一个 **块表（Block Table）**，记录逻辑块到物理块的映射（类似页表）
   - 好处：消除了传统实现中因预分配最大长度 KV Cache 导致的显存浪费（内部碎片），
     实现按需分配，显存利用率接近 100%

2. **Prefix Caching（前缀缓存）**
   当多个请求共享相同的前缀（如系统提示词、Few-Shot 示例等）时，
   它们的 KV Cache 内容是完全相同的。前缀缓存机制通过：
   - 对每个填满的块计算链式哈希值 hash = xxhash64(prev_hash || token_ids)
   - 维护 hash → block_id 的全局索引
   - 新请求分配块时，先查询索引是否有匹配的块可以复用
   从而避免重复计算相同前缀的 Attention，大幅加速 prefill 阶段。

3. **引用计数（Reference Counting）**
   每个物理块维护 ref_count 引用计数：
   - 分配时设为 1
   - 每次被新序列共享（缓存命中）时 +1
   - 序列释放时 -1
   - 降为 0 时块被回收到空闲队列
   这使得多个序列可以安全地共享同一物理块。

4. **链式哈希（Chained Hashing）**
   块的哈希值不仅取决于自身的 token_ids，还依赖前一个块的哈希值：
     hash_i = xxhash64(hash_{i-1} || token_ids_i)
   这确保了只有前缀 *完全* 匹配时哈希才会相同——
   即使两个不同请求碰巧在某个块位置有相同的 token_ids，
   只要它们之前的某个块不同，后续所有块的哈希都不会匹配。

==================== GPU 端数据结构 ====================

KV Cache 在 GPU 显存中的存储布局为一个 6 维张量：
  shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
  - 2: Key 和 Value 两个缓存
  - num_layers: Transformer 层数
  - num_blocks: 物理块总数
  - block_size: 每块容纳的 token 数
  - num_kv_heads: KV 注意力头数（GQA/MQA 场景下可能少于 Q 头数）
  - head_dim: 每个注意力头的维度

Block 对象是 CPU 端的元数据，通过 block_id 索引到 GPU 端张量的第 3 维。
"""
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV Cache 物理块的 CPU 端元数据。

    每个 Block 对应 GPU 显存中一段固定大小的 KV Cache 存储区域，
    用于存放 block_size 个 token 的 Key/Value 缓存。

    Attributes:
        block_id: 块的唯一标识（即在全局块数组中的下标），对应 GPU 端 KV Cache 张量的索引。
        ref_count: 引用计数——有多少个序列正在使用该块。
            当 ref_count > 1 时表示该块被多个序列共享（前缀缓存复用）。
            当 ref_count == 0 时块可被回收到空闲队列。
        hash: 块内容的链式哈希值。
            -1 表示该块尚未被填满，不可用于前缀缓存匹配。
            填满后由 compute_hash() 计算并设置。
        token_ids: 该块中实际存放的 token id 列表，用于缓存命中时的内容校验
            （防御哈希冲突：虽然 xxhash64 冲突概率极低，仍需校验确保正确性）。
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """当块被填满时，更新其哈希值和 token_ids，使其可用于前缀缓存查找。

        Args:
            hash: 该块的链式哈希值。
            token_ids: 该块包含的 token id 列表。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，准备被分配给新的使用者。
        引用计数初始化为 1（分配即表示有一个使用者），
        清空哈希和 token_ids（该块的内容已不再有效）。
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """KV Cache 块管理器——Paged Attention + Prefix Caching 的核心实现。

    核心职责：
    1. 管理所有物理块的分配（allocate）与释放（deallocate），类似操作系统的内存分配器
    2. 实现基于链式哈希的前缀缓存——当不同请求共享相同的前缀 token 时，
       复用已有的 KV Cache 块，避免重复计算 Attention，加速 prefill 阶段
    3. 在 decode 阶段动态追加块（may_append），支持序列长度的逐步增长
    4. 通过引用计数支持多序列共享同一物理块

    Attributes:
        block_size: 每个块容纳的 token 数量。
        blocks: 所有物理块对象的数组（按 block_id 索引）。
        hash_to_block_id: 链式哈希值 → 块 ID 的映射（前缀缓存的全局索引）。
        free_block_ids: 空闲块 ID 的双端队列（FIFO 策略分配）。
        used_block_ids: 已使用块 ID 的集合（用于 O(1) 判断块是否在使用中）。
    """

    def __init__(self, num_blocks: int, block_size: int):
        """初始化块管理器。

        Args:
            num_blocks: 物理块总数，由可用显存计算得出。
            block_size: 每个块容纳的 token 数。
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算一个块的链式哈希值。

        链式哈希确保前缀完全匹配时才会产生相同哈希：
          hash_i = xxhash64(hash_{i-1} || token_ids_i)

        使用 xxhash64 而非 Python 内置 hash()，原因：
          - xxhash64 是非加密哈希函数，速度极快（数 GB/s 级别）
          - 64 位输出空间，冲突概率约 1/2^64，实际可忽略
          - Python hash() 在不同进程间不一致（随机种子），不适合持久化索引

        Args:
            token_ids: 当前块包含的 token id 列表。
            prefix: 前一个块的哈希值。-1 表示这是第一个块（没有前驱）。

        Returns:
            64 位无符号整数哈希值。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前一个块的哈希值编码为 8 字节小端序，纳入当前哈希计算
            # 这形成了哈希链：每个块的哈希隐式包含了之前所有块的信息
            h.update(prefix.to_bytes(8, "little"))
        # 将 token_ids 转为 numpy 数组的原始字节表示（int64，每个 token 8 字节）
        # tobytes() 直接提供连续内存的字节视图，高效输入到哈希函数
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个指定 ID 的物理块。

        将块从空闲队列移入已使用集合，并重置块的状态（ref_count=1, hash=-1）。

        Args:
            block_id: 要分配的块 ID。

        Returns:
            分配后的 Block 对象。
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0   # 确保该块当前没有被任何序列引用
        block.reset()                 # 重置状态：ref_count=1, hash=-1, token_ids=[]
        self.free_block_ids.remove(block_id)   # 从空闲队列中移除
        self.used_block_ids.add(block_id)      # 加入已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放一个物理块，将其从已使用集合移回空闲队列。

        注意：这里 **不会** 重置块的 hash 和 token_ids！
        这是前缀缓存能跨请求生效的关键——
        块被释放后，其内容信息仍保留在 hash_to_block_id 索引中。
        后续新请求如果有相同前缀，可以直接通过哈希查找复用该块，
        重新将其从空闲队列分配出去（_allocate_block 时才会重置状态）。

        【学习笔记】释放 ≠ 清零——与操作系统内存管理的类比：
          就像 OS 释放内存页时不会把页面内容清零一样，释放 KV Cache 块只是
          在 CPU 端元数据上做标记，GPU 显存中的 KV 数据原封不动留在那里。
          具体来说，这里只做了两件事：
            1. used_block_ids.remove(block_id) —— 标记该块不再被使用
            2. free_block_ids.append(block_id) —— 把块 ID 放回空闲队列
          GPU 端没有任何操作。旧的 KV 数据会在块被重新分配给新序列时，
          由 attention kernel 在 prefill/decode 阶段直接覆盖写入新的 K/V 值。

        【学习笔记】这种设计的好处：
          1. 避免不必要的 GPU 显存清零操作（cudaMemset 本身有带宽开销）
          2. hash_to_block_id 索引保留，使得 Prefix Caching 能跨请求生效——
             被释放的块如果后续被新请求的前缀匹配命中，可以直接复用 GPU 上
             残留的 KV 数据，跳过重复的 Attention 计算

        Args:
            block_id: 要释放的块 ID。
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """判断当前空闲块数量是否足够为该序列分配所有所需的块。

        在 prefill 调度前调用，确保有足够的物理块来存储整个序列的 KV Cache。

        Args:
            seq: 待分配的序列。

        Returns:
            True 表示空闲块充足，可以分配。
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """为一个新序列分配 KV Cache 块——prefill 阶段调用。

        这是前缀缓存的核心匹配逻辑：
        1. 逐块计算序列 token_ids 的链式哈希
        2. 在 hash_to_block_id 中查找是否有相同哈希的块（前缀缓存命中）
        3. 命中：复用已有块（增加引用计数），跳过这些 token 的 Attention 计算
        4. 未命中（cache_miss）：从空闲队列分配新块
        5. 一旦某个块未命中，后续所有块都必须新分配
           （因为哈希链断裂，后续块的哈希不可能匹配）

        前缀缓存命中的 token 数记录在 seq.num_cached_tokens 中，
        prefill 阶段只需计算 [num_cached_tokens:] 之后的 token。

        Args:
            seq: 待分配的序列（其 block_table 必须为空，即尚未分配过）。
        """
        assert not seq.block_table  # 确保序列尚未分配过块

        h = -1                # 前一个块的哈希值，初始为 -1（表示没有前驱块）
        cache_miss = False    # 标记是否已经发生缓存未命中

        # 遍历序列的所有逻辑块，逐个尝试缓存匹配或新分配
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)   # 获取第 i 个块对应的 token_ids 切片

            # 只有当块被完全填满时才计算哈希（未填满的最后一个块哈希设为 -1，不参与缓存匹配）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 尝试在前缀缓存索引中查找匹配的块
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 缓存未命中：
                #   - block_id == -1: 索引中没有该哈希
                #   - token_ids 不匹配: 哈希冲突的防御性检查（虽然概率极低但必须校验）
                cache_miss = True

            if cache_miss:
                # 缓存未命中：从空闲队列头部取一个新块来分配
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：复用已有块，将已缓存的 token 数累加
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 该块正被其他序列使用中，增加引用计数即可实现共享
                    # （Copy-on-Write 的简化版本——这里 KV Cache 是只读的，不存在写冲突）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 该块虽然在缓存索引中，但已被释放到空闲队列
                    # （之前使用它的序列都已完成），重新激活分配它
                    block = self._allocate_block(block_id)

            if h != -1:
                # 块已填满：更新其哈希值和 token 内容，注册到前缀缓存索引
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 将块 ID 追加到序列的块表中
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放一个序列占用的所有 KV Cache 块。

        在序列完成（FINISHED）或被抢占（Preempt）时调用。

        逆序遍历块表，逐个减少引用计数：
        - ref_count 降为 0 时，真正释放该块到空闲队列
        - 但其哈希信息仍保留在 hash_to_block_id 索引中，
          以便后续新请求可能的前缀缓存命中——这就是缓存能跨请求生效的关键

        逆序释放的原因：后面的块（序列尾部）最不可能被其他序列共享，
        优先释放它们可以提高空闲块的可用性。

        【学习笔记】被抢占/完成时的完整数据流向：
          CPU 端（本方法）：
            ├── ref_count -= 1（减少引用计数）
            ├── 若 ref_count == 0：block_id 移回 free_block_ids（可被重新分配）
            ├── 保留 hash_to_block_id 索引（前缀缓存的关键！）
            └── 清空 seq.block_table 和 seq.num_cached_tokens
          GPU 端：
            └── 什么都不做！KV 数据原封不动留在显存中
          后续两种可能：
            ├── 新序列前缀匹配 → allocate() 中缓存命中，直接复用 GPU 上的 KV 数据
            └── 新序列不匹配  → _allocate_block() 重新分配该块，GPU 上写入新 KV 数据

        Args:
            seq: 要释放的序列。
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """判断是否有足够的空闲块来追加一个新 token（decode 阶段）。

        在 decode 阶段，每个 step 为每个序列追加一个 token。
        大多数时候，当前块还有空间，不需要新块。
        只有当 len(seq) % block_size == 1 时（刚好跨入新块的第一个 token），
        才需要额外分配一个新的空闲块。

        Args:
            seq: 要追加 token 的序列。

        Returns:
            True 表示可以安全追加。
        """
        # len(seq) % block_size == 1 时需要 1 个新块，否则需要 0 个
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """在 decode 阶段（逐 token 生成）追加一个 token 后，更新块管理状态。

        根据序列当前长度与 block_size 的关系，分三种情况处理：

        1. len(seq) % block_size == 1:
           上一个块刚好填满（在上一步），当前 token 是新块的第一个 token。
           需要分配一个新的空物理块。

        2. len(seq) % block_size == 0:
           当前块刚好被这个 token 填满。
           计算该块的链式哈希并注册到前缀缓存索引中，
           使后续请求可以复用该块。

        3. 其他情况:
           当前块仍有空间，无需任何操作。

        Args:
            seq: 刚追加了一个 token 的序列。
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # 情况 1: 上一个块已满（其 hash 应该已经被设置过），需要新分配一个块
            assert last_block.hash != -1   # 上一个块应该已经被填满并计算过哈希
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # 情况 2: 当前块刚好被填满，计算哈希并注册到前缀缓存
            assert last_block.hash == -1   # 填满之前哈希应该是未设置状态
            token_ids = seq.block(seq.num_blocks-1)   # 获取最后一个（刚填满的）块的 token_ids
            # 取前一个块的哈希作为链式哈希的前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id   # 注册到全局缓存索引

        else:
            # 情况 3: 当前块未满，直接返回（下一个 token 继续填充同一块）
            assert last_block.hash == -1   # 未满的块哈希应为 -1
