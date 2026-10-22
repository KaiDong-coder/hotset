# hotset

[![tests](https://github.com/KaiDong-coder/hotset/actions/workflows/tests.yml/badge.svg)](https://github.com/KaiDong-coder/hotset/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[English](README.md) | **简体中文**

**在一块 48 GB GPU 加 30 GB 内存上服务 gpt-oss-120B（117B 总参数，5.1B 激活）——
decode 30.6 tok/s，prefill 97.8 tok/s，外加一个经实测校准的字节模型，告诉你接下来
那 20 tok/s 需要付出什么代价。**

117B 的模型跑在单张 RTX A6000 上，瓶颈是内存带宽，不是算力：每生成一个 token，
专家权重都要从主机内存经 PCIe 搬进 CPU。hotset 做三件事来对付它——按机器实际能力
切分模型、在调优过的 MXFP4 内核上算 MoE、把"下一个杠杆值多少钱"用真实路由轨迹算出来，
而不是拍脑袋。

## 这台机器，实测

| | | 由什么决定 |
|---|---|---|
| decode，8k 上下文，f16 KV | **30.6 tok/s** | 本机带宽允许上限 40.8 tok/s 的 92% |
| prefill，8k 上下文 | **97.8 tok/s** | CPU 上算 MoE（`--no-op-offload`），9.1 个核满载 |
| 显存 | 47.7 / 49.1 GiB | 36 层 MoE 里有 8 层放在 CPU |
| 主机内存 | 17.3 / 30 GB | 第一约束是 `MemAvailable`，不是显存 |

RTX A6000 48 GB、i9-10900X（10 核 20 线程）、PCIe 3.0 x16。每一行都取自
`llama-server` 自己的 `timings` 块，贪心解码；`./hotset bench` 可复现。

| 其他负载 | decode tok/s | prefill tok/s |
|---|---|---|
| 32k 上下文，q8_0 KV，单流 | 24.9 | 179 |
| 8k 上下文，f16 KV，2 路并发 | 26.6（聚合） | — |

布局和内核同样重要：**同一台机器**换成 `ncmoe=9` 只有 25.4 tok/s decode / 81.1
tok/s prefill——把多一层放到 GPU 上，值 20%。内核快速路径在相同布局下再快 11.6%
（每次 `mul_mat_id` 1.093 vs 1.236 ms，实测）。

## 墙在哪里，路在哪里

```
  decode tok/s
   70 ┤                                      ┌─ 67.6   ncmoe=10 + 28-expert cache   (projected)
   60 ┤                               ┌──────┘
   50 ┤                        ┌──────┘ 50.3          ncmoe=9 + 14-expert cache    (projected)
   40 ┤                 ┌──────┘ 40.8                ceiling: 16.5 GB/s host bandwidth
   30 ┤───────●─────────┘ 30.6                       measured today, ncmoe=8
   20 ┤
      └──────────────────────────────────────────────────────────────────▶
         404.5 MiB/token        246.6                      fewer bytes, not more FLOPs
```

decode 每 token 要读 `ncmoe × 50.56 MiB`。按本机实测的 16.5 GB/s，这直接把 decode
钉在 40.8 tok/s 上限，而引擎已经跑出 12.4 GB/s——上限的 92%。再优化内核也不会改变
这个数，只有少读字节才行。

路由恰好足够偏斜：每层 128 个专家里，最热的 14 个吃掉了 54–68% 的路由槽位（均匀
分布只有 10.9%）。所以把这一小撮专家常驻显存，就能删掉大部分读取。

`scripts/batch_union.py` 回放一段真实的 897-token 路由轨迹来定价这个缓存；它 `b=1`
那一行精确复现了实测的 30.6 tok/s——正因如此，其余投影行才值得发出来。完整推导在
[`docs/PERF-LEVERS.md`](docs/PERF-LEVERS.md)，包括为什么 batching 是缓存
**替代品**而不是补充。

## 仓库里有什么

| 组件 | 作用 |
|---|---|
| `hotset` | 入口：`serve`、`fit`、`bench`、`test`、`queue` |
| `scripts/moe_fit.py` | 直接读 GGUF 张量表做精确切分；会拒绝必然抖动的配置 |
| `scripts/serve.sh` | 即 `moe_fit --run`：先打印选定的配置，再 exec `llama-server` |
| `scripts/bench.sh` | 一条命令，在你自己的机器上复现上面的表 |
| `scripts/routing_skew.py` | 回放路由轨迹：每层专家偏斜、LRU 命中率 |
| `scripts/batch_union.py` | 用真实轨迹给专家缓存和 batching 定价（字节/token → tok/s） |
| `scripts/gpu_queue.py` | 在**共享** GPU 上等出显存预算再跑一条命令，绝不给别人的进程发信号 |
| `scripts/bench_queue.py` | 排队等显存的基准任务，串行执行 |
| `scripts/membw.c` | 测出 16.5 GB/s 那个上限的主机流式读探针 |
| `tests/run_tests.py` | 布局策略单元测试 + 内核 A/B 等价性套件 |
| [`docs/PROBE-REPORT.md`](docs/PROBE-REPORT.md) | 全部测量、A/B 实验与失败模式 |
| [`docs/PERF-LEVERS.md`](docs/PERF-LEVERS.md) | 与实测 tok/s 校准过的字节模型 |
| [`docs/SPLIT-MOE.md`](docs/SPLIT-MOE.md) | 专家缓存的设计（v0.2，进行中） |

## 快速开始

前置条件：NVIDIA 驱动 + CUDA toolkit、`git`、`cmake ≥ 3.20`、`python3`；
约 80 GB 空闲磁盘（构建树加 63.4 GB 的 GGUF）；一块 48 GB 显存的 GPU，以及
**约 19 GB 的 `MemAvailable`** 用来放常驻 CPU 的专家层。实测环境是 Ubuntu 26.04 +
CUDA 13.1。

```bash
git clone https://github.com/KaiDong-coder/hotset && cd hotset

# 1. 构建。llama.cpp 的 CMake 树就在仓库根目录。
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli llama-bench -j "$(nproc)"

# 2. 下载模型（约 59 GiB，可断点续传）。需要 `pip install --user requests`。
python3 scripts/download_modelscope.py

# 3. 这台机器该用什么配置——打印出来，不是猜：
./hotset fit --ctx 8192

# 4. 起服务（先打印布局，再启动 llama-server）：
./hotset serve --ctx 8192
```

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "n_predict": 32, "temperature": 0}' \
  | python3 -m json.tool
```

### 复现那张表

```bash
./hotset bench                 # 每个 上下文/KV/并发 组合一行
REPS=5 ./hotset bench          # 多采几次；结果在 results/bench-*.json
```

### 证据

这一页上的每个数字背后都有一个文件——`bench_serve.py` 的原始输出，含服务端的
完整命令、每次运行的 `timings`、峰值显存、峰值 RSS 以及每个请求的磁盘读取量。
每条结论对应哪个文件，见 [`results/README.md`](results/README.md)。记录里的绝对
路径已规范成仓库相对路径；**数字没有动过**。

### 测试

```bash
./hotset test                  # 单元 + 内核 A/B（不需要模型，不需要 GPU）
./hotset test --e2e            # 另加 perplexity 与 needle 测试，需要 GGUF
CMAKE=/path/to/cmake ./hotset test   # cmake 不在 PATH 上时
```

内核阶段会构建 `test-mmid-cpu`，分别在快速路径开、关两种情况下运行，再逐元素比较。
这个比较是刻意设计的：内核内部的自检用的是同一组指针做参考，2026-10-21 那个
`ry[s]` 激活行索引 bug 正是因此躲过了检查——只有拿快速路径去对通用路径才会暴露。

### 共享一张不属于你的 GPU

这里的好几次测量都被一个无关占用者挡住（它占着 41.9 / 49.1 GiB）。hotset 从不杀掉
或降低别人进程的优先级——它等待，并且跑不了时留下证据：

```bash
python3 -I scripts/gpu_queue.py --need-mib 47700 -- <一条命令>
python3 -I scripts/bench_queue.py --list     # 谁在等显存
```

## 这个项目适合你吗

- **适合**：你有一块 48 GB 的 GPU，想在自己已有的机器上服务 120B 级模型，并且想
  知道它的速度到底从哪来。
- **适合**：你做 MoE 推理，想要一个经过校准的字节模型和一套轨迹回放工具，能直接
  对着自己的硬件跑——这个仓库里大部分东西就是这些。
- **暂不适合**：你需要很多并发用户。2 路并发的聚合吞吐是 26.6 tok/s，只比单流高
  ~8%。连续批处理目前只是 [`docs/PERF-LEVERS.md`](docs/PERF-LEVERS.md) 里的投影，
  不是已上线的功能。

## 状态

- **v0.1，已实测**：CPU 常驻的 MoE 布局、MXFP4 `mul_mat_id` 快速路径（decode 与
  prefill）、自动布局规划器、路由轨迹工具、测试套件。
- **进行中**：split-MoE 的 GPU 专家缓存（`--moe-cache N`）。设计见
  [`docs/SPLIT-MOE.md`](docs/SPLIT-MOE.md)（CPU 与 CUDA 内核都要跳过被屏蔽的 id，
  一次 `mul_mat_id` 拆成两次）。上面 50.3 / 67.6 tok/s 两行是**基于真实轨迹的字节
  模型投影**，不是实测——在真测出来之前，这份 README 不会把它们写成 measured。
- **做不到**：不减少读取字节的前提下，单流 decode 无法超过 40.8 tok/s（PCIe 3.0
  的 16.5 GB/s 就是墙）。

v0.2 需要什么、哪里最缺贡献，见 [`ROADMAP.md`](ROADMAP.md)。

## 诚实的局限

- **PCIe 3.0 是一个前提假设。** 在 Gen 3 下，把 MoE 放在 CPU 上算是把专家流式搬到
  GPU 上的约 3.3 倍（相同布局 25.4 vs 7.7 tok/s），所以规划器会传
  `--no-op-offload`。`moe_fit.py` 在 Gen 3 以上会把 `op_offload` 翻成 `auto`，
  但那条分支没有实测过。
- **第一约束是 30 GB 内存，不是显存。** `ncmoe=12`（19.4 GiB 的 CPU 专家）会触发
  专家页换出：三个请求读了 122 GB 磁盘，decode 塌到 21.8 tok/s。规划器按
  `MemAvailable` 决定 CPU 层数，内存不够时会报告缺口，而不是装作没事。
- **默认 f16 KV，q8_0 是妥协。** 32k 上下文配 `ncmoe=8` 只有 q8_0 塞得下，而且它
  略快一点，但它在长上下文 needle 测试里失败，而 f16 通过。用它来"塞得下"，
  别用它来"更快"。
- **这是一台机器的数字。** GPU 是共享的，连续两次运行会有 ~7% 的漂移；要比配置就
  在同一次运行内比，别跨天比。你的主机带宽决定你的天花板——相信任何 tok/s 数字之前，
  先跑一下 `scripts/membw.c`。
- **这是上游树的 fork。** 仓库里带的是改过内核的 llama.cpp 检出，不是上游的直接
  替代品。见 [`NOTICE.md`](NOTICE.md)。

## 参与

对这个仓库从未见过的硬件做一次测量，是你能带来的最有价值的东西——见
[`CONTRIBUTING.md`](CONTRIBUTING.md) 与
[issue #2](https://github.com/KaiDong-coder/hotset/issues/2)。问题与还不成形的
结果都欢迎发到 [Discussions](https://github.com/KaiDong-coder/hotset/discussions)。

## 名字

*hot set* 指一个大工作集里那部分小而高频的子集——在这里，就是每层 128 个专家中吃掉
54–68% 路由槽位的那 ~14 个。hotset 是一个引擎：找出这个集合，把它放在读起来便宜的
地方，其余的从便宜不了的地方取。

## 许可

MIT，与 llama.cpp 一致。见 [`LICENSE`](LICENSE)（上游）与
[`NOTICE.md`](NOTICE.md)（改了什么）。
