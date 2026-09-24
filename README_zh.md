# 海上航线规划系统（Maritime Route Planner）

**基于神经网络地理区域分类的海船航线构建应用程序**


> 英文版说明见 [README.md](README.md)；详细安装步骤见 [docs/INSTALL.md](docs/INSTALL.md)；完整使用手册见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

---

## 一、程序能做什么

给定地球上的起点和终点，程序规划一条**全程位于可航行水域、通行代价最小**的海上航线，并与最短距离航线对比。

1. **区域分类**：神经网络（PyTorch 实现的残差多层感知机）把地理栅格上的每个格子分成四类：

   | 类别 | 含义 | 通行权重 |
   |---|---|---|
   | `OPEN_SEA` | 开阔海，远离海岸的深水区 | 1.0 |
   | `COASTAL_SEA` | 近岸海，靠近海岸的可航行水域 | 1.8 |
   | `NEAR_COAST` | 海岸带，距海岸 5 km 以内的陆地 | 不可通行 |
   | `COASTLINE` | 内陆，远离海洋的陆地 | 不可通行 |

2. **代价图**：按区域权重生成通行代价图；紧挨陆地的水域格子再乘 1.6 的安全惩罚，让航线与海岸保持距离。
3. **路径搜索**：在同一张代价图上提供四种搜索方法，它们都基于同一个离散步代价 `c(u, v) = d_gc(u, v) · (w(u) + w(v)) / 2`：
   * **A\***：精确算法，使用可容许的大圆距离启发函数（默认）；
   * **Dijkstra**：精确算法，无启发函数，作为 A\* 的对照；
   * **动态规划**：精确算法；用价值迭代求解贝尔曼方程 `V(v) = min_u [c(v, u) + V(u)]`，`V(终点) = 0`，再从起点沿价值函数下降得到航线；
   * **遗传算法**：以适应度函数引导、在航点序列上进行进化搜索，结果与最优解对比。
4. **对比与导出**：与大圆航线（球面最短距离，通常会穿过陆地）比较，并导出为 **GeoJSON、CSV、PDF**。每条航线都存入 **SQLite** 数据库。

浏览器页面通过 **WebSocket** 实时显示规划过程：建栅格、逐格分类进度、A\* 展开的节点数，或遗传算法的代数。

## 二、实测结果

以下数字都由 `scripts/` 中的脚本生成，可以复现（见第六节）。

**数据清洗**：152,838 个带标签的点中，有 5,075 个（3.32%）的类别与实测高程/水深相矛盾，例如"近岸海"点位于海平面以上数米。训练前把它们删除（`src/maritime_route/data/cleaning.py`，容差 ±5 m）。

**分类模型**（清洗后语料中的 22,165 个独立测试点）：

| 指标 | 数值 |
|---|---|
| 准确率 | **96.62%** |
| Macro F1 / Cohen κ / Macro ROC-AUC | 0.9660 / 0.9536 / 0.9982 |
| 水域 ↔ 陆地误判（对航行安全最关键的错误） | 74 个（0.33%） |
| 参数量 / 训练 | 98,364 个参数；CPU 上训练 105 轮，约 4.5 分钟 |

十个海域各自的准确率写在 `models/region_metrics.json`，从南海的 92.3% 到红海的 99.9%。

**航线规划**（欧洲水域 6 条航线 + 十个海域各 1 条航线）：

| 结果 | 数值 |
|---|---|
| 穿过陆地的大圆航线 | 16 条中 16 条 |
| 可航行的优化航线 | 16 条中 16 条 |
| A\*、Dijkstra、动态规划 | 16 条航线上最优代价完全相同；A\* 展开的节点平均比 Dijkstra 少 2.43 倍 |
| 遗传算法 | 16 条中 15 条找到可航行航线；平均比最优解高 1.1%（十个海域）和 5.5%（欧洲） |
| 安全航线多走的距离 | 平均 +16%（欧洲）、+22%（十个海域） |

遗传算法在 Rotterdam → Gdańsk 上失败：它收敛到了厄勒海峡，而栅格在那里恰好差一个格子不通，无法跳到 A\* 找到的大贝尔特海峡。这是进化搜索在"迷宫式"地形中的已知弱点，论文中有讨论。

四条预设航线的实测结果（A\*，默认分辨率 0.25°，航速 14 节）：

| 预设航线 | 优化航线 | 大圆距离 | 大圆航线在陆地上的比例 | 规划耗时 |
|---|---|---|---|---|
| Kiel → Tallinn | 1,226 km | 1,052 km | 39% | 约 0.4 s |
| Rotterdam → Lisbon | 2,103 km | 1,809 km | 45% | 约 0.2 s |
| Helsinki → Copenhagen | 1,146 km | 883 km | 33% | 约 2.3 s（自动加密到 0.1375°） |
| Odesa → Novorossiysk | 759 km | 581 km | 58% | 约 0.2 s |

## 三、运行环境

| 项目 | 最低 | 推荐 |
|---|---|---|
| 操作系统 | Windows 10 / Ubuntu 20.04 / macOS 12 | 任意 64 位新版系统 |
| Python | 3.10 | 3.11 |
| 内存 | 4 GB | 8 GB |
| 磁盘空间 | 3 GB（PyTorch 较大） | 5 GB |
| GPU | 不需要 | 不需要 |
| 浏览器 | 支持 WebSocket 即可 | Chrome / Firefox / Edge 新版 |

## 四、安装与启动

```bash
git clone https://github.com/EDXiaoM/MaritimeRouteAI.git
cd MaritimeRouteAI

python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_server.py
```

看到 `Zone classifier ready` 后，用浏览器打开 <http://127.0.0.1:8000>。仓库里已带训练好的模型（`models/`），**不需要先训练**。

如果 `pip` 装不上 PyTorch，单独安装 CPU 版：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

启动参数：

```bash
python scripts/run_server.py --port 8081              # 换端口（8000 被占用时）
python scripts/run_server.py --host 0.0.0.0           # 允许局域网内其他电脑访问
```

## 五、使用方法

页面左侧是控制面板，右侧是地图。右上角两个状态标识：**live connection**（实时通道已连接）和 **model 96.6% acc**（已加载的模型及其准确率）。

| 标签页 | 用途 |
|---|---|
| **Plan** | 设置起终点、选择算法和参数、构建航线、导出结果 |
| **Data** | 上传 AIS 数据（`.geojson` / `.csv` / `.json`），查看分类结果和数据库统计 |
| **Compare** | 优化航线与大圆航线对比；A\*、Dijkstra、遗传算法、动态规划、大圆五种方法对比表 |
| **Model** | 模型结构、测试指标、混淆矩阵、训练曲线 |
| **History** | 数据库中已保存的航线，点击可重新显示 |

1. **设置起终点**：在地图上点两次（先起点 A，后终点 B）、拖动标记、输入坐标，或点预设航线：Kiel → Tallinn、Rotterdam → Lisbon、Helsinki → Copenhagen、Odesa → Novorossiysk。
2. **选择参数**：Search algorithm（A\* / Dijkstra / Genetic algorithm / Dynamic programming / Great circle）、Grid resolution（0.1°–1.0°，默认 0.25°）、Service speed（6–30 节，默认 14 节）。
3. **点 Build route**：进度区依次显示建栅格 → 神经网络逐格分类 → 代价图完成 → 搜索（A\* 显示展开节点数，遗传算法显示代数和当前最优代价，动态规划显示价值迭代的轮数）→ Route ready。找不到航线时会自动加密并扩大栅格重试（最多 3 次）。
4. **看结果**：地图上是按区域着色的优化航线和红色虚线的大圆航线；Route summary 给出长度、大圆距离、绕行系数、代价、航行时间、耗时、节点数。
5. **对比**：Compare 标签页自动更新；点 **Compare algorithms** 在同一张代价图上跑五种方法，并显示遗传算法比最优解高多少、动态规划与 A\* 的代价差（应为 0）。
6. **导出**：GeoJSON / CSV / PDF report。
7. **加载 AIS 数据**：在 Data 标签页拖入文件，可用样例 `data/samples/sample_ais_track.geojson`。

联网时地图用 OpenStreetMap；断网时自动切换到服务器根据模型渲染的离线海图（On-board chart）。

## 六、复现实验结果

数据集 `merged.geojson`（116 MB）超过 GitHub 的文件大小限制，没有放进仓库。把它放到 `data/raw/merged.geojson` 后依次运行：

```bash
python scripts/prepare_dataset.py     # 约 15 秒：清洗、划分、特征        -> data/processed/
python scripts/train_model.py         # CPU 约 5 分钟                     -> models/
python scripts/evaluate_regions.py    # 按海域统计准确率                  -> models/region_metrics.json
python scripts/benchmark_routing.py   # 欧洲 6 条航线                     -> docs/figures/routing_benchmark.json
python scripts/regional_examples.py   # 十个海域 10 条航线 + 地图         -> docs/figures/regional_routes.json
python scripts/make_figures.py        # 论文全部插图                      -> docs/figures/
```

`prepare_dataset.py --no-clean` 可以生成不做清洗的数据集，用于对比。

## 七、运行测试

```bash
pip install -r requirements-dev.txt
pytest
```

应显示 79 个测试全部通过。

## 八、目录结构

```
MaritimeRouteAI/
├── src/maritime_route/            应用程序主包
│   ├── config.py                  路径、区域定义、代价权重、海域范围、全部超参数
│   ├── data/                      数据读取与校验、清洗、特征工程、数据集准备
│   ├── model/                     神经网络结构、训练、推理
│   ├── routing/                   大圆几何、代价图、A*、Dijkstra、遗传算法、动态规划、规划器
│   ├── storage/                   SQLite 表结构与数据访问
│   ├── export/                    GeoJSON / CSV / PDF 导出
│   └── web/                       FastAPI 服务、WebSocket、瓦片、浏览器前端
├── scripts/                       命令行工具（第六节）与 run_server.py
├── tests/                         79 个自动化测试
├── models/                        训练好的权重、标准化器、参考索引、元数据
├── data/samples/                  AIS 样例轨迹
└── docs/                          安装说明、用户手册、插图、图纸、截图
```

每个模块、类和函数都有英文文档字符串，说明用途、参数和返回值；不直观的步骤有行内注释。中文注释是作者的补充说明。

## 九、常见问题

| 现象 | 原因与解决 |
|---|---|
| 地图空白或全黑 | 连不上 OpenStreetMap。在地图左上角图层控件选 *On-board chart (offline)* |
| `Address already in use` | 8000 端口被占用，改用 `--port 8081` |
| 右上角显示 *reconnecting…* | 服务重启过，页面会自动重连 |
| `503 Model weights not found` | `models/` 缺失，重新下载仓库，或按第六节重新训练 |
| 提示 *lies on land* | 港口离最近的可航行格子超过 60 km，通常是因为它在训练数据覆盖的十个海域之外 |
| 遗传算法提示 *no navigable route evolved* | 进化搜索卡在了不通的海峡，请改用 A\* |
| `ModuleNotFoundError: maritime_route` | 没在项目根目录运行或没激活虚拟环境；也可设置 `PYTHONPATH=src` |

## 十、使用限制

- 本程序是决策支持工具，**不是**经过认证的航海系统，不使用官方电子海图；
- 代价模型没有考虑天气、洋流、潮汐、冰情、分道通航制和具体船舶的吃水；
- 语料覆盖十个海域（北欧、黑海和东地中海、红海、波斯湾和阿拉伯海、孟加拉湾、马六甲海峡、南海、日本、南部非洲、中美洲）；在这些海域之外分类不可靠，例如德班外海会被判为陆地；
- 比一个格子还窄的海峡要靠自动加密栅格才能通过，最细到 0.04°。
