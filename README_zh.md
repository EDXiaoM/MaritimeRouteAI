# 海上航线规划系统（Maritime Route Planner）— 毕业设计

**俄文版 — [README_ru.md](README_ru.md)** · English — [README.md](README.md)

Li Haoran · 布列斯特国立技术大学 智能信息技术教研室 · 指导老师 А.А. Козинский · 2026/27 学年

**毕业设计**：*基于神经网络地理区域分类的海船航线构建应用程序。*

> 作者的**课程设计**（Sea Course GA，遗传算法）在另一个仓库：<https://github.com/EDXiaoM/SeaCourseGA>。

## 毕业设计材料

| 材料 | 位置 |
|---|---|
| 说明书**第一稿**（Word 和 PDF） | [`docs/diploma/`](docs/diploma) |
| 任务书（俄文原件） | [`docs/diploma/Assignment_diploma_ru.pdf`](docs/diploma/Assignment_diploma_ru.pdf) |
| 图纸（5 张 A1，PDF 和 PNG） | [`docs/posters/`](docs/posters) |
| **用户手册**（安装与使用，PDF） | [`docs/User_Guide_MaritimeRouteAI.pdf`](docs/User_Guide_MaritimeRouteAI.pdf) |
| 用户手册与安装说明（Markdown） | [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)、[`docs/INSTALL.md`](docs/INSTALL.md) |
| 程序代码 | [`src/`](src)、[`scripts/`](scripts)、[`tests/`](tests) |
| 训练好的模型 | [`models/`](models) |
| 论文插图和截图 | [`docs/figures/`](docs/figures)、[`docs/screenshots/`](docs/screenshots) |
| 带标签数据集（第二版，156,838 个点，压缩包 7 MB） | [Release v1.0](https://github.com/EDXiaoM/MaritimeRouteAI/releases/tag/v1.0) 中的 `merged_geojson_v2.zip` |

详细安装步骤见 [docs/INSTALL.md](docs/INSTALL.md)，完整使用手册见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

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

**语料**：使用指导老师提供的第二版带标签语料，共 156,838 个点、覆盖十一个海域；它在第一版基础上新增了爱琴海的 4,000 个点（每类 1,000 个）。

**数据清洗**：156,838 个点中，有 5,427 个（3.46%）的类别与实测高程/水深相矛盾，例如"近岸海"点位于海平面以上数米。每个点的两个测量值（`noaa_depth` 和 `elevation`）分别检查，容差 ±5 m，任一矛盾即在训练前删除（`src/maritime_route/data/cleaning.py`）。清洗后剩 151,411 个点，与指导老师给出的数字完全一致。

**分类模型**（清洗后语料中的 22,712 个独立测试点）：

| 指标 | 数值 |
|---|---|
| 准确率 | **96.95%** |
| Macro F1 / Cohen κ / Macro ROC-AUC | 0.9695 / 0.9582 / 0.9985 |
| 水域 ↔ 陆地误判（对航行安全最关键的错误） | 67 个（0.29%） |
| 参数量 / 训练 | 98,364 个参数；CPU 上训练 97 轮，约 4.3 分钟 |

十一个海域各自的准确率写在 `models/region_metrics.json`，从南海的 93.5% 到红海的 99.5%；新增的爱琴海为 97.9%。

**航线规划**（欧洲水域 6 条航线 + 十一个海域各 1 条航线）：

| 结果 | 数值 |
|---|---|
| 穿过陆地的大圆航线 | 17 条中 17 条 |
| 可航行的优化航线（按分类器判断） | 17 条中 17 条 |
| A\*、Dijkstra、动态规划 | 17 条航线上最优代价完全相同；A\* 展开的节点平均比 Dijkstra 少 2.5 倍 |
| 遗传算法 | 17 条中 15 条找到可航行航线；平均比最优解高 1.1%（十一个海域）和 3.8%（欧洲） |
| 安全航线多走的距离 | 平均 +17%（欧洲）、+28%（十一个海域） |

**100 条随机航线测试**（`scripts/route_audit.py`；每条航线每公里用独立的 1 km 陆地掩膜 GLOBE / global-land-mask 检查一次）：

| 结果 | 航线数 |
|---|---|
| 航线正确 | 66 |
| 擦到海岸（陆地段短于 3 km） | 5 |
| 穿过陆地 | 19（16 条是分类器把样本稀少的陆地当成了水，3 条是网格斜向一步跨过窄陆地） |
| 没找到航线 | 10（7 条没有开放的海上通道：加勒比海 ↔ 太平洋、博斯普鲁斯海峡；3 条通道在分类网格上被堵住） |

所有航线记录在 `docs/figures/route_audit.csv`，地图在论文附录 C。错误的位置说明了语料在哪些海岸需要补充样本。

**自动海岸线检查（1.2 版）。** 测试表明分类器有时会把样本稀少的陆地当成水，所以现在规划器对*每一条*航线都做同样的检查（`src/maritime_route/routing/coast_check.py`）：航线每公里用独立的陆地掩膜检查一次；如果连续穿越陆地 3 km 以上，就把相应的网格格子封锁后重新搜索（必要时换更细的网格）；仍然穿陆的航线会显示红色警告。检查结果显示在航线摘要上方的横幅里，也写进 PDF 报告、GeoJSON 导出和数据库。开启检查后重做 100 条航线测试：

| 结果 | 无检查 | 有检查 |
|---|---|---|
| 航线正确 | 66 | 75 |
| 擦到海岸 | 5 | 15 |
| 穿过陆地 | 19 | 0 |
| 没找到航线 | 10 | 10 |

19 条错误航线中有 19 条被自动修正；原本正确的航线保持不变，平均请求时间只增加约 1%。界面的 5 条预设航线中有 4 条在没有检查时也穿过陆地（Helsinki → Copenhagen 穿陆 108 km），现在全部通过检查。检查需要 `global-land-mask` 包（已写入 `requirements.txt`）；没有它时规划器照旧工作。

遗传算法在两条航线上失败：Rotterdam → Gdańsk 收敛到了厄勒海峡（栅格在那里恰好差一个格子不通），无法跳到 A\* 找到的大贝尔特海峡；Piraeus → Thessaloniki 在爱琴海岛群中始终没有找到不穿越陆地的通道。这是进化搜索在"迷宫式"地形中的已知弱点，论文中有讨论。

爱琴海航线使用 0.1° 网格（点预设按钮会自动选择）：默认 0.25° 时，网格的对角步会直接穿过比一个格子还窄的阿提卡半岛。另外，语料在爱琴海部分岛屿（如 Kea、Naxos、Skiathos）上没有任何陆地样本，因此模型会把它们判为海域。

五条预设航线的实测结果（A\*，默认分辨率 0.25°，爱琴海为 0.1°，航速 14 节）：

| 预设航线 | 优化航线 | 大圆距离 | 大圆航线在陆地上的比例 | 规划耗时 |
|---|---|---|---|---|
| Kiel → Tallinn | 1,249 km | 1,052 km | 36% | 约 3 s（自动加密到 0.1375°） |
| Rotterdam → Lisbon | 2,103 km | 1,809 km | 45% | 约 0.3 s |
| Helsinki → Copenhagen | 1,152 km | 883 km | 33% | 约 0.3 s |
| Odesa → Novorossiysk | 759 km | 581 km | 62% | 约 0.2 s |
| Piraeus → Thessaloniki（爱琴海，0.1°） | 532 km | 299 km | 33% | 约 0.3 s |

## 1.3 版 —— 按导师意见修改

* **参数在程序界面中设置，而不是写在代码里。** 面板 *算法参数*（*Algorithm parameters*）：A\* / Dijkstra 最大步数、
  A\* 启发权重、动态规划最大迭代次数、遗传算法的种群规模、代数（epoch）、早停、锦标赛、交叉/变异概率、
  精英数、航点数、陆地惩罚和随机种子，以及网格参数和**时间限制**。
* **计时器和“停止”按钮**：搜索时显示已用时间，**■ Stop** 可在 0.22 秒内中止任何搜索。
  结果状态：找到 / 无路径 / 达到步数上限 / 超时 / 已停止。
* **每次搜索结果（成功或失败）都保存为 JSON 文件**，位于 `outputs/results`；文件名包含点坐标、算法、
  结果和搜索时间；文件中有请求、全部参数、时间和**所有点的坐标**；屏幕上显示文件的完整路径。
* **自定义航线和途经点**：在地图任意位置点击设置 A、B，可拖动或手动输入；**Shift + 点击**添加途经点。
* **复杂航线**：自适应搜索区域、陆地掩膜融合进代价地图、走廊两级搜索。
* **同一航线上比较各算法**（带进度和停止），地图上画出所有算法的航线，并给出**理论复杂度**与实测工作量。
* **界面支持英文、俄文和中文**：顶栏的 **EN / RU / 中文** 按钮切换语言，选择保存在浏览器中，屏幕上的结果
  会立即用新语言重绘（`web/static/js/i18n.js`）。结果文件和报告保持英文。
* 实验脚本 `scripts/complex_routes.py`；Dijkstra: 1,000 steps → step limit, 10,000 → route (6,728 steps needed); A* needs 5,351。

| 复杂航线 | 1.2 版 | 1.3 版 |
|---|---|---|
| Gdańsk → Bergen | 擦岸, 5.7 s | 擦岸, 1.5 s |
| Helsinki → Copenhagen | 擦岸, 0.7 s | 正确, 0.6 s |
| Rotterdam → Gdańsk | 正确, 3.9 s | 擦岸, 3.9 s |
| Piraeus → Rhodes | 擦岸, 1.0 s | 擦岸, 1.1 s |
| Jeddah → Dubai | 无航线, 126.4 s | 正确, 5.5 s |
| Singapore → Hong Kong | 正确, 0.7 s | 正确, 0.6 s |
| Kapellskär → Oulu | 穿陆 10 km, 11.1 s | 穿陆 5 km, 20.3 s |

用 1.3 版重新规划 100 条随机航线：正确 79，擦岸 11，穿陆 2，无航线 8。

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

页面左侧是控制面板，右侧是地图。右上角两个状态标识：**live connection**（实时通道已连接）和 **model 96.9% acc**（已加载的模型及其准确率）。

| 标签页 | 用途 |
|---|---|
| **Plan** | 设置起终点、选择算法和参数、构建航线、导出结果 |
| **Data** | 上传 AIS 数据（`.geojson` / `.csv` / `.json`），查看分类结果和数据库统计 |
| **Compare** | 优化航线与大圆航线对比；A\*、Dijkstra、遗传算法、动态规划、大圆五种方法对比表 |
| **Model** | 模型结构、测试指标、混淆矩阵、训练曲线 |
| **History** | 数据库中已保存的航线，点击可重新显示 |

1. **设置起终点**：在地图上点两次（先起点 A，后终点 B）、拖动标记、输入坐标，或点预设航线：Kiel → Tallinn、Rotterdam → Lisbon、Helsinki → Copenhagen、Odesa → Novorossiysk、Piraeus → Thessaloniki。
2. **选择参数**：Search algorithm（A\* / Dijkstra / Genetic algorithm / Dynamic programming / Great circle）、Grid resolution（0.1°–1.0°，默认 0.25°）、Service speed（6–30 节，默认 14 节）。
3. **点 Build route**：进度区依次显示建栅格 → 神经网络逐格分类 → 代价图完成 → 搜索（A\* 显示展开节点数，遗传算法显示代数和当前最优代价，动态规划显示价值迭代的轮数）→ Route ready。找不到航线时会自动加密并扩大栅格重试（最多 3 次）。
4. **看结果**：地图上是按区域着色的优化航线和红色虚线的大圆航线；Route summary 给出长度、大圆距离、绕行系数、代价、航行时间、耗时、节点数。
5. **对比**：Compare 标签页自动更新；点 **Compare algorithms** 在同一张代价图上跑五种方法，并显示遗传算法比最优解高多少、动态规划与 A\* 的代价差（应为 0）。
6. **导出**：GeoJSON / CSV / PDF report。
7. **加载 AIS 数据**：在 Data 标签页拖入文件，可用样例 `data/samples/sample_ais_track.geojson`。

联网时地图用 OpenStreetMap；断网时自动切换到服务器根据模型渲染的离线海图（On-board chart）。

## 六、复现实验结果

数据集 `merged.geojson`（119 MB，含爱琴海的第二版）太大，不能直接放进仓库，已作为附件 `merged_geojson_v2.zip`（7 MB）放在 [Release v1.0](https://github.com/EDXiaoM/MaritimeRouteAI/releases/tag/v1.0)。下载解压到 `data/raw/merged.geojson` 后依次运行：

```bash
python scripts/prepare_dataset.py     # 约 16 秒：清洗、划分、特征        -> data/processed/
python scripts/train_model.py         # CPU 约 5 分钟                     -> models/
python scripts/evaluate_regions.py    # 按海域统计准确率                  -> models/region_metrics.json
python scripts/benchmark_routing.py   # 欧洲 6 条航线                     -> docs/figures/routing_benchmark.json
python scripts/regional_examples.py   # 十一个海域 11 条航线 + 地图       -> docs/figures/regional_routes.json
python scripts/make_figures.py        # 论文全部插图                      -> docs/figures/
python scripts/route_audit.py         # 100 条随机航线测试                -> docs/figures/route_audit.*（需要 global-land-mask）
```

`prepare_dataset.py --no-clean` 可以生成不做清洗的数据集，用于对比。

## 七、运行测试

```bash
pip install -r requirements-dev.txt
pytest
```

应显示 139 个测试全部通过。

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
├── tests/                         139 个自动化测试
├── models/                        训练好的权重、标准化器、参考索引、元数据
├── data/samples/                  AIS 样例轨迹
└── docs/
    ├── diploma/                   毕业设计说明书第一稿（Word、PDF）和任务书
    ├── posters/                   毕业设计图纸（5 张 A1）
    ├── User_Guide_MaritimeRouteAI.pdf   用户手册（安装与使用）
    └── INSTALL.md、USER_GUIDE.md、figures/、screenshots/
```

每个模块、类和函数都有英文文档字符串，说明用途、参数和返回值；不直观的步骤有行内注释。中文注释是作者的补充说明。

## 九、常见问题

| 现象 | 原因与解决 |
|---|---|
| 地图空白或全黑 | 连不上 OpenStreetMap。在地图左上角图层控件选 *On-board chart (offline)* |
| `Address already in use` | 8000 端口被占用，改用 `--port 8081` |
| 右上角显示 *reconnecting…* | 服务重启过，页面会自动重连 |
| `503 Model weights not found` | `models/` 缺失，重新下载仓库，或按第六节重新训练 |
| 提示 *lies on land* | 港口离最近的可航行格子超过 60 km，通常是因为它在训练数据覆盖的十一个海域之外 |
| 遗传算法提示 *no navigable route evolved* | 进化搜索卡在了不通的海峡，请改用 A\* |
| `ModuleNotFoundError: maritime_route` | 没在项目根目录运行或没激活虚拟环境；也可设置 `PYTHONPATH=src` |

## 十、使用限制

- 本程序是决策支持工具，**不是**经过认证的航海系统，不使用官方电子海图；
- 代价模型没有考虑天气、洋流、潮汐、冰情、分道通航制和具体船舶的吃水；
- 语料覆盖十一个海域（北欧、爱琴海、黑海和东地中海、红海、波斯湾和阿拉伯海、孟加拉湾、马六甲海峡、南海、日本、南部非洲、中美洲）；在这些海域之外分类不可靠，例如德班外海会被判为陆地；
- 比一个格子还窄的海峡要靠自动加密栅格才能通过，最细到 0.04°。
