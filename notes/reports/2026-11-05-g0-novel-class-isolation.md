# G0 数据核查：novel 类隔离判定（iSAID 源域 vs SCORE 四测试集）

日期：2026-11-05。纯 CPU 只读数据核查 + 人工语义映射；未改代码、未加依赖、未读
像素、未运行模型。前置阅读：`reports/2026-10-28-data-runtime-readiness.md` §2/§3、
`notes/decisions-2026-10.md` 第 8 条（“新类别”指相对本项目适配训练标签未见，
不声称未出现在 SAM3 预训练中）。

机器可读明细（逐类判定、映射表、统计）：
`sam3_ovis/logs/g0_novel_class_isolation.json`。

## 0. 结论速览

| 测试集 | 类数 | seen | novel-严格 | novel-仅父类 | novel-上位概念 | 严格 novel 类占比 | 严格 novel 实例占比 |
|---|---|---|---|---|---|---|---|
| NWPU | 10 | 9 | **0** | 0 | 1（vehicle） | **0%** | **0%** |
| SOTA | 18 | 15 | **3** | 0 | 0 | 16.7% | **0.12%**（34/29,247） |
| FAST | 37 | 7 | **0** | 29 | 1（Intersection） | **0%** | **0%** |
| SIOR-val | 20 | 9 | **9** | 1（overpass） | 1（vehicle） | 45% | 7.0% |

- **NWPU 与 FAST 不含任何严格 novel 类**：NWPU 是“9 个直接重合 + vehicle 上位概
  念”；FAST 78.4% 的类是源域大类（plane/ship/large-vehicle）的下位子类。
- 成规模的严格 novel 只在 **SIOR-val**（9 类、8,759 实例）；SOTA 的 3 个严格
  novel 类合计仅 34 实例，只能作定性证据。
- FAST 的正确用途是**层级迁移 / 细粒度下位泛化**测试集（61.2% 实例为下位类），
  不能对外称 novel 类测试。

## 1. 读取方式与数据规模

- 注册表：`sam3_ovis/data/score_datasets.py` 的 `DATASETS`；数据根
  `$SAM3_OVIS_DATA_ROOT`（默认 `/data/wangcheng/dataset/SCORE`）。
- 源域（适配训练）：`iSAID/train/instances_train.json`，28,029 图 /
  704,684 实例 / 15 类（人工实例标注）。
- 测试集：NWPU/SOTA/FAST/SIOR 各自 val json 的 `categories`
  （`NWPU/val`、`SOTA/val`、`FAST/val`、`SIOR/val`）。
- 图/实例/类别数与注册表 `expected_*` 全部一致（NWPU 731/2,661，SOTA
  874/29,247，FAST 3,207/43,258，SIOR-val 11,738/124,373）。
- 说明：SIOR 在注册表中 role=train（SCORE 交叉设定是“iSAID 训练 → SIOR-val
  测试”），本文按任务要求取其 val json 作测试侧词表。iSAID-val 与源域同为
  15 类，不参与 novel 判定。

## 2. 归一化规则与人工映射表（登记，来源=人工判断）

归一化：`lower → strip → [-_\s]+ 折叠为单空格`。在此之上登记以下人工映射；
同义=合并后计 seen；父子/逆父子=层级迁移，不计严格 novel；近义不合并=保持
novel 但附警告。

同义（4 条，合并计 seen）：

| 测试侧 | 源域 | 理由 |
|---|---|---|
| airplane | plane | 遥感通用同义词：NWPU/DIOR 用 airplane，iSAID/SOTA 用 plane |
| baseball field | baseball_diamond | 同一概念（棒球场），DIOR 与 DOTA/iSAID 命名不同 |
| football field | Soccer_ball_field | FAIR1M 的 Football-Field（足球场）= iSAID 足球场 |
| small car | Small_Vehicle | 遥感惯例 car 即 small vehicle |

父子（31 条，计层级迁移）：

| 方向 | 测试类 | 源域父/子类 | 理由 |
|---|---|---|---|
| 正向（子类测试，父类在源域） | overpass | Bridge | 立交桥是桥的子类 |
| 正向 | A220/A321/A330/A350/ARJ21/Boeing737/747/777/787/C919（11 类） | plane | 具体机型是 plane 的下位类 |
| 正向 | Dry-Cargo-Ship/Engineering-Ship/Fishing-Boat/Liquid-Cargo-Ship/Motorboat/Passenger-Ship/Tugboat/Warship（8 类） | ship | 船型子类；boat≈ship 同义用于子类名解析 |
| 正向 | Bus/Cargo-Truck/Dump-Truck/Van/Tractor/Trailer/Truck-Tractor/Excavator（8 类） | Large_Vehicle | 车型子类；Excavator 为工程车辆，保守按层级迁移 |
| 正向（残余桶） | other-airplane / other-ship / other-vehicle | plane / ship / small+large vehicle | “其他”桶等价于源域大类概念本身 |
| 逆向（上位概念测试，源域只有子类） | vehicle（NWPU、SIOR） | Small_Vehicle + Large_Vehicle | vehicle 是二者的父类 |
| 逆向 | Intersection（FAST） | Roundabout | 环岛是路口的子类，源域只有 roundabout |

近义不合并（6 条，保持 novel、附警告）：

| 测试类 | 警告 |
|---|---|
| golffield | DIOR 的 golf field=高尔夫球场，**非**足球场；字面与 soccer/football field 相近，prompt 建议写 golf course 并人工抽检 |
| stadium | 场馆包含源域 sports field 类，语义相邻但不构成父子；弱证据，用作 novel 前需人工抽检 |
| dam | 条状跨水构造与 bridge 外观相邻，非桥 |
| airport | 图像中共现源域 plane，但类本身是基础设施 |
| helipad | 与 helicopter 名称相邻，但为基础设施而非飞行器子类 |
| container crane | 港口固定机械，非 harbor 子类亦非 vehicle 子类；实例仅 11 个 |

以上均为人工判断，未做像素核验；训练/评测 prompt 定稿后须按同表复核。

## 3. 逐测试集判定表

判定取值：seen / novel-严格 / novel-仅父类（父类在源域、子类不在）/
novel-上位概念（逆父子：源域只有其子类）。

### NWPU（731 图 / 2,661 实例 / 10 类）

| 测试类 | 实例 | 关系 | 命中源域 | 判定 |
|---|---|---|---|---|
| airplane | 395 | 同义 | plane | seen |
| ship | 176 | 归一化同名 | ship | seen |
| storage_tank | 405 | 同名 | storage_tank | seen |
| baseball_diamond | 306 | 同名 | baseball_diamond | seen |
| tennis_court | 384 | 同名 | tennis_court | seen |
| basketball_court | 104 | 同名 | basketball_court | seen |
| ground_track_field | 133 | 同名 | Ground_Track_Field | seen |
| harbor | 241 | 同名 | Harbor | seen |
| bridge | 85 | 同名 | Bridge | seen |
| vehicle | 432 | 上位概念 | Small/Large_Vehicle | novel-上位概念 |

类占比：seen 90%，novel-上位 10%；实例占比：seen 83.8%，novel-上位 16.2%。

### SOTA（874 图 / 29,247 实例 / 18 类）

| 测试类 | 实例 | 关系 | 命中源域 | 判定 |
|---|---|---|---|---|
| large-vehicle | 2,493 | 同名 | Large_Vehicle | seen |
| swimming-pool | 277 | 同名 | Swimming_pool | seen |
| helicopter | 112 | 同名 | Helicopter | seen |
| bridge | 216 | 同名 | Bridge | seen |
| plane | 937 | 同名 | plane | seen |
| ship | 5,571 | 同名 | ship | seen |
| soccer-ball-field | 44 | 同名 | Soccer_ball_field | seen |
| basketball-court | 72 | 同名 | basketball_court | seen |
| ground-track-field | 47 | 同名 | Ground_Track_Field | seen |
| small-vehicle | 17,162 | 同名 | Small_Vehicle | seen |
| baseball-diamond | 84 | 同名 | baseball_diamond | seen |
| tennis-court | 330 | 同名 | tennis_court | seen |
| roundabout | 67 | 同名 | Roundabout | seen |
| storage-tank | 973 | 同名 | storage_tank | seen |
| harbor | 828 | 同名 | Harbor | seen |
| container-crane | 11 | 近义不合并 | — | novel-严格 |
| airport | 18 | 近义不合并 | — | novel-严格 |
| helipad | 5 | 近义不合并 | — | novel-严格 |

类占比：seen 83.3%，novel-严格 16.7%；实例占比：seen 99.88%，novel-严格 0.12%。

### FAST（3,207 图 / 43,258 实例 / 37 类）

| 测试类 | 实例 | 关系 | 命中/父类 | 判定 |
|---|---|---|---|---|
| Baseball-Field | 104 | 同义 | baseball_diamond | seen |
| Basketball-Court | 103 | 同名 | basketball_court | seen |
| Bridge | 105 | 同名 | Bridge | seen |
| Football-Field | 103 | 同义 | Soccer_ball_field | seen |
| Roundabout | 52 | 同名 | Roundabout | seen |
| Small-Car | 15,279 | 同义 | Small_Vehicle | seen |
| Tennis-Court | 363 | 同名 | tennis_court | seen |
| A220 | 583 | 父子 | plane | novel-仅父类 |
| A321 | 271 | 父子 | plane | novel-仅父类 |
| A330 | 145 | 父子 | plane | novel-仅父类 |
| A350 | 112 | 父子 | plane | novel-仅父类 |
| ARJ21 | 29 | 父子 | plane | novel-仅父类 |
| Boeing737 | 384 | 父子 | plane | novel-仅父类 |
| Boeing747 | 166 | 父子 | plane | novel-仅父类 |
| Boeing777 | 135 | 父子 | plane | novel-仅父类 |
| Boeing787 | 168 | 父子 | plane | novel-仅父类 |
| C919 | 15 | 父子 | plane | novel-仅父类 |
| other-airplane | 1,175 | 父子（残余桶） | plane | novel-仅父类 |
| Dry-Cargo-Ship | 1,186 | 父子 | ship | novel-仅父类 |
| Engineering-Ship | 270 | 父子 | ship | novel-仅父类 |
| Fishing-Boat | 640 | 父子 | ship | novel-仅父类 |
| Liquid-Cargo-Ship | 269 | 父子 | ship | novel-仅父类 |
| Motorboat | 1,010 | 父子 | ship | novel-仅父类 |
| Passenger-Ship | 190 | 父子 | ship | novel-仅父类 |
| Tugboat | 115 | 父子 | ship | novel-仅父类 |
| Warship | 79 | 父子 | ship | novel-仅父类 |
| other-ship | 357 | 父子（残余桶） | ship | novel-仅父类 |
| Bus | 110 | 父子 | Large_Vehicle | novel-仅父类 |
| Cargo-Truck | 1,282 | 父子 | Large_Vehicle | novel-仅父类 |
| Dump-Truck | 2,592 | 父子 | Large_Vehicle | novel-仅父类 |
| Excavator | 110 | 父子（保守） | Large_Vehicle | novel-仅父类 |
| Tractor | 8 | 父子 | Large_Vehicle | novel-仅父类 |
| Trailer | 86 | 父子 | Large_Vehicle | novel-仅父类 |
| Truck-Tractor | 25 | 父子 | Large_Vehicle | novel-仅父类 |
| Van | 14,463 | 父子 | Large_Vehicle | novel-仅父类 |
| other-vehicle | 496 | 父子（残余桶） | Small/Large_Vehicle | novel-仅父类 |
| Intersection | 678 | 上位概念 | Roundabout | novel-上位概念 |

类占比：seen 18.9%，novel-仅父类 78.4%，novel-上位 2.7%，**novel-严格 0%**；
实例占比：seen 37.2%，novel-仅父类 61.2%，novel-上位 1.6%。

### SIOR-val（11,738 图 / 124,373 实例 / 20 类）

| 测试类 | 实例 | 关系 | 命中源域 | 判定 |
|---|---|---|---|---|
| airplane | 8,212 | 同义 | plane | seen |
| baseball field | 3,434 | 同义 | baseball_diamond | seen |
| basketball court | 2,146 | 同名 | basketball_court | seen |
| bridge | 2,587 | 同名 | Bridge | seen |
| ground track field | 1,885 | 同名 | Ground_Track_Field | seen |
| harbor | 3,105 | 同名 | Harbor | seen |
| ship | 35,185 | 同名 | ship | seen |
| storage tank | 23,320 | 同名 | storage_tank | seen |
| tennis court | 7,343 | 同名 | tennis_court | seen |
| overpass | 1,782 | 父子 | Bridge | novel-仅父类 |
| vehicle | 26,615 | 上位概念 | Small/Large_Vehicle | novel-上位概念 |
| airport | 666 | 近义不合并 | — | novel-严格 |
| chimney | 1,029 | 无关系 | — | novel-严格 |
| expressway-service-area | 1,085 | 无关系 | — | novel-严格 |
| expressway-toll-station | 687 | 无关系 | — | novel-严格 |
| dam | 538 | 近义不合并 | — | novel-严格 |
| golffield | 575 | 近义不合并 | — | novel-严格 |
| stadium | 672 | 近义不合并（弱证据） | — | novel-严格 |
| train station | 509 | 无关系 | — | novel-严格 |
| windmill | 2,998 | 无关系 | — | novel-严格 |

类占比：seen 45%，novel-严格 45%，novel-仅父类 5%，novel-上位 5%；
实例占比：seen 70.1%，novel-严格 7.0%，novel-仅父类 1.4%，novel-上位 21.4%。

## 4. 重点结论

### 4.1 绝不能当 novel 证据的类（与源域直接重合或层级关联）

- **NWPU 全部 10 类**：9 类与 iSAID 同义/同名；vehicle 是源域细类的上位概念。
  NWPU 整集只能作 seen/同分布参照。
- **SOTA 的 15 个同名类**（large-vehicle、swimming-pool、helicopter、bridge、
  plane、ship、soccer-ball-field、basketball-court、ground-track-field、
  small-vehicle、baseball-diamond、tennis-court、roundabout、storage-tank、
  harbor）。
- **SIOR-val 的 11 类**：上表 9 个 seen + vehicle（上位）+ overpass
  （bridge 子类）。
- **FAST 全部 37 类**，其中三类陷阱须点名：
  - 同义陷阱（字面最像 novel）：Small-Car≈Small_Vehicle、
    Football-Field≈Soccer_ball_field、Baseball-Field≈baseball_diamond；
  - 残余桶（字面最像 novel）：other-airplane/other-ship/other-vehicle 等价于
    源域大类概念本身；
  - 其余 29 个机型/船型/车型均为源域大类下位类，Intersection 是 roundabout
    的上位概念。
- 凡含 **vehicle**（NWPU/SIOR）的结论不得表述为“对未见类 vehicle 的泛化”。

### 4.2 可作为 novel 确认集的类别清单（建议）

一级（严格 novel、语义干净、实例量可用；均来自 SIOR-val）：

| 类别 | 实例数 |
|---|---|
| windmill | 2,998 |
| chimney | 1,029 |
| expressway-service-area | 1,085 |
| expressway-toll-station | 687 |
| airport | 666 |
| dam | 538 |
| train station | 509 |

二级（严格 novel 但须警告/抽检后使用）：golffield（575，prompt 建议写
golf course 并抽检确认非足球场）、stadium（672，弱证据：与源域场地类语义
相邻）；SOTA 的 airport（18）、container-crane（11）、helipad（5）合计仅
34 实例，仅可作定性补充。

NWPU 与 FAST 无任何类可进入 novel 确认集；FAST 建议改标为层级迁移
（hierarchical transfer）评测集单独报告。

引用 SIOR/SOTA/FAST 的 novel 结果时必须注明其 GT 为 SAM 伪标签（SAMRS），
与 iSAID 人工标注不同源（`sam3_ovis/docs/gt-provenance.md`）。

### 4.3 遗留的协议问题（不在此决定）

SIOR/NWPU 的 vehicle 占实例 16–21%：评测时预测“small/large vehicle”是否计
命中 GT“vehicle”属评测协议选择，需负责人确认后写入协议，本文只登记问题。

## 5. 已知限制

- 判定基于类别名字符串 + 人工语义映射；未读像素、未做跨数据集图像重合检查
  （后者仍需另检，见 2026-10-28 报告 §2）。
- golffield/stadium/excavator/dam 等边界情形依赖人工判断，已在 JSON 日志中
  附 flags；prompt 定稿后应复核。
- 本报告不构成对任何模型结果的评估。

## 6. 复现命令

数据提取（输出可与 JSON 日志中 `source_categories` / `test_sets.*.classes`
逐条对照；判定层=本文 §2 映射表）：

```bash
export SAM3_OVIS_DATA_ROOT=${SAM3_OVIS_DATA_ROOT:-/data/wangcheng/dataset/SCORE}
python -I - <<'EOF'
import json, collections, os
root = os.environ.get("SAM3_OVIS_DATA_ROOT", "/data/wangcheng/dataset/SCORE")
for key, rel in {
  "iSAID-train": "iSAID/train/instances_train.json",
  "NWPU": "NWPU/val/instances_val.json",
  "SOTA": "SOTA/val/instances_val.json",
  "FAST": "FAST/val/instances_val.json",
  "SIOR-val": "SIOR/val/instances_val.json",
}.items():
    d = json.load(open(os.path.join(root, rel)))
    names = {c["id"]: c["name"] for c in d["categories"]}
    cnt = collections.Counter(a["category_id"] for a in d["annotations"])
    print(key, len(d["images"]), len(d["annotations"]),
          sorted((n, names[i], cnt.get(i, 0)) for i, n in names.items()))
EOF
```

明细与统计（含每类 verdict/flags/实例数、类级与实例级占比）：
`sam3_ovis/logs/g0_novel_class_isolation.json`。
