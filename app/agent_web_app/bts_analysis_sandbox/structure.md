# BTS Glue Gap Diagnostic — 数据结构与推理逻辑（v2）

## 目录

1. [整体数据流](#1-整体数据流)
2. [阶段一：event_extractor — 事件提取](#2-阶段一event_extractor--事件提取)
3. [阶段二：GlueGapDiagnosticFSM.run() — 事件分发与处理](#3-阶段二gluegapdiagnosticfsmrun--事件分发与处理)
4. [阶段三：GlueGapDiagnosticFSM.get_results() — 结果汇总与分析](#4-阶段三gluegapdiagnosticfsmget_results--结果汇总与分析)
5. [阶段四：Machine Anomaly Detection — 纸卷异常检测](#5-阶段四machine-anomaly-detection--纸卷异常检测)
6. [最终输出：fsm_results.json](#6-最终输出fsm_resultsjson)
7. [前端读取映射](#7-前端读取映射)
8. [推理逻辑总结](#8-推理逻辑总结)

---

## 1. 整体数据流

```
log_parser.test_ips_and_glue_template_pg()
  ├─→ extractor.process()              → set_func_call_events, material_events
  ├─→ extractor.process_orders()       → order_events
  ├─→ extractor.process_machine_run_data()
  │     └─→ machine_anomaly_events     → 挂载到 order_event.anomalies
  └─→ extractor.get_all_events()       → all_events (merged)

 ── 数据源扩展 ──
 T_IPS_HisRunningData_20260622
   └─→ machine_run_data  (F_CreateTime, F_OrderID, F_MachineID, F_Remainning_mm)

GlueGapDiagnosticFSM.__init__(extractor)
  └─→ self.all_events
       ├─ material_events  (material + order type)
       └─ set_func_call_events  (glue type)

GlueGapDiagnosticFSM.run()
  ├─→ material event → OrderAndMaterialFSM._on_material  → match_list
  ├─→ material event → PositionFSM._on_material          → current_parts (tracking)
  ├─→ glue event     → PositionFSM._on_glue              → full_events (validated)
  └─→ order event    → OrderAndMaterialFSM._on_order     → order_list

GlueGapDiagnosticFSM.get_results()
  ├─→ glue_events[pos][]            ← PositionFSM.full_events
  ├─→ material_events[]             ← all material events
  ├─→ order_check                   ← OrderAndMaterialFSM.get_results()
  │     ├─ order_list[]
  │     ├─ material_list[]
  │     ├─ match_list[]
  │     └─ summary{}
  └─→ glue_events[].analysis[]     ← GlueGapDiagnosticFSM.analyze() (嵌入)
```

---

## 2. 阶段一：event_extractor — 事件提取

### 2.1 GlueEventExtractor.process()

解析 T_Log 日志，匹配模板 G1-G16/I1-I18，生成三类事件。

---

### 2.2 material_event（换材事件）

每个接纸机换材时生成一条。**源头**：I8(正常换材), I12(糊机换材), I13(横切换材), I15(实材换材), I16(初始化)。

```json
{
  "id": "uuid-v1-string",
  "part": "ls0 | ms1 | ls1 | ms2 | ls2 | df",
  "type": "material",
  "msg": "(prev_mat,prev_w,prev_flute) -> (curr_mat,curr_w,curr_flute)",
  "time": "2026-06-22 00:01:12.563278",
  "reason": "normal | hq | real | reset"
}
```

| `reason` | 含义 | 触发条件 |
|----------|------|----------|
| `normal` | 正常接纸机换材 | I8/I12 事件 |
| `hq` | 横切校验换材 | I13 事件 |
| `real` | 实际材质（ERP）直送 | G16 糊机实材 |
| `reset` | 系统初始化 | I16 InitInfos |

---

### 2.3 glue_event（胶水赋值事件）

每个糊间隙位置（GU1-3/SF1-3）完成一次赋值计算时生成。**源头**：G12(GU写值完成), G5(SF写值完成), G13(写入终止)。

```json
{
  "id": "uuid-v1-string",
  "func": "SetGlueGU | SetGlueSF1 | SetGlueSF2",
  "part": "GU1 | GU2 | GU3 | SF1 | SF2 | SF3",
  "type": "glue",
  "material": "E.2.3.4.E | 4/Z",
  "flute_type": "5FC",
  "set_values": {
    "columns": ["speed", "min_glue", "max_glue", "min_weight", "max_weight",
                "current_glue_weight", "speed_factor", "min_speed",
                "qdm_factor", "ui_factor", "value"],
    "data": [[...8 rows × n cols...]]
  },
  "time": "2026-06-22 00:07:15.123456",
  "event_issue": "normal | disable"
}
```

| `event_issue` | 含义 |
|---------------|------|
| `normal` | 正常赋值写入 |
| `disable` | 部位未启用 (G13) |

### 2.4 order_event（订单切换事件）

```json
{
  "id": "uuid-v1-string",
  "type": "order",
  "order_id": "9102",
  "time": "2026-06-22 00:00:00",
  "machine": "MS1",
  "paper_code": "HD.07.07.07.B9",
  "flute": "5BA",
  "width": 2150,
  "erp_paper_code": "07",
  "erp_weight": 170.0,
  "erp_width": 2150.0,
  "anomalies": []
}
```

### 2.5 machine_anomaly_event（纸卷异常事件）

```json
{
  "id": "uuid-v1-string",
  "type": "machine_anomaly",
  "machine": "MS1",
  "time": "2026-06-22 00:05:47",
  "reason": "剩余量频繁波动 | 剩余量长时间停滞 | 剩余量长期未递减",
  "detail": "回升率 20%，周期 300s",
  "start_remaining_mm": 1000.0,
  "end_remaining_mm": 500.0,
  "order_ids": ["9102", "9103"]
}
```

异常事件生成后挂载到对应时间点的 `order_event.anomalies[]`。

---

## 3. 阶段二：GlueGapDiagnosticFSM.run() — 事件分发与处理

### 3.1 PositionFSM 校验输出：full_event

每个 glue_event 经过校验后，追加 `errors/warnings/passes` 到原 event dict。

```json
{
  "errors": [
    {
      "detail": "材质匹配失败",
      "type": "material_dismatch",
      "args": {
        "id": "uuid-material-event",
        "msg": "DF材质匹配失败，赋值材质：A，目前材质：B"
      }
    }
  ],
  "warnings": [
    {
      "detail": "取消",
      "type": "cancel",
      "args": {}
    }
  ],
  "passes": [
    {
      "detail": "克重匹配",
      "type": "weight_pass",
      "args": {"weight_id": "...", "msg": "克重=500g, 档案:500g, 匹配完成"}
    }
  ]
}
```

#### Issue/Pass/Warning 类型枚举

| Type | 含义 | 严重度 |
|------|------|--------|
| `material_dismatch` | 赋值用材质 ≠ 当前实际材质 | 🔴 error |
| `qdm_dismatch` | QDM 系数 ≠ 档案 | 🔴 error |
| `qdm_not_exist` | QDM 配置缺失 | 🔴 error |
| `weight_dismatch` | 克重 ≠ 档案 | 🟠 warning |
| `weight_not_exist` | 克重档案缺失 | 🟠 warning |
| `basedoc_dismatch` | 基础设置 ≠ 档案 | 🟡 warning |
| `basedoc_not_exist` | 基础资料缺失 | 🟡 warning |
| `speed_coef_dismatch` | 车速系数 ≠ 档案 | 🔵 error |
| `speed_coef_not_exist` | 车速系数缺失 | 🔵 error |
| `no_set_values` | 无计算结果 | ⚪ info |
| `cancel` | 赋值取消 (G13/G15) | 🟠 warning |
| `material_pass` | 材质匹配成功 | ✅ pass |
| `qdm_pass` | QDM 匹配 | ✅ pass |
| `weight_pass` | 克重匹配 | ✅ pass |
| `speed_pass` | 车速系数匹配 | ✅ pass |
| `basedoc_pass` | 基础设置匹配 | ✅ pass |

---

### 3.2 OrderAndMaterialFSM 输出：order_check

三个并行数组 + 汇总统计。

#### order_list[]

每次材料事件或订单无换材时 append 当前订单号。

```json
["9102", "9102", "9103", "9103", "9104", ...]
```

#### material_list[]

每次材料事件的时间/部件/msg。

```json
[
  {"time": "2026-06-22 00:01:12.563278", "part": "ls0",
   "msg": "(-,0,-) -> (HD,2050,A)"},
  {"time": "", "part": "", "msg": "(该订单无换材记录)"},
  ...
]
```

#### match_list[]

每条目（与 material_list 对齐）中 `slots` 包含 6 槽位（ls0/ms1/ls1/ms2/ls2/df）的匹配快照。

```json
[
  {
    "material_event_id": "uuid-material-event",
    "order_id": "9102",
    "part": "ls0",
    "slots": {
      "ls0": {
        "actual_material": "HD",
        "expected_material": "HD",
        "actual_width": 2050,
        "expected_width": 2050,
        "match": true,
        "id": "uuid-latest-material-event-for-this-slot"
      },
      "ms1": {
        "actual_material": "-",
        "expected_material": "07",
        "actual_width": 0,
        "expected_width": 2050,
        "match": false,
        "id": null
      },
      "df": {
        "actual_material": "HD.07.-.-.-",
        "expected_material": "HD.07.07.07.LI",
        "actual_width": 2050,
        "expected_width": 2050,
        "match": false,
        "id": "uuid-df-material-event"
      }
    },
    "all_match": false
  },
  null
]
```

**匹配规则**（v2 — 已去掉门幅）：

```
match = (actual_material == expected_material or expected_material == '-')
```

不再校验 `actual_width == expected_width`。只比材质码，不比门幅。

#### summary{}

```json
{
  "9102": {
    "total": 16,
    "matched": 8,
    "mismatched": 8,
    "paper_code": "HD.07.07.07.B9",
    "width": 2050
  }
}
```

---

## 4. 阶段三：GlueGapDiagnosticFSM.get_results() — 结果汇总与分析

### 4.1 glue_events (序列化后)

`PositionFSM.full_events` 转 dict → errors/warnings/passes 序列化 + analysis 嵌入。

### 4.2 analysis[] — GlueGapDiagnosticFSM.analyze() 嵌入

对每个 glue_event 追加 `analysis` 字段：

```json
{
  "analysis": [
    {
      "slot": "ms2 | ls2 | df | ms1 | ls1",
      "actual": "04",
      "current_order": "9107",
      "verdict": "未知材质错误 | 换材滞后（仍在用上一订单材质） | "
                "换材提前（已为下一订单备料） | 实际材质触发（实材直送）",
      "related_order": "9106 | null",
      "reason": "normal | hq | real | unknown",
      "origin": {
        "order_id": "9105",
        "direction": "之前 | 之后",
        "distance_seconds": 10800
      }
    }
  ]
}
```

**判定优先级（从高到低）**：

| 优先级 | 条件 | verdict | 颜色 |
|--------|------|---------|------|
| 1 | reason == 'real' | `实际材质触发（实材直送）` | 🟣 `#8b5cf6` |
| 2 | actual matches prev order | `换材滞后（仍在用上一订单材质）` | 🟠 `#f97316` |
| 3 | actual matches next order | `换材提前（已为下一订单备料）` | 🟡 `#f59e0b` |
| 4 | none | `未知材质错误` | 🔴 `#ef4444` |

**origin 追溯**：仅 `verdict == '未知材质错误'` 时触发，遍历所有历史订单（前后双向），取时间最近的匹配订单。

---

## 5. 阶段四：Machine Anomaly Detection — 纸卷异常检测

### 5.1 数据源

`T_IPS_HisRunningData_20260622` 表按时间查询各接纸机（LS0/MS1/LS1/MS2/LS2/MS3/LS3）的 `F_Remainning_mm`（纸卷剩余毫米数）。

```
F_CreateTime, F_OrderID, F_MachineID, F_Remainning_mm
```

### 5.2 周期模型

纸卷剩余量呈锯齿波递减：

```
剩余量(mm)
  ^
  |  /\── 新卷开始 (delta ≥ 500)
  | /  \── 单调递减 (每秒 1-3mm)
  |/    \── 逼近 0 → 下一个跳升
  -----------------------------------→ 时间
```

### 5.3 异常检测

| 异常类型 | 条件 | 阈值 |
|----------|------|------|
| 剩余量频繁波动 | 正变化比例 > 阈值 | `MAX_POSITIVE_RATIO = 0.15` |
| 剩余量长时间停滞 | 零值比例 > 阈值 | `MAX_ZERO_RATIO = 0.5` |
| 剩余量长期未递减 | 连续非递降秒数 > 阈值 | `MAX_FLAT_SEG = 300` |

正常周期（递减 → 接近 0 → 跳升）不输出任何事件。异常周期输出 `machine_anomaly_event` 并挂载到对应 `order_event.anomalies[]`。

---

## 6. 最终输出：fsm_results.json

```json
{
  "glue_events": {
    "GU1": [{ /* full_event + analysis */ }],
    "GU2": [...], "GU3": [...],
    "SF1": [...], "SF2": [...], "SF3": [...]
  },
  "material_events": [{ /* material_event */ }],
  "order_check": {
    "order_list": ["9102", ...],
    "material_list": [{ /* time, part, msg */ }],
    "match_list": [{ /* slots + all_match */ }],
    "summary": { "9102": { "total": 16, ... } }
  },
  "qdm_df": [{ "paper": "...", "flute": "...", "glue1": ..., "glue2": ..., "glue3": ... }],
  "qdm_sf": [{ "ms": "...", "ls": "...", "flute": "...", "glue": ... }],
  "basedoc_gu": [{ "flute": "...", "position": "...", "min_glue": ..., ... }],
  "basedoc_sf": [{ "flute": "...", "min_glue": ..., "max_glue": ..., ... }],
  "speed_coef": [{ "position": 1, "speed": 80, "coef": 1.05 }],
  "paper_codes": [{ "code": "A", "weight": 145 }],
  "description": "..."
}
```

> 注：`order_event.anomalies` 在 `get_all_events()` 中合入事件流，不在 `fsm_results.json` 顶层字段，而是通过前端从 `material_events` 的 `order` 类型条目读取。

---

## 7. 前端读取映射

### FSMViewer.jsx + TimelineView.jsx + ChartView.jsx

| JSON 字段 | 前端组件 | 用途 |
|-----------|----------|------|
| `glue_events[pos][]` | FSMViewer → TimelineView | 时间线显示胶水事件 |
| `glue_events[pos][].errors[]` | TimelineView (IssueBadge) | 红/黄色错误标签 |
| `glue_events[pos][].warnings[]` | TimelineView (IssueBadge) / ChartView (IssuePanel) | ⚠️ 警告标签 |
| `glue_events[pos][].passes[]` | ChartView (IssuePanel) | ✅ 通过标签 |
| `glue_events[pos][].set_values` | ChartView | 详情表格 |
| `glue_events[pos][].analysis[]` | OrderMatchTimeline / ChartView | 泳道颜色判定 / 根因分析区块 |
| `material_events[]` | FSMViewer → TimelineView (MAT tab) | 换材时间线 |
| `order_check.order_list[], material_list[], match_list[]` | OrderMatchTimeline | 订单匹配泳道 |
| `order_check.summary{}` | OrderMatchTimeline | 订单纸码 + 宽度 |

### OrderMatchTimeline.jsx（订单匹配）

- **竖向表格**：每行一个时间点，列=槽位
  - `时间 | 订单 | ls0 | ms1 | ls1 | ms2 | ls2 | df | GU1 | GU2 | GU3 | SF1 ms1 | SF1 ls1 | SF2 ms2 | SF2 ls2`
- 颜色规则见 §4.2 表
- 订单切换时白色粗边框分隔
- **空状态**：无数据时显示 `📋 暂无订单匹配数据` + `🔄 加载数据` 按钮
- **可点击胶水单元格** → 弹出 Modal：
  - 左：`ChartView` 组件（表格 + IssuePanel + 根因分析 + 附近换材记录）
  - 右：Bot 面板（🤖 分析摘要），点击 ✨ 按钮发起 AI SSE 请求
- `origin` 溯源信息通过单元格 tooltip 展示
- Bot 面板使用 `sharedThreadId` 共享外层 OpenCodeChat 的 session

### ChartView.jsx（胶水详情弹窗）

- Header 右侧 `✨` 按钮（蓝色渐变，类似 Copilot 风格）
- 点击 ✨ → 拼接 prompt（参数校验 + 根因分析 + 赋值参数） → SSE 流式显示在 Bot 面板
- Bot 面板：流式输出时闪烁光标 `▋`，完成时消失，自动滚动至底部
- **根因分析**区块：显示每个槽位的 verdict + 溯源信息

---

## 8. 推理逻辑总结

### Layer 1: 参数独立校验 (PositionFSM._on_glue)

```
材料校验    │ actual_material == current_parts[part] → pass/dismatch
QDM 校验    │ qdm_factor == DB value → pass/dismatch
克重校验    │ glue_weight == sum(SPC_GlueWeight per slot) → pass/dismatch
基础资料校验│ min_glue/max_glue/min_weight/max_weight == DB → pass/dismatch
车速系数校验│ speed_factor == DB per position → pass/dismatch
```

---

### Layer 2: 订单-材质匹配 (OrderAndMaterialFSM)

**规则（v2 — 已去掉门幅）：**

```
match = (actual_material == expected_material or expected_material == '-')
```

只比材质码，不比门幅。

---

### Layer 3: 相位差 + 实材判别 + 溯源 (GlueGapDiagnosticFSM.analyze)

**决策树（v2）：**

```
match ?
├─ true  → skip (normal)
└─ false
    ├─ reason='real'  → 实际材质触发（优先级最高）
    ├─ actual matches order before current → 换材滞后
    ├─ actual matches order after current  → 换材提前
    └─ unknown
        └─ bidirectional origin search → 材质溯源 (or null)
```

- 溯源搜索全部订单（前后双向），取时间最近匹配
- `origin` 包含 `order_id`、`direction`（之前/之后）、`distance_seconds`

---

### Layer 4: Machine Anomaly Detection (KeyEventExtractor.process_machine_run_data)

**独立层**— 纸卷剩余量锯齿波检测：

```
正常周期（递减→0→跳升）→ 不输出
异常周期 → machine_anomaly_event
  ├─ 频繁波动（回升率 > 15%）
  ├─ 长时间停滞（零值率 > 50%）
  └─ 长期未递减（连续 ≥ 300s 不降）
```

异常事件挂载到对应时间点的 `order_event.anomalies[]`。

---

### 表达方式总结

| 层 | 问题 | 输出 | 可视化 |
|----|------|------|--------|
| Layer 1 | 这个赋值计算正确吗？ | errors/warnings/passes | Timeline Badges + Detail Panel |
| Layer 2 | 换材跟订单匹配吗？ | match_list → 6槽×全部换材 | 订单泳道（绿/红/gray） |
| Layer 3 | 不匹配的原因是什么？ | analysis → verdict + origin | 胶水泳道（5色）+ ChartView 根因区块 + tooltip 溯源 |
| Layer 4 | 纸卷剩余量正常吗？ | machine_anomaly_event | order_event.anomalies[] |

### 前端交互链

```
订单匹配泳道 (OrderMatchTimeline)
  ├─ 单元格悬停 → tooltip (actual vs expected, origin)
  ├─ 点击胶水库 → Modal:
  │   ├─ ChartView (set_values表 + IssuePanel + 根因分析 + 换材记录)
  │   └─ ✨ → Bot 面板 (AI SSE 流式分析)
  │        └─ sharedThreadId (共享外层 OpenCodeChat session)
  └─ 🔄 刷新按钮 + 📋 空状态
```
