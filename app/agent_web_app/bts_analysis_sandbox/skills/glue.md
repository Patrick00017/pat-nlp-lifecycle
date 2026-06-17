# 糊间隙赋值诊断指南

## 事件链概览

```
G1  HandleGuGlueMsg（换材消息入口）
  ↓
G7  SetGlueGu entry（解析 material / brand / flute）
  ↓  [可选：材质-设备部位匹配失败]
G10 降级为默认匹配
  ↓  [可选：延迟等待中被新任务抢断]
G8  延迟中取消
  ↓
G14 GU 各部位计算结果（GU1/GU2/GU3，每部位 8 段车速曲线）
  ↓  [可选：写值前被新任务抢占]
G15 写值前取消
  ↓
    WriteVar × 约 48 个 PLC 点（成功无日志）
  ↓
G12 SetGlueGu 写值完成（GU 周期结束）

--- SF 赋值路径 ---

G4 SF 各部位计算结果（SF1/SF2/SF3，每部位 8 段车速曲线）
  ↓
G5  SetGlueSF 写值完成（SF 周期结束）

--- 替代路径 ---
G11 HandleChangeNow（手动立即换材，跳过延迟）
  ↓  （same SetGlueGu/SetGlueSF flow，isChangeNow=true）
```

代码中以 `EventId` 标识事件，控制台和报告里触发原因统一显示为 `代号（描述）` 格式：

```python
TRIG_LABELS = {'G7': '换材触发', 'G11': '立即换材'}
```

例如：`G7（换材触发）`、`G11（立即换材）`。

## 弯翘调平 → 糊间隙赋值交互

### 通信机制

弯翘模块与胶水模块通过 `GlobalControl.execWarpSetDatail.warpPositionValue` 共享字典通信：

```
WarpCtrl（WARP5/WARP6 写入）
  └→ warpPositionValue.AddOrUpdate(GlueGU1/2/3, offset)
  └→ warpPositionValue.AddOrUpdate(GlueSF1/2/3, offset)
        │
        ▼   （读取发生在下一次换材触发时）
GlueCtrl（G4/G14 计算时读取）
  └→ warpPositionValue.TryGetValue(key) → offSet
  └→ setValue += offSet    ← 8 段车速统一添加
```

### 弯翘触发场景

| WARP 事件 | 触发原因 | 对胶水的影响 |
|-----------|---------|-------------|
| WARP5 | 弯翘自动调平 | 写入 GlueSF1/2/3 / GlueGU1/2/3 偏移量 |
| WARP6 | 弯翘手动调平 | 同上 |
| WarpPaperChange | 底纸换材 | 清空所有弯翘偏移至 0 |
| RestCurvedWarp | 弯翘复位 | 清空所有偏移并推送通知 |

### 胶水计算中的弯翘偏移位置

```
setValue = baseFormula × qdmCoef × formCoef × speedCoef
setValue += warpOffset      ← 弯翘偏移（G4 的 offset 字段 / G14 的 warp_offset 字段）
setValue += brandOffset     ← 品牌偏移
```

注意：

- SF1/SF2 日志输出 `弯翘偏移量={offSet}`（G4 模板已解析为 `offset` 字段）。
- GU1/GU2/GU3 在代码中应用了弯翘偏移，但日志不输出（G14 的 `warp_offset` 默认为 0）。
- SF3 同样应用弯翘偏移但日志不输出。

### 时序关键点

```
弯翘调平（异步 1s 循环）              胶水换材（同步触发）
         │                                 │
         ▼                                 ▼
    WARP5/WARP6                      G1/G7 触发
    warpPositionValue                 （等待延迟）
    .AddOrUpdate(key, offset)              │
         │                                ▼
         │                           G4/G14 计算
         │                           TryGetValue(key) → offSet
         │                           setValue += offSet
         │                                ▼
         │                           G12/G5 写值到 PLC
         │
         └── 弯翘偏移量修改不立即生效 ──┘
             仅在下次换材触发胶水计算时生效
```

### 诊断要点

- G4 的 `offset` 字段 ≠ 0 → 该 SF 胶水值受弯翘调平影响。
- G14 的 `warp_offset` 字段 ≠ 0 → 该 GU 胶水值受弯翘调平影响（当前默认 0，需模板支持）。
- 弯翘调平独立于换材循环，偏移量会持续存在直到被清空或覆盖。
- 如果弯翘调平发生在换材延迟期间，则本次换材计算会用到旧的偏移值（不受新调平影响）。

### 异常场景

| 场景 | 问题 |
|------|------|
| WarpPaperChange（底纸换材）清空偏移，但 Glue 尚未计算 | 弯翘偏移重置值可能在胶水计算后被覆盖 → 需确认时序 |
| RestCurvedWarp 清空所有偏移 | 下次换材时胶水值会跳变（偏移归零） |
| 弯翘频繁调平（偏移频繁变化） | 偏移值不稳定的中间状态可能被胶水计算捕捉 |

## 诊断维度

### 1. 周期完整性（已实现）

| 模式 | 含义 |
|------|------|
| G7 后无 G12/G15 | 周期被中断（G8 抢断 / G15 取消 / 异常崩溃） |
| G14 后无 G12 | 计算完成但写入失败/取消 → 检查 G15 |
| G12 前出现多次 G14 | 重复计算 → 系统不稳定 |
| 单个周期内 G14 > 3 次 | 过多计算（`excessive_calculation`） |

代码实现：`GlueGapDiagnostic.check_cycle_completeness()`。

### 2. 取消率（已实现）

- G8 频率（延迟中取消）→ 换材请求过于密集，系统来不及处理。
- G15 频率（写值前取消）→ 浪费计算，比 G8 更严重。
- 当前代码计算方式：`(G8 + G15) / 总周期数 > 30%` 触发警告。

> 注：`glue.md` 早期版本写为 `/ G7`，实际代码分母为 `total_cycles`（包含 G7 与 G11 触发）。

代码实现：`GlueGapDiagnostic.calc_cancellation_rate()`。

### 3. 值合理性（已实现）

从 G14 计算结果检查：

- 车速是否单调递增。
- 相邻车速段糊间隙值跳变是否超过 2.0。
- 是否存在负值。
- 是否超过硬限制 60。
- 弯翘偏移量是否非零（`warp_offset` / `offset` 列）。

> 尚未实现：MinGlue/MaxGlue 范围检查、克重-间隙正相关性验证。

代码实现：`GlueGapDiagnostic.check_value_plausibility(layer)`。

### 4. 材质一致性（已实现）

- 比较 `SetGlueGu`/`SetGlueSF` 中的 `material` 与生命周期 `lifecycle.df.msg` 中的材质。
- 不一致时产生 `material_mismatch`，与跨来源一致性问题（克重不匹配、QDM 系数不匹配、QDM 无配置、基础设置不匹配）共同列为 **确认错误**。
- 若该周期同时为 G11 立即换材，则额外记录 `immediate_change`。

> 注：G11（立即换材）本身不是异常，仅当同周期存在 `material_mismatch` 时作为上下文附带输出。

代码实现：`GlueGapDiagnostic.check_material_consistency()`。

### 5. 跨来源一致性（已实现）

通过 `GlueGapDiagnostic.check_cross_source_consistency()` 实现，需传入 `dev_ips`（`PostgreSQLHelper` 实例）。

- **克重一致性**：提取 `material` 中所有非 `-` 部分，逐段查询 `S_PaperCodes.SPC_GlueWeight` 累加，比对总和与 G14 的 `current_glue_weight`。与 PLC 代码 `GetSumWeight(info.Code)` 逻辑一致。
- **QDM 系数一致性**：使用**压缩码**（去掉 `-` 后的材质码，如 `Q.-.-.0.Q` → `Q.0.Q`）匹配 `TB_IPS_QdmCoefDF`，精确匹配失败时回退 `LIKE` 模糊匹配。比对 `F_Glue1/2/3` 与 G14 的 `qdm_factor`。
- **品牌偏移一致性**（规划中）：当前未实现，需通过 `S_PaperCodeBrands` + `TB_IPS_GlueGuBrand` 联合查询。
- **基础设置一致性**：查询 `TB_IPS_GlueGu`（GU 层）或 `TB_IPS_GlueSF`（SF 层），比对 `F_MinGlue`/`F_MaxGlue`/`F_MinWeight`/`F_MaxWeight` 与 G14 的 `min_glue`/`max_glue`/`min_weight`/`max_weight`。有差异即报告 `base_setting_mismatch`。
- **车速系数一致性**：查询 `TB_IPS_GlueSpeedCoef`（`F_Position` 按 GU1=1, GU2=2, GU3=3, SF1=4, SF2=5 映射），逐段比对 `F_Coef` 与 G14 的 `speed_factor`。差异超过 0.01 时报告 `speed_coef_mismatch`。

结果在控制台输出 `--- 跨来源一致性检查 ---` 段，在 report.md 中以 `## 跨来源一致性检查` 表格展示。

### 6. 全生命周期关联（规划中）

- G12 后验证 `ipsValueInfos` 与 G14 输出是否一致。
- 车速变化时验证 `CalGlueRealTime` 是否正确切换车速段。

> 当前代码未实现。

### 7. 设备通信（规划中）

`comm.WriteVar` 成功时无日志，仅错误会打印：

- `"糊机糊间隙赋值异常"` → PLC 通信故障。
- 检查 `CalGlueRealTime` 错误 → 通信稳定性。

> 当前代码未实现日志关键词搜索。

## 根因追溯："糊间隙值与预期不符"

### 步骤 1：定位生效周期

| 事件模式 | 含义 | 追溯方向 |
|---------|------|---------|
| G7 → ... → G12 | 正常 GU 换材赋值 | 以该周期为生效周期 |
| G7 → ... → G5 | 正常 SF 换材赋值 | 以该周期为生效周期 |
| G11 → ... → G12 | 手动立即换材（GU） | 跳过延迟逻辑 |
| G11 → ... → G5 | 手动立即换材（SF） | 跳过延迟逻辑 |
| G7/11 → G15 | 写值前取消 | 当前值来自上一周期 |
| G7/11 → G8 → interrupted | 延迟中被抢断 | 当前值仍来自上一完成周期，下一完成周期才会生效 |
| 无 G12/G5 | 赋值未完成/失败 | 检查错误日志 |

代码实现：`GlueGapDiagnostic.traceback(target_time, expected_values)`。

### 步骤 2：回退计算输入（G14）

每个 G14 包含 8 段车速曲线。对比：

```
预期值  vs  G14.result
       vs  G14.weight
       vs  G14.qdm_factor
       vs  G14.ui_factor
       vs  G14.speed_factor
       vs  G14.min_glue / G14.max_glue
```

| 症状 | 可能根因 |
|------|---------|
| weight 错误 | 材质解析错误 + 品牌偏移不匹配 |
| QDM 系数错误 | `QdmCtrl.GetQdmDFCoef(paper, flute)` 查询错误 |
| UI 系数错误 | HMI 上手动修改了 `FormSetQdmFactorInfo` |
| 车速段错误 | 设备车速匹配到非预期段 |
| offset 错误 | 弯翘调平偏移干扰 |

代码在 `generate_report()` 中会对 8 段曲线逐段做公式验证：

```
result = base_gap × qdm × ui × speed_coef + offset
```

### 步骤 3：回退材质解析路径（规划中）

从 `G7.material` 和 `G7.brand***` 追踪：

```
G7.material → paperList 解析 → pCodeFloor1/2/3
            → paperOldList → brand 匹配（brandpCodeFloor1/2/3）
            → brandPapers 查询 → BrandOffset
            → allPapers 查询 → weight
```

常见偏差：

- 材质码含 `-` 占位符 → 层数解析错误。
- brand 未在 `brandPapers` 中找到 → BrandOffset 默认 0。
- `paperOldList` 与 `driverList`（用户勾选部位）映射错位 → G10 降级匹配。

> 当前代码未按上述路径自动回溯，仅做最终 material 与 lifecycle 的一致性校验。

### 步骤 4：评估延迟/取消影响

```
G7（入口）
  ↓ 延迟等待
  G8? → 新任务到达 → 当前值仍来自上一周期，下一周期才会生效
  ↓ 计算完成
  G14×N（GU）/ G4×N（SF）
  ↓
  G15? → 写值前取消 → 当前值来自上一周期
  ↓
  G12（GU 写值完成）/ G5（SF 写值完成）
```

关键判断：

- G8 存在 → 值来自**下一**完成周期（尚未写入）。
- G15 存在 → 值来自**上一**完成周期。
- 多次 G8+G15 → 反复抢断，值来自更后面的完成周期。

> 注：G8 前向查找（"值来自下一周期"）当前已实现决策树描述，但代码尚未主动把下一完成周期作为 `active_cycle`。

### 步骤 5：检查异常路径（规划中）

| 日志关键词 | 问题 |
|-----------|------|
| `"糊机糊间隙赋值异常"` | SetGlueGu 异常，DB/PLC 故障 |
| `"糊机糊间隙设备部位和材质匹配失败"` | HMI 勾选部位与材质层数不匹配 |
| `"用户勾选的糊机糊间隙使用部位和材质匹配对应不上，使用默认情况处理"` | G10 降级匹配触发 |
| `HandleChangeNow` + `偏移量={OffSetValue}` | 手动立即换材带覆盖偏移 |

> 当前代码未实现日志关键词自动搜索。

## 诊断决策树

```
用户："某时刻糊间隙值与预期不符"
  ↓
找到最近的 G12（GU 写值完成）/ G5（SF 写值完成）
  ↓
├─ G12/G5 存在 → 该周期为生效周期
│   ├─ 检查 G14/G4：逐段公式验证
│   ├─ 检查 G7：触发材质/楞型
│   ├─ 检查 G10：是否降级匹配
│   ├─ 检查 G11：是否立即换材
│   └─ 检查 material_mismatch：是否确认错误
│
├─ 无 G12/G5，存在 G15 → 值来自上一周期
│   └─ 找上一个 G12/G5
│
├─ 无 G12/G5，存在 G8 → 值来自下一完成周期
│   └─ 找下一个 G12/G5
│
└─ 无 G12/G5，日志中有异常关键词 → 通信/计算故障
    └─ 检查 PLC 通信或 QDM 系数数据库
```

## 糊间隙计算公式

### 完整公式

```
result = base_gap × qdm_coef × ui_coef × speed_coef + offset
```

其中 `base_gap` 由纸克重线性插值得出：

```
base_gap = min_gap + (cur_weight - min_weight) / (max_weight - min_weight) × (max_gap - min_gap)
```

- `min_gap` / `max_gap` — 该材质的最小/最大糊间隙设定值。
- `min_weight` / `max_weight` — 该材质的最小/最大克重范围。
- `cur_weight` — 当前纸卷克重（在最小/最大克重范围内插值）。
- `qdm_coef` — QDM 系数（纸板/楞型决定）。
- `ui_coef` — 界面系数（HMI 手动调整）。
- `speed_coef` — 车速系数（8 段速度曲线，车速越高系数越低）。
- `offset` — 弯翘偏移量（G4 为 `offset`，G14 为 `warp_offset`，GU 日志中不输出，默认 0）。

### 验证示例

从 G4/G14 日志中取第一段车速验证：

```
段1：车速=30
base_gap = 10 + (280 - 200) / (400 - 200) × (35 - 10)
         = 10 + 0.40 × 25
         = 20.00
result   = 20.00 × 0.80 × 1.10 × 1.80 + 0
         = 31.68
验证：通过，与日志记录值一致
```

代码在 `generate_report()` 中会对全部 8 段重复上述计算并标记是否通过。

## 报告输出格式

### 控制台报告（`test.py`）

共 4 个部分：

1. **发现的异常**
   - `[警告]` — 取消率过高、降级匹配、弯翘影响。
   - `[信息]` — 被抢断、无弯翘影响、材质变更记录不一致。
   - 注意：由于 GBK 控制台编码限制，不输出 emoji。

2. **周期详细报告**
   - 以周期为单位合并展示，每个周期包含：
     ```
     周期 #22 (完成)  G7（换材触发）  2026-01-08 17:47:10  材质=Q.-.-.0.Q
       计算值: GU2: @30=45.54 / @60=40.48 / @90=35.42 / @120=30.36
       错误: 材质不匹配（DF材质...不一致）; 克重不匹配（...）; QDM无配置
       警告: 降级匹配; 基础设置不匹配
       信息: 被抢断
     ```
   - `错误` — 材质不匹配、克重不匹配、QDM系数不匹配、QDM无配置、基础设置不匹配。
   - `警告` — 降级匹配、弯翘影响、重复计算、值跳变等。
   - `信息` — 被抢断、写值完成但缺少计算过程、写值取消且没有记录计算。

3. **最近赋值事件序列**
   - 以目标时间为锚点，逆序展示最近 N 个事件：
     ```
     T-5 (#22) *生效 2026-01-08 17:47:10  GU2  Q.-.-.0.Q  @30=45.54 / @60=40.48 ...
                   异常: 降级匹配, 材质不匹配
                   错误: 周期#22: DF生命周期材质"A.-.-.0.A"与G7材质"Q.-.-.0.Q"不一致
     ```
   - `*生效` 表示该周期为当前目标时间的生效周期。
   - `错误:` 行仅在存在 `material_mismatch` 时显示，内容来自 `error_detail` 字段。

4. **排除建议**
   - 提示如何调整时间范围或关注部位重新分析。

### 完整技术报告（`diagnostic_report.md`）

`generate_report()` 生成 Markdown 报告，包含：

- **根因追溯** — 目标时间、生效周期、取消干扰、期望值对比（紧凑表格）。
- **周期详情** — 所有完成周期独立成节，每节包含：
  - 触发信息（时间 / G7/G11 / 材质 / 楞型）
  - 材质变更记录
  - **克重计算说明**（按材质码各段解析 `SPC_GlueWeight` 并累加，与 G14 实际使用值比对）
  - **每层 G14 计算值完整表格** + 公式逐段验证（base_gap 计算 + 8 段车速验证）
  - 错误 / 警告 / 信息标签
- **最近赋值事件序列表格** — T-N 序列 + 结论。结论按周期汇总确认错误，去重后用自然语言表述（如"周期 #22 存在以下错误：材质和系统记录对不上；QDM配方没找到对应配置"）。无错误时输出"这几次赋值都没有发现任何问题，数据正常"。
- 总体周期统计。
- 完整性异常列表。
- **跨来源一致性检查** — 克重 / QDM系数 / 基础设置比对。

## 关键类与方法

### `GlueGapDiagnostic`

| 方法 | 作用 |
|------|------|
| `__init__(extractor, warp_extractor=None, dev_ips=None)` | 初始化，分组周期，可选关联弯翘提取器 + devIPS 连接。 |
| `_group_cycles()` | 把事件流按 G7/G11 起点、G12/G5/G15 终点切分为周期，合并 GU（G12）和 SF（G5）的 `set_values`。 |
| `check_cycle_completeness()` | 检查周期完整性异常。 |
| `calc_cancellation_rate()` | 计算取消率并给出警告。 |
| `check_value_plausibility(layer)` | 检查指定部位的 G14 值合理性。 |
| `check_material_consistency()` | 检查材质一致性，返回 `material_mismatch`。G11（立即换材）仅在同周期存在材质错位时附带输出。 |
| `check_cross_source_consistency()` | 跨来源一致性检查（需 `dev_ips` 连接），返回克重/QDM系数不匹配项。 |
| `traceback(target_time, expected_values, recent_count=5)` | 根因追溯，返回生效周期、取消干扰、弯翘事件、最近赋值序列、跨来源问题。 |
| `generate_report(target_time, expected_values)` | 生成完整 Markdown 报告。 |
| `generate_json(target_time, expected_values)` | 生成完整诊断数据的 JSON 字典（供 Web 前端使用）。 |
| `print_cycle_summary()` | 打印周期汇总表。 |
| `_extract_layer_values(set_values)` | 从 `set_values` 中提取每层的 speed→value 列表。 |

### `TRIG_LABELS`

```python
TRIG_LABELS = {'G7': '换材触发', 'G11': '立即换材'}
```

所有用户可见的触发原因统一使用 `代号（描述）` 格式，例如 `G7（换材触发）`。
