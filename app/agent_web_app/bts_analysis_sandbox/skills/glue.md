# Glue Gap Assignment Diagnostic Guide

## Event Chain Overview

```
G1  HandleGuGlueMsg (triggered by material change)
  ↓
G7  SetGlueGu entry (material, brand, flute parsed)
  ↓  [optional: 材质-部位匹配失败]
G10  fallback to default matching
  ↓  [optional: 延迟等待中被新任务打断]
G8   task cancelled during delay
  ↓
G14  GU1 calculation result (8 speed segments)
G14  GU2 calculation result (8 speed segments)
G14  GU3 calculation result (8 speed segments)
  ↓  [optional: 写值前被新任务抢占]
G15  pre-write cancellation
  ↓
     WriteVar × ~48 PLC points (silent, no log per write)
  ↓
G12  SetGlueGu write complete

--- Alternate path ---
G11  HandleChangeNow (manual immediate change, skips delay)
  ↓  (same SetGlueGu flow with isChangeNow=true)
```

## Warp Leveling → Glue Assignment Interaction

### Communication Mechanism

弯翘模块与胶水模块通过 `GlobalControl.execWarpSetDatail.warpPositionValue` 共享字典通信：

```
WarpCtrl (WARP5/WARP6 写入)
  └→ warpPositionValue.AddOrUpdate(GlueGU1/2/3, offset)
  └→ warpPositionValue.AddOrUpdate(GlueSF1/2/3, offset)
        │
        ▼   (读取发生在下一次换材触发时)
GlueCtrl (G4/G14 计算时读取)
  └→ warpPositionValue.TryGetValue(key) → offSet
  └→ setValue += offSet    ← 8段车速统一添加
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
setValue += warpOffset      ← 弯翘偏移（G4的offset字段 / G14的warp_offset字段）
setValue += brandOffset     ← 品牌偏移
```

**注意**：
- SF1/SF2 日志输出 `弯翘偏移量={offSet}`（G4 模板已解析为 `offset` 字段）
- GU1/GU2/GU3 在代码中应用了弯翘偏移，但日志**不输出**（G14 的 warp_offset 默认为 0）
- SF3 同样应用弯翘偏移但日志**不输出**

### 时序关键点

```
弯翘调平 (异步 1s 循环)             胶水换材 (同步触发)
         │                                │
         ▼                                ▼
    WARP5/WARP6                      G1/G7 触发
    warpPositionValue                 (等待延迟)
    .AddOrUpdate(key, offset)              │
         │                                 ▼
         │                            G4/G14 计算
         │                            TryGetValue(key) → offSet
         │                            setValue += offSet
         │                                 ▼
         │                            G12/G5 写值到 PLC
         │
         └── 弯翘偏移量修改不立即生效 ──┘
             仅在下次换材触发胶水计算时生效
```

### 诊断要点

- G4 的 `offset` 字段 ≠ 0 → 该 SF 胶水值受弯翘调平影响
- G14 的 `warp_offset` 字段 ≠ 0 → 该 GU 胶水值受弯翘调平影响（当前默认 0，需模板支持）
- 弯翘调平独立于换材循环，偏移量会持续存在直到被清空或覆盖
- 如果弯翘调平发生在换材延迟期间，则本次换材计算会用到旧的偏移值（不受新调平影响）

### 异常场景

| 场景 | 问题 |
|------|------|
| WarpPaperChange（底纸换材）清空偏移，但 Glue 尚未计算 | 弯翘偏移重置值可能在胶水计算后被覆盖 → 需确认时序 |
| RestCurvedWarp 清空所有偏移 | 下次换材时胶水值会跳变（偏移归零） |
| 弯翘频繁调平（偏移频繁变化） | 偏移值不稳定的中间状态可能被胶水计算捕捉 |

## Diagnostic Dimensions

### 1. Timing Completeness

| Pattern | Meaning |
|---------|---------|
| `G7` without subsequent `G12` | Assignment interrupted (cancelled by G8/G15, or crash) |
| `G14` without `G12` | Calculation done but write failed/cancelled → check G15 |
| `G14` appears multiple times before `G12` | Multiple recalculations → system instability |
| Long gap `G7` → `G14` | Material parsing or QDM coefficient query timeout |

### 2. Cancellation Rate

- **G8 frequency** (cancelled during delay wait) → material change requests too密集, system overwhelmed
- **G15 frequency** (cancelled right before write) → wasted computation, more severe than G8
- `(G8 + G15) / G7 > 30%` likely indicates scheduling issues

### 3. Value Plausibility (from G14)

- **Speed segment monotonicity**: 8 speed values must be strictly increasing
- **Glue gap range**: computed values within `[MinGlue, MaxGlue]` and `[0, 60]` hard limit
- **Curve smoothness**: adjacent segments should not have drastic jumps
- **Weight vs gap correlation**: heavier paper should generally produce wider gap

### 4. Cross-Source Consistency

- **G7 material+brand** vs **G14 computed weight** → brand offset lookup correctness
- **G11 offset value** vs **SetGlueGu warp offset** → consistency check
- **G1/G7 flute type** vs **G14 QDM coefficient** → QDM lookup correctness

### 5. Full Lifecycle Correlation

After `G12`, verify `ipsValueInfos` match G14 output for corresponding GU position.
When machine speed changes, verify `CalGlueRealTime` correctly switches speed segments.

### 6. Device Communication

`comm.WriteVar` is silent on success; only errors appear:
- `"糊机糊间隙赋值异常"` in logs → PLC communication failure
- Check `CalGlueRealTime` for errors → communication stability

## Root Cause Traceback: "Glue gap value is not what I expected"

### Step 1: Identify the target write cycle

Find the nearest complete assignment cycle around the target time:

| Event Pattern | Meaning | Traceback Direction |
|--------------|---------|-------------------|
| `G7 → ... → G12` | Normal material change assignment | Follow this cycle |
| `G11 → ... → G12` | Immediate change (manual) | Skip delay logic |
| `G7/11 → G15` | Cancelled before write | **Final value came from previous cycle** |
| No `G12` | Assignment incomplete/failed | Check error logs |

### Step 2: Backtrack through calculation inputs (G14)

Each G14 contains 8 speed-segment curves. Compare:

```
Expected value  vs  G14.result
                vs  G14.weight
                vs  G14.qdm_factor
                vs  G14.ui_factor
                vs  G14.speed_factor
                vs  G14.min_glue / G14.max_glue
```

| Symptom | Likely Root Cause |
|---------|------------------|
| Wrong weight | Material parsing error + brand offset mismatch |
| Wrong QDM factor | `QdmCtrl.GetQdmDFCoef(paper, flute)` lookup error |
| Wrong UI factor | User manually modified `FormSetQdmFactorInfo` on HMI |
| Wrong speed segment | Machine speed matched unexpected segment |
| Wrong offset | **WarpCtrl offset interference** (gu1Offset/gu2Offset/gu3Offset) |

### Step 3: Backtrack material parsing path

From `G7.material` and `G7.brand***`, trace:

```
G7.material → paperList parsing → pCodeFloor1/2/3
            → paperOldList → brand match (brandpCodeFloor1/2/3)
            → brandPapers query → BrandOffset
            → allPapers query → weight
```

Common deviations:
- Material code contains `-` placeholder → layer count parsing mismatch
- Brand not found in `brandPapers` → BrandOffset defaults to 0
- `paperOldList` vs `driverList` (user-enabled parts) **mapping misalignment** → G10 fallback

### Step 4: Evaluate delay/cancellation impact

```
G7 (entry)
  ↓ delay waiting
  G8? → new task arrives → final value comes from later cycle
  ↓ calculation done
  G14×3
  ↓
  G15? → cancelled before write → final value comes from previous cycle
  ↓
  G12 (write complete)
```

Key judgment:
- G8 exists → value is from the **next** cycle
- G15 exists → value is from the **previous** cycle
- Multiple G8+G15 → repeated抢断, value from later cycles

### Step 5: Check exception paths

| Log Keyword | Problem |
|------------|---------|
| `"糊机糊间隙赋值异常"` | SetGlueGu exception, DB/PLC fault |
| `"糊机糊间隙设备部位和材质匹配失败"` | User's GUI parts don't match material layers |
| `"用户勾选的糊机糊间隙使用部位和材质匹配对应不上，使用默认情况处理"` | G10 fallback triggered |
| `HandleChangeNow` + `偏移量={OffSetValue}` | Manual immediate change with override offset |

## Diagnostic Decision Tree

```
User: "Glue gap value at this time is not what I expected"
  ↓
Find nearest G12 (write complete)
  ↓
├─ G12 exists → value from this cycle →
│   ├─ Check G14: compare each calculation parameter
│   ├─ Check G7: material/brand parsing
│   ├─ Check G10: fallback triggered?
│   └─ Check G11: immediate change (skipped delay)?
│
├─ No G12, G15 exists → value from previous cycle →
│   └─ Find previous G12
│
├─ No G12, G8 exists → value from later cycle →
│   └─ Find next G12
│
└─ No G12, exception in logs → communication/compute fault
    └─ Check PLC communication or QDM coefficient DB
```

## Glue Gap Calculation Formula

### 完整公式

```
result = base_gap × qdm_coef × ui_coef × speed_coef + offset
```

其中 `base_gap` 由纸克重线性插值得出：

```
base_gap = min_gap + (cur_weight - min_weight) / (max_weight - min_weight) × (max_gap - min_gap)
```

- `min_gap` / `max_gap` — 该材质的最小/最大糊间隙设定值
- `min_weight` / `max_weight` — 该材质的最小/最大克重范围
- `cur_weight` — 当前纸卷克重（在最小/最大克重范围内插值）
- `qdm_coef` — QDM系数（纸板/楞型决定）
- `ui_coef` — 界面系数（HMI 手动调整）
- `speed_coef` — 车速系数（8段速度曲线，车速越高系数越低）
- `offset` — 弯翘偏移量（G4为`offset`，G14为`warp_offset`，GU 日志中不输出，默认0）

### 验证示例

从 G4/G14 日志数据中，取第一段车速结果验证：

```
段1：车速=30
base_gap = 10 + (280 - 200) / (400 - 200) × (35 - 10)
         = 10 + 0.40 × 25
         = 20.00
result   = 20.00 × 0.80 × 1.10 × 1.80 + 0
         = 31.68
验证：✓ 与日志记录值一致
```

### 控制台报告输出格式

`test.py` 的 DB 模式输出三段式报告：

1. **发现的异常** — 按严重程度列出问题：
   - `⚠️` — 需处理（取消率过高、降级匹配、弯翘影响）
   - `ℹ️` — 仅供参考（被抢断、无弯翘影响）
2. **周期概览** — 每行一个赋值周期，含时间、状态、问题标签
3. **糊间隙计算值** — 仅显示完成的周期中 8 段车速的最终结果：

   ```
   周期 #1 (SF2) → 31.68 / 28.16 / 24.64 / 21.12 / 19.36 / 17.60 / 15.84 / 14.96
   ```

### 完整技术报告（diagnostic_report.md）

`generate_report()` 生成包含：
- 根因追溯 + 生效周期详情
- 完整 8 段车速曲线表 + **计算说明**（公式验证）
- 弯翘调平影响（有弯翘事件时）
- 总体周期统计 + 完整性异常列表
