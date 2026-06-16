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
