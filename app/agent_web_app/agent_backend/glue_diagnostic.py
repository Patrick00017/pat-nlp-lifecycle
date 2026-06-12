#!/usr/bin/env python3
"""
GlueCtrl 胶水间隙诊断分析工具
根据日志记录重现 SetGlueGu / SetGlueSF1/2/3 的完整计算逻辑并进行逐项验证。

输入:
  --records   records.json   (必需) 日志记录列表
  --config    config.json    (可选) 扁平配置,提供后可做全量联表校验
  --output    report.json    (可选) JSON 报告输出

输出: 控制台诊断报告 + (可选) JSON 文件

配置格式 (config.json):
  {
    "auto_delay": {
      "DF":  {"type1": 3.0, "type2": 5.0},
      "MS1": {"type1": 2.5, "type2": 4.5},
      "MS2": {"type1": 2.5, "type2": 4.5},
      "MS3": {"type1": 2.5, "type2": 4.5}
    },
    "form_set": {
      "GU_1st_IsOn": true, "GU_2nd_IsOn": true, "GU_3rd_IsOn": false,
      "SF1_IsOn": true, "SF2_IsOn": true, "SF3_IsOn": true,
      "GU_1st_Form_Factor": 1.15, "GU_2nd_Form_Factor": 1.10, "GU_3rd_Form_Factor": 1.00,
      "SF1_Form_Factor": 1.10, "SF2_Form_Factor": 1.10, "SF3_Form_Factor": 1.10
    },
    "glue_set": {
      "GU": [
        {"position": "Floor1", "min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 500, "coef": 1.0},
        {"position": "Floor2", "min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 500, "coef": 1.0},
        {"position": "Floor3", "min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 500, "coef": 1.0}
      ],
      "SF1": {"min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 400, "coef": 1.0},
      "SF2": {"min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 400, "coef": 1.0},
      "SF3": {"min_glue": 10, "max_glue": 35, "min_weight": 200, "max_weight": 400, "coef": 1.0}
    },
    "speed_coef": [
      {"position": "SF2", "speed": 30, "coef": 1.80, "min_value": 30},
      ...
    ]
  }
"""

import json
import sys
import argparse
from datetime import datetime
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _d(val) -> Decimal:
    """安全转为 Decimal，处理字符串/数字/None。"""
    if val is None:
        return Decimal(0)
    return Decimal(str(val))


def _rd(val: Decimal, ndigits: int = 2) -> Decimal:
    """用 ROUND_HALF_EVEN (银行家舍入, C# decimal 默认) 舍入。"""
    quant = Decimal("0.1") ** ndigits
    return val.quantize(quant, rounding=ROUND_HALF_EVEN)


def _parse_time(ts: str) -> Optional[datetime]:
    """解析时间戳字符串，支持 %f 及不带微秒的格式。"""
    if not ts or not ts.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_msg_codes(msg: str):
    """从生命周期消息 '(prev) -> (curr)' 中提取前后纸板代码。"""
    if "->" not in msg:
        return None, None
    parts = msg.split("->")
    if len(parts) != 2:
        return None, None
    prev_str = parts[0].strip().strip("()")
    curr_str = parts[1].strip().strip("()")
    prev_fields = prev_str.split(",") if prev_str else []
    curr_fields = curr_str.split(",") if curr_str else []
    prev_code = prev_fields[0] if prev_fields else ""
    curr_code = curr_fields[0] if curr_fields else ""
    return prev_code, curr_code


# ---------------------------------------------------------------------------
# 诊断结果容器
# ---------------------------------------------------------------------------


class CheckResult:
    """单条检查结果。"""

    def __init__(self, name: str):
        self.name = name
        self.status = "PASS"  # PASS | WARN | ERROR
        self.detail = ""
        self.per_point: list[dict] = []

    def ok(self, detail: str = ""):
        self.status = "PASS"
        self.detail = detail

    def warn(self, detail: str):
        if self.status != "ERROR":
            self.status = "WARN"
        self.detail = detail

    def error(self, detail: str):
        self.status = "ERROR"
        self.detail = detail

    @property
    def icon(self) -> str:
        return {"PASS": "[PASS]", "WARN": "[WARN]", "ERROR": "[ERR!]"}[self.status]

    def __repr__(self):
        return f"  {self.icon} {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# 核心诊断类
# ---------------------------------------------------------------------------


class GlueRecordDiagnostic:
    """对一条日志记录执行 9 项诊断检查。"""

    # 各函数期望的生命周期部件列表
    PARTS_LIFECYCLE = {
        "SetGlueGu": ["ls0", "ms1", "ls1", "ms2", "ls2", "df"],
        "SetGlueSF1": ["ms1", "ls1"],
        "SetGlueSF2": ["ms2", "ls2"],
        "SetGlueSF3": ["ms3", "ls3"],
    }

    # 函数 → 自动延迟位置
    FUNC_POS_MAP = {
        "SetGlueGu": "DF",
        "SetGlueSF1": "MS1",
        "SetGlueSF2": "MS2",
        "SetGlueSF3": "MS3",
    }

    def __init__(self, record: dict, config: Optional[dict] = None):
        self.rec = record
        self.cfg = config or {}
        self.func = record["func"]
        self.part = record.get("part", "")
        self.material = record.get("material", "")
        self.flute = record.get("flute_type", "")
        self.set_values: dict = record.get("set_values", {})
        self.lifecycle: dict = record.get("lifecycle", {})
        self.ts_str = record.get("time", "")
        self.results: list[CheckResult] = []

    # ── 1. 生命周期完整性 ──────────────────────────────────

    def check_lifecycle(self) -> CheckResult:
        r = CheckResult("lifecycle")
        expected = self.PARTS_LIFECYCLE.get(self.func, [])
        actual = []
        for k in expected:
            v = self.lifecycle.get(k, {})
            if isinstance(v, dict) and v.get("msg", "").strip():
                actual.append(k)
        missing = [k for k in expected if k not in actual]
        if not missing:
            r.ok(f"{len(actual)}/{len(expected)} parts received")
        else:
            r.warn(f"missing lifecycle parts: {missing}")
        return r

    # ── 2. 触发-执行时序 ──────────────────────────────────

    def check_timing(self) -> CheckResult:
        r = CheckResult("timing")
        sf = self.lifecycle.get("set_func", {})
        exec_ts = sf.get("time", "")
        exec_dt = _parse_time(exec_ts)
        if not exec_dt:
            r.warn("no set_func exec time")
            return r

        # 找出生命周期中最后一条消息的时间
        expected_parts = self.PARTS_LIFECYCLE.get(self.func, [])
        last_msg_dt: Optional[datetime] = None
        for k in expected_parts:
            v = self.lifecycle.get(k, {})
            t = _parse_time(v.get("time", ""))
            if t and (last_msg_dt is None or t > last_msg_dt):
                last_msg_dt = t

        if last_msg_dt is None:
            r.warn("no lifecycle timestamps available")
            return r

        delay_sec = (exec_dt - last_msg_dt).total_seconds()

        # 从配置获取预期延迟
        expected = self._expected_delay_sec()
        if expected is None:
            r.ok(f"actual delay = {delay_sec:.1f}s (no config)")
        elif abs(delay_sec - expected) <= 3.0:
            r.ok(f"{delay_sec:.1f}s vs expected ~{expected:.1f}s")
        elif delay_sec < expected - 3.0:
            r.warn(
                f"delay too short: {delay_sec:.1f}s < expected ~{expected:.1f}s (executed before waiting finish)"
            )
        else:
            r.warn(
                f"delay too long: {delay_sec:.1f}s > expected ~{expected:.1f}s (check machine speed)"
            )
        return r

    def _expected_delay_sec(self) -> Optional[float]:
        """从 auto_delay 配置读取预期延迟秒数。"""
        ad = self.cfg.get("auto_delay", {})
        pos = self.FUNC_POS_MAP.get(self.func)
        if not pos or pos not in ad:
            return None
        direction = self._weight_direction()
        if direction is None:
            # 未知方向，取两个类型中较大的作为粗估值
            vals = [v for v in ad[pos].values() if isinstance(v, (int, float))]
            return max(vals) if vals else None
        key = "type1" if direction == "up" else "type2"
        return ad[pos].get(key)

    def _weight_direction(self) -> Optional[str]:
        """根据生命周期消息推测纸重变化方向: 'up' | 'down' | None。"""
        expected = self.PARTS_LIFECYCLE.get(self.func, [])
        for k in reversed(expected):
            v = self.lifecycle.get(k, {}).get("msg", "")
            prev_code, curr_code = _extract_msg_codes(v)
            if prev_code is None:
                continue
            if prev_code == "-" and curr_code != "-":
                return "up"  # 空→有纸，肯定升重
            if prev_code != "-" and curr_code == "-":
                return "down"  # 有纸→空，肯定降重
            # 有→有: 无法从代码直接判定重量变化，需 DB 查询
        return None

    # ── 3. 自动延迟参数 ──────────────────────────────────

    def check_auto_delay(self) -> CheckResult:
        r = CheckResult("auto_delay")
        ad = self.cfg.get("auto_delay", {})
        pos = self.FUNC_POS_MAP.get(self.func)
        if not pos:
            r.ok("no auto_delay position for this function")
            return r
        if pos not in ad:
            r.warn(f"position '{pos}' not configured in auto_delay")
            return r
        direction = self._weight_direction()
        if direction:
            r.ok(f"pos={pos}, direction={direction}, config={ad[pos]}")
        else:
            r.ok(f"pos={pos}, direction=unknown, config={ad[pos]}")
        return r

    # ── 4. 材料编码解析 ──────────────────────────────────

    def check_material_parsing(self) -> CheckResult:
        r = CheckResult("material_parsing")
        parsed = self._parse_material()
        if parsed["error"]:
            r.error(parsed["error"])
            return r

        if self.func.startswith("SetGlueGu"):
            cnt = parsed["count"]
            detail = f"paperList count={cnt}"
            if parsed.get("pCodeFloor1"):
                detail += f", floor1={parsed['pCodeFloor1']}"
            if parsed.get("pCodeFloor2"):
                detail += f", floor2={parsed['pCodeFloor2']}"
            if parsed.get("pCodeFloor3"):
                detail += f", floor3={parsed['pCodeFloor3']}"
            if cnt not in (3, 4, 5, 6, 7):
                r.warn(detail + " (unusual count)")
            else:
                r.ok(detail)
        else:
            ms = parsed.get("ms_code", "")
            ls = parsed.get("ls_code", "")
            if not ms or not ls:
                r.error(f"missing MS/LS: MS={ms}, LS={ls}")
            else:
                r.ok(f"MS={ms}, LS={ls}")
        return r

    def _parse_material(self) -> dict:
        """模拟代码 lines 156-170 的材料解析。"""
        result = {"error": None, "count": 0}
        mat = self.material or ""
        if self.func.startswith("SetGlueGu"):
            if "." in mat:
                paper_old = mat.split(".")
            else:
                paper_old = list(mat)
            paper_list = [p for p in paper_old if p != "-"]
            cnt = len(paper_list)
            result.update(
                {
                    "count": cnt,
                    "paper_old_list": paper_old,
                    "paper_list": paper_list,
                    "pCodeFloor1": "",
                    "pCodeFloor2": "",
                    "pCodeFloor3": "",
                }
            )
            if cnt >= 3:
                result["pCodeFloor1"] = f"{paper_list[0]}/{paper_list[1]}"
            if cnt >= 5:
                result["pCodeFloor2"] = f"{paper_list[2]}/{paper_list[3]}"
            if cnt >= 7:
                result["pCodeFloor3"] = f"{paper_list[4]}/{paper_list[5]}"
            if cnt not in (3, 4, 5, 6, 7):
                result["error"] = f"paperList count={cnt} not in expected set (3-7)"
        else:
            parts = mat.split("/")
            result.update(
                {
                    "ms_code": parts[0] if len(parts) > 0 else "",
                    "ls_code": parts[1] if len(parts) > 1 else "",
                }
            )
        return result

    # ── 5. 纸层-驱动位映射 ──────────────────────────────────

    def check_driver_mapping(self) -> CheckResult:
        r = CheckResult("driver_mapping")
        actual_drivers = list(self.set_values.keys())

        if self.func.startswith("SetGlueGu"):
            parsed = self._parse_material()
            if parsed["error"]:
                r.warn(f"material parse failed: {parsed['error']}")
                return r
            cnt = parsed["count"]
            floor_count = {3: 1, 4: 1, 5: 2, 6: 2, 7: 3}.get(cnt, 0)

            fs = self.cfg.get("form_set", {})
            if fs:
                expected_drivers = []
                if fs.get("GU_1st_IsOn"):
                    expected_drivers.append("GU1")
                if fs.get("GU_2nd_IsOn"):
                    expected_drivers.append("GU2")
                if fs.get("GU_3rd_IsOn"):
                    expected_drivers.append("GU3")
                expect_active = expected_drivers[:floor_count]
                detail = f"floors={floor_count}, expected={expect_active}, actual={actual_drivers}"
                if not actual_drivers:
                    r.warn(f"no driver data, {detail}")
                elif len(actual_drivers) != len(expect_active):
                    r.warn(f"driver count mismatch, {detail}")
                elif actual_drivers[0] not in expected_drivers:
                    r.warn(f"mapped driver {actual_drivers[0]} unexpected, {detail}")
                else:
                    r.ok(detail)
            else:
                r.ok(f"actual drivers={actual_drivers} (no form_set config)")
        else:
            r.ok(f"SF drivers={actual_drivers}")
        return r

    # ── 6. CalGlueGap 基值验算 ──────────────────────────

    def check_base_calculation(self) -> CheckResult:
        r = CheckResult("base_calculation")
        points = self._get_speed_points()
        if not points:
            r.warn("no speed-data points")
            return r

        max_err = Decimal(0)
        mismatches = []
        for pt in points:
            try:
                min_g = _d(pt["min_glue"])
                max_g = _d(pt["max_glue"])
                min_w = _d(pt["min_weight"])
                max_w = _d(pt["max_weight"])
                cur_w = _d(pt["current_glue_weight"])
                qdm = _d(pt.get("qdm_factor", 1))
                form = _d(pt.get("ui_factor", 1))
                spd = _d(pt.get("speed_factor", 1))
                rec_val = _d(pt["value"])
            except (KeyError, ValueError) as e:
                mismatches.append(str(e))
                continue

            if max_w <= min_w:
                mismatches.append(
                    f"speed={pt.get('speed','?')}: max_weight={max_w} <= min_weight={min_w}"
                )
                continue

            # CalGlueGap 公式: MinG + (MaxG-MinG) * (curW - minW) / (maxW - minW)
            base_exp = min_g + (max_g - min_g) * (cur_w - min_w) / (max_w - min_w)
            base_exp = _rd(base_exp)

            # 从最终值反推基值
            divisor = qdm * form * spd
            if divisor == 0:
                mismatches.append(f"speed={pt.get('speed','?')}: zero divisor")
                continue
            base_inf = rec_val / divisor
            base_inf = _rd(base_inf)

            err = abs(base_inf - base_exp)
            max_err = max(max_err, err)
            if err > Decimal("2.0"):
                mismatches.append(
                    f"speed={pt.get('speed','?')}: base_exp={base_exp}, base_inf={base_inf}, err={err}"
                )

        if mismatches:
            r.warn(f"{len(mismatches)} discrepancies: {mismatches[:5]}")
        else:
            r.ok(f"all {len(points)} points consistent (max err={max_err})")
        return r

    # ── 7. 多系数复合验算 ──────────────────────────────────

    def check_factor_compound(self) -> CheckResult:
        r = CheckResult("factor_compound")
        points = self._get_speed_points()
        if not points:
            r.warn("no speed-data points")
            return r

        max_err = Decimal(0)
        err_count = 0
        details = []
        for pt in points:
            try:
                min_g = _d(pt["min_glue"])
                max_g = _d(pt["max_glue"])
                min_w = _d(pt["min_weight"])
                max_w = _d(pt["max_weight"])
                cur_w = _d(pt["current_glue_weight"])
                qdm = _d(pt.get("qdm_factor", 1))
                form = _d(pt.get("ui_factor", 1))
                spd = _d(pt.get("speed_factor", 1))
                rec_val = _d(pt["value"])
                offset = _d(pt.get("offset", 0))
                sp = pt.get("speed", "?")
            except (KeyError, ValueError) as e:
                continue

            if max_w <= min_w:
                continue

            # CalGlueGap 基值
            base = min_g + (max_g - min_g) * (cur_w - min_w) / (max_w - min_w)
            # setValue = base * qdm * form * speed
            recalc_val = base * qdm * form * spd
            # offset (for SF: 代码中 line 1588-1589; for GU: 代码中 line 950-951)
            recalc_val += offset
            recalc_val = _rd(recalc_val)

            err = abs(recalc_val - rec_val)
            max_err = max(max_err, err)
            if err > Decimal("0.015"):
                err_count += 1
                details.append(
                    f"speed={sp}: calc={recalc_val}, log={rec_val}, err={err}"
                )

        if err_count == 0:
            r.ok(f"all {len(points)} points match (max err={max_err})")
        else:
            r.warn(
                f"{err_count}/{len(points)} mismatches (max err={max_err}): "
                + "; ".join(details[:5])
            )
        return r

    # ── 8. 写设备一致性 ──────────────────────────────────

    def check_write_consistency(self) -> CheckResult:
        r = CheckResult("write_consistency")
        fs = self.cfg.get("form_set", {})
        if not fs:
            r.ok("no form_set config")
            return r

        # 函数 → 可能写的位置列表
        drv_cfg_map = {
            "SetGlueGu": [
                ("GU1", "GU_1st_IsOn"),
                ("GU2", "GU_2nd_IsOn"),
                ("GU3", "GU_3rd_IsOn"),
            ],
            "SetGlueSF1": [("SF1", "SF1_IsOn")],
            "SetGlueSF2": [("SF2", "SF2_IsOn")],
            "SetGlueSF3": [("SF3", "SF3_IsOn")],
        }
        subjects = drv_cfg_map.get(self.func, [])
        if not subjects:
            r.ok(f"no write consistency check for {self.func}")
            return r

        actual = list(self.set_values.keys())
        issues = []
        for drv_name, cfg_key in subjects:
            cfg_on = fs.get(cfg_key, False)
            has_data = drv_name in actual
            if has_data and not cfg_on:
                issues.append(f"{drv_name} has data but IsOn=false")
            elif not has_data and cfg_on:
                issues.append(f"{drv_name} IsOn=true but no data in record")
        if issues:
            r.warn("; ".join(issues))
        else:
            r.ok("config consistent with data")
        return r

    # ── 9. 异常/取消追踪 ──────────────────────────────────

    def check_cancel_patterns(self) -> CheckResult:
        r = CheckResult("cancel_patterns")
        ts = self.ts_str.strip()[:19] if self.ts_str.strip() else "?"
        # 单条记录无法判断取消模式，记录时间戳供后续关联分析
        r.ok(f"record exec at {ts}")
        return r

    # ── 辅助：提取速度点数据 ─────────────────────────────

    def _get_speed_points(self) -> list[dict]:
        """将 set_values 下的各驱动位表格展平为 dict 列表。"""
        points = []
        for pos_key, pos_data in self.set_values.items():
            if not isinstance(pos_data, dict):
                continue
            cols = pos_data.get("columns", [])
            rows = pos_data.get("data", [])
            for row in rows:
                if len(row) != len(cols):
                    continue
                pt = dict(zip(cols, row))
                pt["_position"] = pos_key
                points.append(pt)
        return points

    # ── 批量执行 ────────────────────────────────────────

    def diagnose_all(self) -> list[CheckResult]:
        self.results = [
            self.check_lifecycle(),
            self.check_timing(),
            self.check_auto_delay(),
            self.check_material_parsing(),
            self.check_driver_mapping(),
            self.check_base_calculation(),
            self.check_factor_compound(),
            self.check_write_consistency(),
            self.check_cancel_patterns(),
        ]
        return self.results


# ---------------------------------------------------------------------------
# 批量诊断入口
# ---------------------------------------------------------------------------


def diagnose_records(records: list[dict], config: Optional[dict] = None) -> list[dict]:
    """对批次记录逐条诊断，返回序列化结果列表。"""
    output = []
    for i, rec in enumerate(records):
        diag = GlueRecordDiagnostic(rec, config)
        results = diag.diagnose_all()
        errs = sum(1 for r in results if r.status == "ERROR")
        warns = sum(1 for r in results if r.status == "WARN")
        passed = sum(1 for r in results if r.status == "PASS")
        output.append(
            {
                "index": i,
                "func": rec.get("func"),
                "part": rec.get("part"),
                "material": rec.get("material"),
                "time": rec.get("time", ""),
                "results": [
                    {"name": r.name, "status": r.status, "detail": r.detail}
                    for r in results
                ],
                "summary": {
                    "errors": errs,
                    "warnings": warns,
                    "passed": passed,
                    "total": len(results),
                },
            }
        )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GlueCtrl 胶水间隙诊断分析工具")
    parser.add_argument("--records", required=True, help="Records JSON 文件路径")
    parser.add_argument("--config", help="Config JSON 文件路径 (可选)")
    parser.add_argument("--output", help="输出 JSON 报告路径 (可选)")
    args = parser.parse_args()

    with open(args.records, "r", encoding="utf-8") as f:
        records = json.load(f)

    config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    results = diagnose_records(records, config)

    total_errs = sum(r["summary"]["errors"] for r in results)
    total_warns = sum(r["summary"]["warnings"] for r in results)
    total_ok = sum(r["summary"]["passed"] for r in results)

    # 控制台输出
    print("=" * 64)
    for rec in results:
        print(
            f"\n=== Record #{rec['index'] + 1} | {rec['func']} | {rec['material']} | {rec.get('time', '')[:19]} ==="
        )
        for r in rec["results"]:
            icon = {"PASS": "[PASS]", "WARN": "[WARN]", "ERROR": "[ERR!]"}[r["status"]]
            print(f"  {icon} {r['name']}: {r['detail']}")
        s = rec["summary"]
        print(
            f"  -> {s['errors']} errors, {s['warnings']} warnings, {s['passed']} passed"
        )

    print(f"\n{'=' * 64}")
    print(
        f"Overall: {len(results)} records, {total_errs} errors, {total_warns} warnings, {total_ok} passed"
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
