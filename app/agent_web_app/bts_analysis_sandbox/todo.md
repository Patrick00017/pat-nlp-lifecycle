# 弯翘模块事件模板 — TODO

- [x] 阅读 WarpCtrl.cs 源码，识别所有 logger.Info/Error 调用
- [x] 创建 warp_template.csv（WARP1~WARP10，不含 AddWarpStatus）
- [ ] 创建 warp_diagnostic.py
- [ ] 扩展 event_extractor.py：新增 WarpEventExtractor 或扩展 process() 支持 WARP 事件
- [ ] 创建 test_warp.py（含真实 DB 查询 + 模拟数据回退）
- [ ] 处理"收到弯翘检测设备传入信息"事件（WARP11 — 来源不在 source/ 中，需单独定义模板）
