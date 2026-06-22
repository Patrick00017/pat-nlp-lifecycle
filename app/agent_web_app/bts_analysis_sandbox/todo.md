# 日志解析例子

- I16 -> {'module': '换材判定模块', 'ip': ' ', 'host': ' ', 'username': '" "', 'df_material': '9.9.9.9.9', 'df_width': '2500', 'df_flute': '5BA', 'ls0_material': '9', 'ls0_width': '2500', 'ls0_flute': '5BA', 'ls0_order_material': '9.9.9.9.9', 'ms1_material': '9', 'ms1_width': '2500', 'ms1_flute': 'B', 'ms1_order_material': '9.9.9.9.9', 'ls1_material': '9', 'ls1_width': '2500', 'ls1_flute': 'B', 'ls1_order_material': '9.9.9.9.9', 'ms2_material': '9', 'ms2_width': '2500', 'ms2_flute': 'A', 'ms2_order_material': '9.9.9.9.9', 'ls2_material': '9', 'ls2_width': '2500', 'ls2_flute': 'A', 'ls2_order_material': '9.9.9.9.9', 'extra': 'MS3--材质=-，门幅=0，楞型=，对应的订单材质=\\r\\nLS3--材质=-，门幅=0，楞型=，对应的订单材质=\\r\\n', 'exception': ' '}
- I11 -> {'module': '换材判定模块', 'ip': ' ', 'host': ' ', 'username': '" "', 'prev_material': '9.9.9.9.9', 'prev_flute_type': '5BA', 'prev_width': '2500', 'material': 'Q.5.Q.9.Q', 'flute_type': '5BA', 'width': '2500', 'next_material': 'Q.5.Q.9.Q', 'exception': ' '}
- I12 -> {'module': '换材判定模块', 'ip': ' ', 'host': ' ', 'username': '" "', 'handle_func_name': 'HandleGuChangePaper', 'exception': ' '}

# 弯翘模块事件模板 — TODO

- [x] 阅读 WarpCtrl.cs 源码，识别所有 logger.Info/Error 调用
- [x] 创建 warp_template.csv（WARP1~WARP10，不含 AddWarpStatus）
- [ ] 创建 warp_diagnostic.py
- [ ] 扩展 event_extractor.py：新增 WarpEventExtractor 或扩展 process() 支持 WARP 事件
- [ ] 创建 test_warp.py（含真实 DB 查询 + 模拟数据回退）
- [ ] 处理"收到弯翘检测设备传入信息"事件（WARP11 — 来源不在 source/ 中，需单独定义模板）

# 想法扩充

接纸机换材记录不能作为lifecycle事件加入到setfunc的事件中，需要单独的接纸机换材的数据结构记录，分发到各个状态机作为环境变量，也就是全局的变量。接纸机记录的数据结构需要生成对应的id，广播给各个状态机的是接纸机记录的id。这样后续可以查到换材的前后顺序，而不是只是显示换的那一次。QDM表以及基础信息表也是类似的记录方式。拿到赋值方法的时候，针对目前的环境信息，分析这次的赋值是否有问题
