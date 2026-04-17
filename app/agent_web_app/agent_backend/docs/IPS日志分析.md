# IPS日志分析

## 换材换卷

SF:

1. I7 -> {"Module":"换材判定模块","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{handle_func_name}--已经进入换材准备中状态了，用下批理论材质进行赋值操作\r\n当前：材质={material}，门幅={width}，楞型={flute_type}\r\n下批理论：材质={next_material}，门幅={next_width}，楞型={next_flute_type}\r\n","ExceptionInfo":null}

2. I8 -> {"Module":"换材判定模块","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{handle_func_name} 准备发送{splicer_part}换材消息(正常换卷换材)，通知各执行类","ExceptionInfo":null}

DF: 

1. I11 -> {"Module":"换材判定模块","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"DFChangePaper--糊机判定为换材\r\n上次使用的：材质={prev_material}，楞型={prev_flute_type}，门幅={prev_width}\r\n即将使用的：材质={material}，楞型={flute_type}，门幅={width};下批材质={next_material}\r\n准备进入HandleGuChangePaper具体执行函数\r\n","ExceptionInfo":null}

2. I12 -> {"Module":"换材判定模块","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{handle_func_name} 已发送糊机换材消息，通知各执行类","ExceptionInfo":null}

### 数据拿取结果：

换材事件会按照时间排序

```
[
   {
      "part":"ls0",
      "msg":"(-,0,-) -> (D,2500,5EB)",
      "time":"2026-01-10 14:06:23.690000"
   },
   {
      "part":"ls2",
      "msg":"(D,2500,B) -> (A,2400,B)",
      "time":"2026-01-10 14:11:11.443000"
   },
   {
      "part":"ms1",
      "msg":"(9,2500,E) -> (8,2400,E)",
      "time":"2026-01-10 14:11:14.190000"
   },
   {
      "part":"df",
      "msg":"(D.9.9.8.D,2400,5EB) -> (N.8.9.8.A,2400,5EB)",
      "time":"2026-01-10 14:11:31.983000"
   },
   {
      "part":"ls0",
      "msg":"(D,2500,5EB) -> (N,2400,5EB)",
      "time":"2026-01-10 14:11:38.800000"
   },
   {
      "part":"ms1",
      "msg":"(8,2400,E) -> (0,2400,E)",
      "time":"2026-01-10 14:25:41.477000"
   },
   {
      "part":"ls1",
      "msg":"(9,2500,E) -> (2,2400,E)",
      "time":"2026-01-10 14:25:46.800000"
   },
   {
      "part":"df",
      "msg":"(N.8.9.8.A,2400,5EB) -> (D.0.2.8.A,2400,5EB)",
      "time":"2026-01-10 14:29:45.283000"
   },
   {
      "part":"ls0",
      "msg":"(N,2400,5EB) -> (D,2400,5EB)",
      "time":"2026-01-10 14:29:53.880000"
   },
   {
      "part":"ls2",
      "msg":"(A,2400,B) -> (L,2400,B)",
      "time":"2026-01-10 14:39:27.710000"
   }
]
```

## 控制

### 糊间隙

#### SF

1. G4 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{glue_part}糊间隙计算结果：车速值={speed1};最小糊间隙={min_glue1};最大糊间隙={max_glue1};最小克重={min_weight1};最大克重={max_weight1};当前胶水克重={current_glue_weight1};车速系数={speed_factor1};车速限制的最小值={min_speed1};QDM系数={qdm_factor1};界面系数={ui_factor1};偏移量={offset1};计算结果：{value1}\r\n{glue_part}糊间隙计算结果：车速值={speed2};最小糊间隙={min_glue2};最大糊间隙={max_glue2};最小克重={min_weight2};最大克重={max_weight2};当前胶水克重={current_glue_weight2};车速系数={speed_factor2};车速限制的最小值={min_speed2};QDM系数={qdm_factor2};界面系数={ui_factor2};偏移量={offset2};计算结果：{value2}\r\n{glue_part}糊间隙计算结果：车速值={speed3};最小糊间隙={min_glue3};最大糊间隙={max_glue3};最小克重={min_weight3};最大克重={max_weight3};当前胶水克重={current_glue_weight3};车速系数={speed_factor3};车速限制的最小值={min_speed3};QDM系数={qdm_factor3};界面系数={ui_factor3};偏移量={offset3};计算结果：{value3}\r\n{glue_part}糊间隙计算结果：车速值={speed4};最小糊间隙={min_glue4};最大糊间隙={max_glue4};最小克重={min_weight4};最大克重={max_weight4};当前胶水克重={current_glue_weight4};车速系数={speed_factor4};车速限制的最小值={min_speed4};QDM系数={qdm_factor4};界面系数={ui_factor4};偏移量={offset4};计算结果：{value4}\r\n{glue_part}糊间隙计算结果：车速值={speed5};最小糊间隙={min_glue5};最大糊间隙={max_glue5};最小克重={min_weight5};最大克重={max_weight5};当前胶水克重={current_glue_weight5};车速系数={speed_factor5};车速限制的最小值={min_speed5};QDM系数={qdm_factor5};界面系数={ui_factor5};偏移量={offset5};计算结果：{value5}\r\n{glue_part}糊间隙计算结果：车速值={speed6};最小糊间隙={min_glue6};最大糊间隙={max_glue6};最小克重={min_weight6};最大克重={max_weight6};当前胶水克重={current_glue_weight6};车速系数={speed_factor6};车速限制的最小值={min_speed6};QDM系数={qdm_factor6};界面系数={ui_factor6};偏移量={offset6};计算结果：{value6}\r\n{glue_part}糊间隙计算结果：车速值={speed7};最小糊间隙={min_glue7};最大糊间隙={max_glue7};最小克重={min_weight7};最大克重={max_weight7};当前胶水克重={current_glue_weight7};车速系数={speed_factor7};车速限制的最小值={min_speed7};QDM系数={qdm_factor7};界面系数={ui_factor7};偏移量={offset7};计算结果：{value7}\r\n{glue_part}糊间隙计算结果：车速值={speed8};最小糊间隙={min_glue8};最大糊间隙={max_glue8};最小克重={min_weight8};最大克重={max_weight8};当前胶水克重={current_glue_weight8};车速系数={speed_factor8};车速限制的最小值={min_speed8};QDM系数={qdm_factor8};界面系数={ui_factor8};偏移量={offset8};计算结果：{value8}\r\n","ExceptionInfo":null}

2. G5 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--{glue_part}糊间隙往设备写值动作完成,材质={material},楞型={flute_type}","ExceptionInfo":null}

#### DF

1. G14 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{glue_part}糊间隙计算结果：车速值={speed1};最小糊间隙={min_glue1};最大糊间隙={max_glue1};最小克重={min_weight1};最大克重={max_weight1};当前胶水克重={current_glue_weight1};车速系数={speed_factor1};车速限制的最小值={min_speed1};QDM系数={qdm_factor1};界面系数={ui_factor1};计算结果：{value1}\r\n{glue_part}糊间隙计算结果：车速值={speed2};最小糊间隙={min_glue2};最大糊间隙={max_glue2};最小克重={min_weight1};最大克重={max_weight1};当前胶水克重={current_glue_weight2};车速系数={speed_factor2};车速限制的最小值={min_speed2};QDM系数={qdm_factor2};界面系数={ui_factor2};计算结果：{value2}\r\n{glue_part}糊间隙计算结果：车速值={speed3};最小糊间隙={min_glue3};最大糊间隙={max_glue3};最小克重={min_weight3};最大克重={max_weight3};当前胶水克重={current_glue_weight3};车速系数={speed_factor3};车速限制的最小值={min_speed3};QDM系数={qdm_factor3};界面系数={ui_factor3};计算结果：{value3}\r\n{glue_part}糊间隙计算结果：车速值={speed4};最小糊间隙={min_glue4};最大糊间隙={max_glue4};最小克重={min_weight4};最大克重={max_weight4};当前胶水克重={current_glue_weight4};车速系数={speed_factor4};车速限制的最小值={min_speed4};QDM系数={qdm_factor4};界面系数={ui_factor4};计算结果：{value4}\r\n{glue_part}糊间隙计算结果：车速值={speed5};最小糊间隙={min_glue5};最大糊间隙={max_glue5};最小克重={min_weight5};最大克重={max_weight5};当前胶水克重={current_glue_weight5};车速系数={speed_factor5};车速限制的最小值={min_speed5};QDM系数={qdm_factor5};界面系数={ui_factor5};计算结果：{value5}\r\n{glue_part}糊间隙计算结果：车速值={speed6};最小糊间隙={min_glue6};最大糊间隙={max_glue6};最小克重={min_weight6};最大克重={max_weight6};当前胶水克重={current_glue_weight6};车速系数={speed_factor6};车速限制的最小值={min_speed6};QDM系数={qdm_factor6};界面系数={ui_factor6};计算结果：{value6}\r\n{glue_part}糊间隙计算结果：车速值={speed7};最小糊间隙={min_glue7};最大糊间隙={max_glue7};最小克重={min_weight7};最大克重={max_weight7};当前胶水克重={current_glue_weight7};车速系数={speed_factor7};车速限制的最小值={min_speed7};QDM系数={qdm_factor7};界面系数={ui_factor7};计算结果：{value7}\r\n{glue_part}糊间隙计算结果：车速值={speed8};最小糊间隙={min_glue8};最大糊间隙={max_glue8};最小克重={min_weight8};最大克重={max_weight8};当前胶水克重={current_glue_weight8};车速系数={speed_factor8};车速限制的最小值={min_speed8};QDM系数={qdm_factor8};界面系数={ui_factor8};计算结果：{value8}\r\n","ExceptionInfo":null}

2. G12 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--糊机糊间隙设备写值完成,材质={material},楞型={flute_type}","ExceptionInfo":null}

#### 糊间隙分析流

1. 单面机换材事件 I7 -> I8
2. 双面机换材事件 I11 -> I12
3. 单面机糊间隙赋值事件 G4 -> G5
4. 双面机糊间隙赋值事件 G14 -> G12

记录每一次换材事件，当发生糊间隙赋值事件时，根据时间拿取换材事件作为材质生命周期，再根据用户的理想材质分析材质错位情况。

#### 糊间隙数据结果

```
[
   {
      "func":"SetGlueSF2",
      "part":"SF2",
      "material":"8/J",
      "flute_type":"B",
      "set_values":{
         "SF2":{
            "columns":[
               "speed",
               "min_glue",
               "max_glue",
               "min_weight",
               "max_weight",
               "current_glue_weight",
               "speed_factor",
               "min_speed",
               "qdm_factor",
               "ui_factor",
               "offset",
               "value"
            ],
            "data":[
               [
                  "30",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.80",
                  "30",
                  "1.07",
                  "1.10",
                  "0",
                  "39.72"
               ],
               [
                  "60",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.60",
                  "25",
                  "1.07",
                  "1.10",
                  "0",
                  "35.31"
               ],
               [
                  "90",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.40",
                  "20",
                  "1.07",
                  "1.10",
                  "0",
                  "30.90"
               ],
               [
                  "120",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.20",
                  "17",
                  "1.07",
                  "1.10",
                  "0",
                  "26.48"
               ],
               [
                  "140",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.10",
                  "15",
                  "1.07",
                  "1.10",
                  "0",
                  "24.28"
               ],
               [
                  "200",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "1.00",
                  "10",
                  "1.07",
                  "1.10",
                  "0",
                  "22.07"
               ],
               [
                  "240",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "0.90",
                  "10",
                  "1.07",
                  "1.10",
                  "0",
                  "19.86"
               ],
               [
                  "260",
                  "10",
                  "35",
                  "200",
                  "400",
                  "270",
                  "0.85",
                  "10",
                  "1.07",
                  "1.10",
                  "0",
                  "18.76"
               ]
            ]
         }
      },
      "time":"2026-01-08 14:22:55.607000",
      "lifecycle":{
         "ms2":{
            "msg":"(-,0,-) -> (8,2400,B)",
            "time":"2026-01-08 14:22:42.530000"
         },
         "ls2":{
            "msg":"(-,0,-) -> (J,2400,B)",
            "time":"2026-01-08 14:22:35.730000"
         },
         "set_func":{
            "name":"SetGlueSF2",
            "time":"2026-01-08 14:22:55.607000"
         }
      }
   }
]
```

### 热喷雾

1. H1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--{part}热喷雾,正常换材,材质={material},楞型={flute_type}","ExceptionInfo":null}
2. H2 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandleChangeNow--{part}热喷雾,立刻赋值，材质={material},楞型={flute_type},偏移量={offset}","ExceptionInfo":null}
3. H3 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} 准备点位赋值，材质={material},楞型={flute_type}","ExceptionInfo":null}
4. H4 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}点位赋值完成，材质={material},楞型={flute_type},设定值={set_value},基础值={base_value}","ExceptionInfo":null}
5. **目前缺失更改确认事件**

补充确认事件，
- {set_func_name}点位赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串


#### 热喷雾分析流

H1或H2 -> H3 -> H4

统计上述流程，根据用户的理想材质分析材质错位情况


### 天桥张力

1. S1 -> {"Module":"module","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {handle_func},本次为正常换材,材质={material},楞型={flute_type}","ExceptionInfo":null}
2. S2 -> {"Module":"module","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 SetBridgeTension ，材质={material},楞型={flute_type}","ExceptionInfo":null}
3. **缺失更改确认事件**

补充确认事件，
- SetBridgeTension赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串

#### 天桥张力分析流

S1 -> S3

### 冷板压力

1. C1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--冷板压力赋值，本次为正常换材,材质={material},楞型={flute_type},门幅={width}","ExceptionInfo":null}
2. C2->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 SetColdPlatePress ，材质={material},楞型={flute_type}","ExceptionInfo":null}
3. **缺失更改确认事件**

补充确认事件，
- SetColdPlatePress赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串

#### 冷板压力分析流

C1 -> C2

### 瓦楞辊

1. CR1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--{part}，本次为正常换材，材质={material},楞型={flute_type}","ExceptionInfo":null}
2. CR2 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} 准备点位赋值，芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}
3. CR3 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{part}瓦楞辊设定值 操作侧={os_set_value},驱动侧={ds_set_value}","ExceptionInfo":null}
4. CR4 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--赋值完成.芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}

#### 瓦楞辊分析流

CR1 -> CR2 -> CR3 -> CR4

### 压板组数

1. hlg1->{"Module":"{module}","Ip":"{ip}","Host":"host","UserName":{username},"Content":"HandlePubMsg--压板组数赋值,本次为正常换材,材质={material},楞型={flute_type},门幅={width}","ExceptionInfo":null}
2. hlg2->{"Module":"{module}","Ip":"{ip}","Host":"host","UserName":{username},"Content":"进入 SetPressGroupQty ,材质={material},楞型={flute_type},门幅={width}","ExceptionInfo":null}
3. hlg3->{"Module":"{module}","Ip":"{ip}","Host":"host","UserName":{username},"Content":"压板组数赋值任务取消,因为该期间内又收到一个新的压板组数赋值任务","ExceptionInfo":null}
4. **缺失更改确认事件**

补充确认事件，
- SetPressGroupQty赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串

#### 压板组数分析流

hlg1 -> hlg2 -> hlg3 -> 确认事件

### 热板压力

1. hpp1->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--热板压力赋值，本次为正常换材,材质={material},楞型={flute_type},门幅={width}","ExceptionInfo":null}	
2. hpp2->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 SetHotPlatePress ，材质={material},楞型={flute_type}","ExceptionInfo":null}	
3. hpp3->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"热板压力赋值任务取消,因为该期间内又收到一个新的热板压力赋值任务","ExceptionInfo":null}	
4. **缺失更改确认事件**

补充确认事件，
- SetHotPlatePress赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串

#### 热板压力分析流

hpp1 -> hpp2 -> hpp3 -> 确认事件

### MP压力辊(完整)

1. MP1->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--{part}，本次为正常换材，材质={material},楞型={flute_type}","ExceptionInfo":null}	
2. MP2->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} 准备点位赋值，芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}	
3. MP3->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--产生了一个新的任务，该任务终止.芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}	
4. MP4->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--基础值计算完毕.芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type},基础值={base_value}","ExceptionInfo":null}	

#### MP压力辊分析流

MP1 -> MP2 -> MP3 -> MP4

#### MP压力辊结果

```
[
   {
      "func":"SetPressRollSF2",
      "ms_material":"8",
      "ls_material":"D",
      "width":"2400",
      "flute_type":"B",
      "base_value":"18",
      "time":"2026-01-10 14:06:16.107000"
   },
   {
      "func":"SetPressRollSF1",
      "ms_material":"9",
      "ls_material":"9",
      "width":"2400",
      "flute_type":"E",
      "base_value":"16",
      "time":"2026-01-10 14:06:19.817000"
   },
   {
      "func":"SetPressRollSF2",
      "ms_material":"8",
      "ls_material":"A",
      "width":"2400",
      "flute_type":"B",
      "base_value":"17",
      "time":"2026-01-10 14:11:24.030000"
   }
]
```

### 接纸机张力(完整)

1. spt1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{handle_func_name}--{part}接纸机张力,正常换材赋值，材质={material},门幅={width}","ExceptionInfo":null}	
2. spt2 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} ,材质={material},门幅={width},立刻赋值={is_set_now}","ExceptionInfo":null}	
3. spt3 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--执行完成,材质={material},门幅={width},立刻赋值={is_set_now}","ExceptionInfo":null}		

#### 接纸机张力分析流

spt1 -> spt2 -> spt3

### 真空泵(完整)

1. vb1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--{part}，本次为正常换材，材质={material},楞型={flute_type}","ExceptionInfo":null}	
2. vb2 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--产生了一个新的任务，该任务终止.芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}	
3. vb3 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} 准备点位赋值，芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}	
4. vb4 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandleChangeNow {part}真空泵立即赋值，材质={material},楞型={flute_type},偏移量={offset}","ExceptionInfo":null}
5. vb5 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{set_func_name}--{part}真空泵 基础值*门幅系数={base_value_multiply_width_factor_result}.芯纸材质={ms_material}，里纸材质={ls_material},门幅={width},楞型={flute_type}","ExceptionInfo":null}	

#### 真空泵分析流

```
vb1 -> vb2 -> vb3 -> vb5
        ^
        |
       vb4
```

#### 真空泵数据结果

```
[
   {
      "func":"SetVacuumBlowerSF1",
      "part":"SF1",
      "base_value_multiply_width_factor_result":"0",
      "ms_material":"0",
      "ls_material":"2",
      "width":"2800",
      "flute_type":"E",
      "time":"2026-01-09 06:43:25.643000",
      "lifecycle":{
         "ms1":{
            "msg":"",
            "time":""
         },
         "ls1":{
            "msg":"",
            "time":""
         },
         "set_func":{
            "name":"SetVacuumBlowerSF1",
            "time":"2026-01-09 06:43:25.643000"
         }
      }
   },
   {
      "func":"SetVacuumBlowerSF2",
      "part":"SF2",
      "base_value_multiply_width_factor_result":"0",
      "ms_material":"8",
      "ls_material":"A",
      "width":"2800",
      "flute_type":"B",
      "time":"2026-01-09 06:48:06.233000",
      "lifecycle":{
         "ms2":{
            "msg":"(0,2300,B) -> (0,2300,B)",
            "time":"2026-01-08 15:32:35.953000"
         },
         "ls2":{
            "msg":"(A,2300,B) -> (L,2250,B)",
            "time":"2026-01-08 15:39:32.567000"
         },
         "set_func":{
            "name":"SetVacuumBlowerSF2",
            "time":"2026-01-09 06:48:06.233000"
         }
      }
   },
   {
      "func":"SetVacuumBlowerSF2",
      "part":"SF2",
      "base_value_multiply_width_factor_result":"0",
      "ms_material":"8",
      "ls_material":"A",
      "width":"2800",
      "flute_type":"B",
      "time":"2026-01-09 07:24:23.477000",
      "lifecycle":{
         "ms2":{
            "msg":"(0,2300,B) -> (0,2300,B)",
            "time":"2026-01-08 15:32:35.953000"
         },
         "ls2":{
            "msg":"(A,2300,B) -> (L,2250,B)",
            "time":"2026-01-08 15:39:32.567000"
         },
         "set_func":{
            "name":"SetVacuumBlowerSF2",
            "time":"2026-01-09 07:24:23.477000"
         }
      }
   },
]
```

### 包角(缺失)

1. w1 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{handle_func_name} ，本次为正常换材，材质={material},楞型={flute_type}","ExceptionInfo":null}	
2. w2 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入{set_func_name}准备点位赋值，材质={material},楞型={flute_type}","ExceptionInfo":null}		
3. w3 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"{machine_id}机芯纸包角赋值任务终止，因为此时新开了一个赋值任务线程！","ExceptionInfo":null}	
4. w4 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"糊机包角赋值任务取消,因为该期间内又收到一个新的糊机包角赋值任务","ExceptionInfo":null}	
5. w5 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"糊机包角设备部位和材质匹配失败！\r\n用户勾选的糊机糊间隙设备部位:\r\n下层={bottom_layer}\r\n中层={mid_layer}\r\n下层={top_layer}\r\n当前包角材质情况：1层={layer1_material}；2层={layer2_material}；3层={layer3_material}\r\n","ExceptionInfo":null}	
6. w6 -> {"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"用户勾选的糊机包角使用部位和材质匹配对应不上，使用默认情况处理","ExceptionInfo":null}	
7. **缺失更改确认事件**

w5中上中下层日志打印出错

补充确认事件，
- {set_func_name}赋值完成，材质={material},楞型={flute_type},写入{params}
- 将设置的params对象转化为json字符串

#### 包角分析流

w1 -> w2 -> 确认事件

### 骑辊(完整)

1. rr1->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"HandlePubMsg--骑辊赋值,正常换材,材质={material},楞型={flute_type},门幅={width}","ExceptionInfo":null}	
2. rr2->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"进入 {set_func_name} ，材质={material},楞型={flute_type}","ExceptionInfo":null}	
3. rr3->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"骑辊--糊机设备部位和材质匹配失败！\r\n用户勾选的糊机糊间隙设备部位:\r\n下层={bottom_layer}\r\n中层={mid_layer}\r\n下层={top_layer}\r\n当前骑辊材质情况：1层={layer1_material}；2层={layer2_material}；3层={layer3_material}\r\n","ExceptionInfo":null}	
4. rr4->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"骑辊--用户勾选的糊机糊间隙使用部位和材质匹配对应不上，使用默认情况处理","ExceptionInfo":null}	
5. rr5->{"Module":"{module}","Ip":"{ip}","Host":"{host}","UserName":{username},"Content":"骑辊{layer_id}层，设定值={set_value}，基础值={base_value_formulation}，系数={factor}","ExceptionInfo":null}

#### 骑辊分析流

rr1 -> rr2 -> rr5

## 应用搭建

根据用户问题，使用对应的分析模块。分析模块会按上述日志事件，整合换材、设置方法、外部调整（根据车速调整、根据弯翘调整）事件，对某一时间段的机器部位进行分析，最终的原始事件会包含**材质生命周期**、**设置方法**、**车速调整**、**弯翘调整**四个部分，再通过与用户理想材质进行比对分析得出结论。

Agent的任务为，将用户问题转换为对应的IPS分析方法以及其需要的各个参数的填写，除此之外不负责任何任务。分析方法的调用会在前端对用户进行反馈。

暂输出类似如下markdown格式的文本：

```
# 📊 函数调用记录

## 📋 基本信息

| 字段 | 值 |
|------|-----|
| **函数** | `SetGlueSF2` |
| **部件** | `SF2` |
| **材料** | `8/J` |
| **瓦楞类型** | `B` |
| **时间** | `2026-01-08 14:22:55.607000` |

## 🔄 生命周期

| 阶段 | 信息 | 时间 |
|------|------|------|
| **MS2** | `(-,0,-) -> (8,2400,B)` | 2026-01-08 14:22:42.530000 |
| **LS2** | `(-,0,-) -> (J,2400,B)` | 2026-01-08 14:22:35.730000 |
| **Set Function** | `SetGlueSF2` | 2026-01-08 14:22:55.607000 |

## ⚙️ 设置值

### 部位: SF2
| Speed | Min Glue | Max Glue | Min Weight | Max Weight | Current Glue Weight | Speed Factor | Min Speed | Qdm Factor | Ui Factor | Offset | Value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 10 | 35 | 200 | 400 | 270 | 1.80 | 30 | 1.07 | 1.10 | 0 | 39.72 |
| 60 | 10 | 35 | 200 | 400 | 270 | 1.60 | 25 | 1.07 | 1.10 | 0 | 35.31 |
| 90 | 10 | 35 | 200 | 400 | 270 | 1.40 | 20 | 1.07 | 1.10 | 0 | 30.90 |
| 120 | 10 | 35 | 200 | 400 | 270 | 1.20 | 17 | 1.07 | 1.10 | 0 | 26.48 |
| 140 | 10 | 35 | 200 | 400 | 270 | 1.10 | 15 | 1.07 | 1.10 | 0 | 24.28 |
| 200 | 10 | 35 | 200 | 400 | 270 | 1.00 | 10 | 1.07 | 1.10 | 0 | 22.07 |
| 240 | 10 | 35 | 200 | 400 | 270 | 0.90 | 10 | 1.07 | 1.10 | 0 | 19.86 |
| 260 | 10 | 35 | 200 | 400 | 270 | 0.85 | 10 | 1.07 | 1.10 | 0 | 18.76 |



 --- 
# 📊 函数调用记录

## 📋 基本信息

| 字段 | 值 |
|------|-----|
| **函数** | `SetGlueGu` |
| **部件** | `DF` |
| **材料** | `P.-.-.8.J` |
| **瓦楞类型** | `3B` |
| **时间** | `2026-01-08 14:23:27.490000` |

## 🔄 生命周期

| 阶段 | 信息 | 时间 |
|------|------|------|
| **LS0** | `(-,0,-) -> (P,2400,3B)` | 2026-01-08 14:23:07.127000 |
| **MS1** | `` |  |
| **LS1** | `` |  |
| **MS2** | `(-,0,-) -> (8,2400,B)` | 2026-01-08 14:22:42.530000 |
| **LS2** | `(-,0,-) -> (J,2400,B)` | 2026-01-08 14:22:35.730000 |
| **DF** | `(-.-.-.-.-,0,-) -> (P.-.-.8.J,2350,3B)` | 2026-01-08 14:23:03.613000 |
| **Set Function** | `SetGlueGu` | 2026-01-08 14:23:27.490000 |

## ⚙️ 设置值

### 部位: GU2
| Speed | Min Glue | Max Glue | Min Weight | Max Weight | Current Glue Weight | Speed Factor | Min Speed | Qdm Factor | Ui Factor | Value |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 10 | 35 | 200 | 500 | 290 | 1.80 | 30 | 1.00 | 1.15 | 36.22 |
| 60 | 10 | 35 | 200 | 500 | 290 | 1.60 | 25 | 1.00 | 1.15 | 32.20 |
| 90 | 10 | 35 | 200 | 500 | 290 | 1.40 | 20 | 1.00 | 1.15 | 28.17 |
| 120 | 10 | 35 | 200 | 500 | 290 | 1.20 | 17 | 1.00 | 1.15 | 24.15 |
| 140 | 10 | 35 | 200 | 500 | 290 | 1.10 | 15 | 1.00 | 1.15 | 22.14 |
| 200 | 10 | 35 | 200 | 500 | 290 | 1.00 | 10 | 1.00 | 1.15 | 20.12 |
| 240 | 10 | 35 | 200 | 500 | 290 | 0.90 | 10 | 1.00 | 1.15 | 18.11 |
| 260 | 10 | 35 | 200 | 500 | 290 | 0.85 | 10 | 1.00 | 1.15 | 17.11 |



 --- 
# 📊 函数调用记录

## 📋 基本信息

| 字段 | 值 |
|------|-----|
| **函数** | `SetGlueSF2` |
| **部件** | `SF2` |
| **材料** | `7/N` |
| **瓦楞类型** | `B` |
| **时间** | `2026-01-08 14:27:42.660000` |

## 🔄 生命周期

| 阶段 | 信息 | 时间 |
|------|------|------|
| **MS2** | `(8,2400,B) -> (7,2350,B)` | 2026-01-08 14:27:30.733000 |
| **LS2** | `(J,2400,B) -> (N,2350,B)` | 2026-01-08 14:27:33.097000 |
| **Set Function** | `SetGlueSF2` | 2026-01-08 14:27:42.660000 |

## ⚙️ 设置值

### 部位: SF2
| Speed | Min Glue | Max Glue | Min Weight | Max Weight | Current Glue Weight | Speed Factor | Min Speed | Qdm Factor | Ui Factor | Offset | Value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 10 | 35 | 200 | 400 | 280 | 1.80 | 30 | 0.80 | 1.10 | 0 | 31.68 |
| 60 | 10 | 35 | 200 | 400 | 280 | 1.60 | 25 | 0.80 | 1.10 | 0 | 28.16 |
| 90 | 10 | 35 | 200 | 400 | 280 | 1.40 | 20 | 0.80 | 1.10 | 0 | 24.64 |
| 120 | 10 | 35 | 200 | 400 | 280 | 1.20 | 17 | 0.80 | 1.10 | 0 | 21.12 |
| 140 | 10 | 35 | 200 | 400 | 280 | 1.10 | 15 | 0.80 | 1.10 | 0 | 19.36 |
| 200 | 10 | 35 | 200 | 400 | 280 | 1.00 | 10 | 0.80 | 1.10 | 0 | 17.60 |
| 240 | 10 | 35 | 200 | 400 | 280 | 0.90 | 10 | 0.80 | 1.10 | 0 | 15.84 |
| 260 | 10 | 35 | 200 | 400 | 280 | 0.85 | 10 | 0.80 | 1.10 | 0 | 14.96 |



 --- 
# 📊 函数调用记录

## 📋 基本信息

| 字段 | 值 |
|------|-----|
| **函数** | `SetGlueGu` |
| **部件** | `DF` |
| **材料** | `N.-.-.7.N` |
| **瓦楞类型** | `3B` |
| **时间** | `2026-01-08 14:28:19.307000` |

## 🔄 生命周期

| 阶段 | 信息 | 时间 |
|------|------|------|
| **LS0** | `(P,2400,3B) -> (N,2350,3B)` | 2026-01-08 14:28:06.293000 |
| **MS1** | `` |  |
| **LS1** | `` |  |
| **MS2** | `(8,2400,B) -> (7,2350,B)` | 2026-01-08 14:27:30.733000 |
| **LS2** | `(J,2400,B) -> (N,2350,B)` | 2026-01-08 14:27:33.097000 |
| **DF** | `(P.-.-.8.J,2350,3B) -> (N.-.-.7.N,2350,3B)` | 2026-01-08 14:27:58.600000 |
| **Set Function** | `SetGlueGu` | 2026-01-08 14:28:19.307000 |

## ⚙️ 设置值

### 部位: GU2
| Speed | Min Glue | Max Glue | Min Weight | Max Weight | Current Glue Weight | Speed Factor | Min Speed | Qdm Factor | Ui Factor | Value |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 10 | 35 | 200 | 500 | 280 | 1.80 | 30 | 1.00 | 1.15 | 34.50 |
| 60 | 10 | 35 | 200 | 500 | 280 | 1.60 | 25 | 1.00 | 1.15 | 30.67 |
| 90 | 10 | 35 | 200 | 500 | 280 | 1.40 | 20 | 1.00 | 1.15 | 26.83 |
| 120 | 10 | 35 | 200 | 500 | 280 | 1.20 | 17 | 1.00 | 1.15 | 23.00 |
| 140 | 10 | 35 | 200 | 500 | 280 | 1.10 | 15 | 1.00 | 1.15 | 21.08 |
| 200 | 10 | 35 | 200 | 500 | 280 | 1.00 | 10 | 1.00 | 1.15 | 19.17 |
| 240 | 10 | 35 | 200 | 500 | 280 | 0.90 | 10 | 1.00 | 1.15 | 17.25 |
| 260 | 10 | 35 | 200 | 500 | 280 | 0.85 | 10 | 1.00 | 1.15 | 16.29 |

 --- 

```

**后续增加材质错位判断分析和参数更改时间线**


## 目前问题

1. 需要补充控制类设置参数的完整流程
2. 各个控制类的日志比较零散
3. 车速和弯翘控制会直接影响最终写入的值

### 弯翘调整问题

1. 弯翘调整类会使用方法**WarpPublish**发布调整事件，订阅该事件的模块会存在handle方法
2. 在handle方法中会直接调用comm.WriteVar
3. 可在进行写值后进行log，打印**调整前数值**与**调整后数值**

### 车速调整问题

1. 每个点位的每次由于车速问题进行数值调整，需要在每次根据车速调整后进行日志的记录
2. 日志组成可以为，原先的IpsValueInfo与调整过后的IpsValueInfo，comm中的点位**原数值**与**调整后的数值**

