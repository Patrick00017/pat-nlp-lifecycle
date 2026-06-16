#region << 版 本 注 释 >>
/*----------------------------------------------------------------
 * 版权所有 (c) 2024  NJRN 保留所有权利。
 * CLR版本：4.0.30319.42000
 * 机器名称：BCCNSHGNB226
 * 公司名称：
 * 命名空间：BTS.Server.Start.IPSBizs.NewCtrl
 * 唯一标识：5250ed73-20d4-4e3e-a67e-f32dff36228a
 * 文件名：IPSMainCtrl
 * 当前用户域：BHS
 * 
 * 创建者：QZhou
 * 创建时间：2024/4/10 10:54:50
 * 版本：V1.0.0
 * 描述：IPS主控制类：
 * 1.实现各接纸机换材逻辑判断
 * 2.实现各单面机换材逻辑判断
 * 3.实现糊机换材逻辑判断
 * 4.判定换材后触发具体执行操作事件（发布者）
 * ----------------------------------------------------------------
 * 修改人：
 * 时间：
 * 修改说明：
 *
 * 版本：V1.0.1
 *----------------------------------------------------------------*/
#endregion << 版 本 注 释 >>

using System.Data;
using System.Text;
using BTS.Commons;
using BTS.Dtos;
using BTS.Entites;
using BTS.Logs;
using BTS.Server.Core;
using BTS.Server.Utils;
using BTS.Services;
using BTS.Services.Services.System;
using Newtonsoft.Json;


namespace BTS.Server
{
    /// <summary>
    /// IPS主控类
    /// </summary>
    public class IPSMainCtrl
    {
        #region <常量>
        private const string module = "换材判定模块";
        #endregion <常量>

        #region <变量>
        private Log logger;
        private DriverLink comm;
        //private TcpService tcpService;
        /// <summary>
        /// 瓦线是几层，从配置文件中读取
        /// </summary>
        private int floor = 5;
        private int begin = 0;//刚开始的时候是0，初始化结束之后是1
        private CancellationTokenSource cts;

        /// <summary>
        /// 糊机状态
        /// </summary>
        public DriveStateInfo stateInfo_gu = new DriveStateInfo();

        /// <summary>
        /// LS0状态
        /// </summary>
        public DriveStateInfo stateInfo_ls0 = new DriveStateInfo();

        /// <summary>
        /// MS1状态
        /// </summary>
        public DriveStateInfo stateInfo_ms1 = new DriveStateInfo();

        /// <summary>
        /// LS1状态
        /// </summary>
        public DriveStateInfo stateInfo_ls1 = new DriveStateInfo();

        /// <summary>
        /// MS2状态
        /// </summary>
        public DriveStateInfo stateInfo_ms2 = new DriveStateInfo();

        /// <summary>
        /// LS2状态
        /// </summary>
        public DriveStateInfo stateInfo_ls2 = new DriveStateInfo();

        /// <summary>
        /// MS3状态
        /// </summary>
        public DriveStateInfo stateInfo_ms3 = new DriveStateInfo();

        /// <summary>
        /// LS3状态
        /// </summary>
        public DriveStateInfo stateInfo_ls3 = new DriveStateInfo();

        /// <summary>
        /// 订单表业务类
        /// </summary>
        private OrderInfoManage oService = BLLFactory<OrderInfoManage>.Instance;
        /// <summary>
        /// 实材数据库业务类
        /// </summary>
        private SPPaperManager paperService = BLLFactory<SPPaperManager>.Instance;


        /// <summary>
        /// 糊机实材临时变量
        /// </summary>
        private GuRealInfo _temp_GU = new GuRealInfo();

        /// <summary>
        /// 接纸机实材临时变量
        /// </summary>
        private List<SPRealInfo> _temp_SPs = new List<SPRealInfo>
        {
            new SPRealInfo() { Name = "MS1", Code = "", Brand = "" },
            new SPRealInfo() { Name = "LS1", Code = "", Brand = "" },
            new SPRealInfo() { Name = "MS2", Code = "", Brand = "" },
            new SPRealInfo() { Name = "LS2", Code = "", Brand = "" },
            new SPRealInfo() { Name = "MS3", Code = "", Brand = "" },
            new SPRealInfo() { Name = "LS3", Code = "", Brand = "" }
        };
        #endregion <变量>

        #region <属性>
        #endregion <属性>

        #region <构造方法和析构方法>
        public IPSMainCtrl(DriverLink _comm)
        {
            logger = LogHelper.GetLogger(typeof(IPSMainCtrl));
            comm = _comm;
            AppConfig config = new AppConfig();
            floor = config.AppConfigGet("WaveLineLayer").ToInt16();
        }
        #endregion <构造方法和析构方法>

        #region <方法>
        /// <summary>
        /// 启动
        /// </summary>
        public void Start()
        {
            InitInfos();
            cts = new CancellationTokenSource();
            Task.Factory.StartNew(async () => { await SPChangePaper(); }, cts.Token);
            Task.Factory.StartNew(async () => { await DFChangePaper(); }, cts.Token);
            Task.Factory.StartNew(async () => { await SPChangePaperReady(); }, cts.Token);
            Task.Factory.StartNew(async () => { await MonitorRealPaper(); }, cts.Token);
            Task.Factory.StartNew(async () => { await SendIpsSetValue(); }, cts.Token);

            if (GlobalInfos.UseSpliceServer)
            {
                Task.Factory.StartNew(async () => { await SendSteamValue(); }, cts.Token);
            }

        }
        /// <summary>
        /// 终止
        /// </summary>
        public void Stop()
        {
            cts.Cancel();
        }

        /// <summary>
        /// 初始化state变量
        /// </summary>
        private void InitInfos()
        {
            try
            {
                var info = oService.GetFirstByWorkNo();
                stateInfo_gu.CurCode = info.WO_PaperCode;
                stateInfo_gu.CurFlute = info.WO_Wave;
                stateInfo_gu.CurWidth = info.WO_Width;
                stateInfo_gu.NextBachCode = info.WO_PaperCode;
                stateInfo_gu.CodeALl = info.WO_PaperCode;

                stateInfo_gu.LastCode = stateInfo_gu.CurCode;
                stateInfo_gu.LastFlute = stateInfo_gu.CurFlute;
                stateInfo_gu.LastWidth = stateInfo_gu.CurWidth;
                stateInfo_gu.BrandLS0 = "";
                stateInfo_gu.BrandLS1 = "";
                stateInfo_gu.BrandLS2 = "";
                stateInfo_gu.BrandLS3 = "";
                stateInfo_gu.BrandMS1 = "";
                stateInfo_gu.BrandMS2 = "";
                stateInfo_gu.BrandMS3 = "";

                #region 先按照当前首笔订单给各个机台赋值一遍
                List<string> everyPaper = new List<string>();
                if (info.WO_PaperCode.Contains("."))
                {
                    everyPaper = info.WO_PaperCode.Split('.').ToList();
                }
                else
                {
                    everyPaper = info.WO_PaperCode.ToCharArray().Select(it => it.ToString()).ToList();
                }
                int fluteIndex = 1;//楞型拆分到机台需要用到的索引下标
                for (int i = 0; i < everyPaper.Count; i++)
                {
                    switch (i)
                    {
                        case 0:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ls0.CurCode = everyPaper[i];
                                stateInfo_ls0.CurWidth = info.WO_Width;
                                stateInfo_ls0.CurFlute = info.WO_Wave;
                                stateInfo_ls0.CodeALl = info.WO_PaperCode;

                                stateInfo_ls0.LastWidth = stateInfo_ls0.CurWidth;
                                stateInfo_ls0.LastFlute = stateInfo_ls0.CurFlute;
                                stateInfo_ls0.LastCode = stateInfo_ls0.CurCode;

                                stateInfo_ls0.BrandLS0 = "";
                                stateInfo_ls0.BrandLS1 = "";
                                stateInfo_ls0.BrandLS2 = "";
                                stateInfo_ls0.BrandLS3 = "";
                                stateInfo_ls0.BrandMS1 = "";
                                stateInfo_ls0.BrandMS2 = "";
                                stateInfo_ls0.BrandMS3 = "";
                            }

                            break;
                        case 1:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ms1.CurCode = everyPaper[i];
                                stateInfo_ms1.CurWidth = info.WO_Width;
                                stateInfo_ms1.CurFlute = info.WO_Wave.Substring(fluteIndex, 1);
                                stateInfo_ms1.CodeALl = info.WO_PaperCode;

                                stateInfo_ms1.LastWidth = stateInfo_ms1.CurWidth;
                                stateInfo_ms1.LastFlute = stateInfo_ms1.CurFlute;
                                stateInfo_ms1.LastCode = stateInfo_ms1.CurCode;

                                stateInfo_ms1.BrandLS0 = "";
                                stateInfo_ms1.BrandLS1 = "";
                                stateInfo_ms1.BrandLS2 = "";
                                stateInfo_ms1.BrandLS3 = "";
                                stateInfo_ms1.BrandMS1 = "";
                                stateInfo_ms1.BrandMS2 = "";
                                stateInfo_ms1.BrandMS3 = "";

                                fluteIndex++;

                            }
                            break;
                        case 2:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ls1.CurCode = everyPaper[i];
                                stateInfo_ls1.CurWidth = info.WO_Width;
                                stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                stateInfo_ls1.CodeALl = info.WO_PaperCode;

                                stateInfo_ls1.LastWidth = stateInfo_ls1.CurWidth;
                                stateInfo_ls1.LastFlute = stateInfo_ls1.CurFlute;
                                stateInfo_ls1.LastCode = stateInfo_ls1.CurCode;

                                stateInfo_ls1.BrandLS0 = "";
                                stateInfo_ls1.BrandLS1 = "";
                                stateInfo_ls1.BrandLS2 = "";
                                stateInfo_ls1.BrandLS3 = "";
                                stateInfo_ls1.BrandMS1 = "";
                                stateInfo_ls1.BrandMS2 = "";
                                stateInfo_ls1.BrandMS3 = "";
                            }
                            break;
                        case 3:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ms2.CurCode = everyPaper[i];
                                stateInfo_ms2.CurWidth = info.WO_Width;
                                stateInfo_ms2.CurFlute = info.WO_Wave.Substring(fluteIndex, 1);
                                stateInfo_ms2.CodeALl = info.WO_PaperCode;

                                stateInfo_ms2.LastWidth = stateInfo_ms2.CurWidth;
                                stateInfo_ms2.LastFlute = stateInfo_ms2.CurFlute;
                                stateInfo_ms2.LastCode = stateInfo_ms2.CurCode;

                                stateInfo_ms2.BrandLS0 = "";
                                stateInfo_ms2.BrandLS1 = "";
                                stateInfo_ms2.BrandLS2 = "";
                                stateInfo_ms2.BrandLS3 = "";
                                stateInfo_ms2.BrandMS1 = "";
                                stateInfo_ms2.BrandMS2 = "";
                                stateInfo_ms2.BrandMS3 = "";

                                fluteIndex++;

                            }
                            break;
                        case 4:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ls2.CurCode = everyPaper[i];
                                stateInfo_ls2.CurWidth = info.WO_Width;
                                stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                stateInfo_ls2.CodeALl = info.WO_PaperCode;

                                stateInfo_ls2.LastWidth = stateInfo_ls2.CurWidth;
                                stateInfo_ls2.LastFlute = stateInfo_ls2.CurFlute;
                                stateInfo_ls2.LastCode = stateInfo_ls2.CurCode;

                                stateInfo_ls2.BrandLS0 = "";
                                stateInfo_ls2.BrandLS1 = "";
                                stateInfo_ls2.BrandLS2 = "";
                                stateInfo_ls2.BrandLS3 = "";
                                stateInfo_ls2.BrandMS1 = "";
                                stateInfo_ls2.BrandMS2 = "";
                                stateInfo_ls2.BrandMS3 = "";
                            }
                            break;
                        case 5:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ms3.CurCode = everyPaper[i];
                                stateInfo_ms3.CurWidth = info.WO_Width;
                                stateInfo_ms3.CurFlute = info.WO_Wave.Substring(fluteIndex, 1);
                                stateInfo_ms3.CodeALl = info.WO_PaperCode;

                                stateInfo_ms3.LastWidth = stateInfo_ms3.CurWidth;
                                stateInfo_ms3.LastFlute = stateInfo_ms3.CurFlute;
                                stateInfo_ms3.LastCode = stateInfo_ms3.CurCode;

                                stateInfo_ms3.BrandLS0 = "";
                                stateInfo_ms3.BrandLS1 = "";
                                stateInfo_ms3.BrandLS2 = "";
                                stateInfo_ms3.BrandLS3 = "";
                                stateInfo_ms3.BrandMS1 = "";
                                stateInfo_ms3.BrandMS2 = "";
                                stateInfo_ms3.BrandMS3 = "";

                                fluteIndex++;
                            }
                            break;
                        case 6:
                            if (everyPaper[i] != "-")
                            {
                                stateInfo_ls3.CurCode = everyPaper[i];
                                stateInfo_ls3.CurWidth = info.WO_Width;
                                stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                stateInfo_ls3.CodeALl = info.WO_PaperCode;

                                stateInfo_ls3.LastWidth = stateInfo_ls3.CurWidth;
                                stateInfo_ls3.LastFlute = stateInfo_ls3.CurFlute;
                                stateInfo_ls3.LastCode = stateInfo_ls3.CurCode;

                                stateInfo_ls3.BrandLS0 = "";
                                stateInfo_ls3.BrandLS1 = "";
                                stateInfo_ls3.BrandLS2 = "";
                                stateInfo_ls3.BrandLS3 = "";
                                stateInfo_ls3.BrandMS1 = "";
                                stateInfo_ls3.BrandMS2 = "";
                                stateInfo_ls3.BrandMS3 = "";
                            }
                            break;
                        default:
                            break;
                    }
                }

                #endregion

                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                if (everyPaper.Count >= 3)
                {
                    if (string.IsNullOrEmpty(stateInfo_ls0.CurFlute))
                    {
                        if (info.WO_PaperCode.Contains("."))
                        {
                            //如果面纸没有楞型
                            var ls0NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[0] != "-");
                            if (ls0NotNull != null)
                            {
                                stateInfo_ls0.CurFlute = ls0NotNull.WO_Wave;
                                stateInfo_ls0.CurCode = ls0NotNull.WO_PaperCode.Split('.')[0];
                                stateInfo_ls0.CurWidth = ls0NotNull.WO_Width;
                                stateInfo_ls0.CodeALl = ls0NotNull.WO_PaperCode;
                            }
                        }
                        else
                        {
                            //如果面纸没有楞型
                            var ls0NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[0] != '-');
                            if (ls0NotNull != null)
                            {
                                stateInfo_ls0.CurFlute = ls0NotNull.WO_Wave;
                                stateInfo_ls0.CurCode = ls0NotNull.WO_PaperCode.ToCharArray()[0].ToString();
                                stateInfo_ls0.CurWidth = ls0NotNull.WO_Width;
                                stateInfo_ls0.CodeALl = ls0NotNull.WO_PaperCode;
                            }
                        }
                    }
                    if (string.IsNullOrEmpty(stateInfo_ms1.CurFlute))
                    {
                        if (info.WO_PaperCode.Contains("."))
                        {
                            //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                            var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[1] != "-");
                            if (ms1NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[1];
                                stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[2];
                                stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                            }
                        }
                        else
                        {
                            //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                            var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[1] != '-');
                            if (ms1NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[1].ToString();
                                stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[2].ToString();
                                stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                            }
                        }
                    }
                }

                if (everyPaper.Count >= 5)
                {
                    if (string.IsNullOrEmpty(stateInfo_ms2.CurFlute))
                    {
                        if (info.WO_PaperCode.Contains("."))
                        {
                            //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                            var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[3] != "-");
                            if (ms2NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[3];
                                stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[4];
                                stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                            }
                        }
                        else
                        {
                            //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                            var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[3] != '-');
                            if (ms2NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[3].ToString();
                                stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[4].ToString();
                                stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                            }
                        }

                    }
                }

                if (everyPaper.Count >= 7)
                {
                    if (string.IsNullOrEmpty(stateInfo_ms3.CurFlute))
                    {
                        if (info.WO_PaperCode.Contains("."))
                        {
                            //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                            var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[5] != "-");
                            if (ms3NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[5];
                                stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[6];
                                stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                            }
                        }
                        else
                        {
                            //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                            var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[5] != '-');
                            if (ms3NotNull != null)
                            {
                                GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[5].ToString();
                                stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[6].ToString();
                                stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                            }
                        }

                    }
                }

                StringBuilder sb = new StringBuilder();
                sb.AppendLine("InitInfos 初始化各部位数据完成:");
                sb.AppendLine($"GU--材质={stateInfo_gu.CurCode}，门幅={stateInfo_gu.CurWidth}，楞型={stateInfo_gu.CurFlute}");
                sb.AppendLine($"LS0--材质={stateInfo_ls0.CurCode}，门幅={stateInfo_ls0.CurWidth}，楞型={stateInfo_ls0.CurFlute}，对应的订单材质={stateInfo_ls0.CodeALl}");
                sb.AppendLine($"MS1--材质={stateInfo_ms1.CurCode}，门幅={stateInfo_ms1.CurWidth}，楞型={stateInfo_ms1.CurFlute}，对应的订单材质={stateInfo_ms1.CodeALl}");
                sb.AppendLine($"LS1--材质={stateInfo_ls1.CurCode}，门幅={stateInfo_ls1.CurWidth}，楞型={stateInfo_ls1.CurFlute}，对应的订单材质={stateInfo_ls1.CodeALl}");
                sb.AppendLine($"MS2--材质={stateInfo_ms2.CurCode}，门幅={stateInfo_ms2.CurWidth}，楞型={stateInfo_ms2.CurFlute}，对应的订单材质={stateInfo_ms2.CodeALl}");
                sb.AppendLine($"LS2--材质={stateInfo_ls2.CurCode}，门幅={stateInfo_ls2.CurWidth}，楞型={stateInfo_ls2.CurFlute}，对应的订单材质={stateInfo_ls2.CodeALl}");
                sb.AppendLine($"MS3--材质={stateInfo_ms3.CurCode}，门幅={stateInfo_ms3.CurWidth}，楞型={stateInfo_ms3.CurFlute}，对应的订单材质={stateInfo_ms3.CodeALl}");
                sb.AppendLine($"LS3--材质={stateInfo_ls3.CurCode}，门幅={stateInfo_ls3.CurWidth}，楞型={stateInfo_ls3.CurFlute}，对应的订单材质={stateInfo_ls3.CodeALl}");
                logger.Info(sb.ToString(), module);
            }
            catch (Exception ex)
            {
                logger.Error($"InitInfos 初始化各部位数据异常失败：{ex}", module);
            }

        }

        /// <summary>
        /// 监听数据库是否有真实材质传入
        /// </summary>
        private async Task MonitorRealPaper()
        {
            //按照每个接纸机编号取表中所有未使用过的纸卷数据（IsUse=0），取时间最近的一条数据
            //取出数据之后把IsUse写成1
            //如果拿到的实材和当前正在使用的纸卷不一样（材质编码或者门幅不一样），则判定为接纸机换材：触发换材事件
            while (true)
            {
                try
                {
                    //程序刚启动，还没有对这些对象初始化赋值，不进行数据库监听处理动作,该线程任务是在SPChangePaper首次执行完之后才执行
                    if (begin == 0)
                        continue;
                    if (stateInfo_gu.CurCode == "")
                        continue;

                    //数据库没有新的材质
                    var list = paperService.GetList(it => it.State == 0);
                    if (list == null || list.Count == 0)
                        continue;

                    //把拿到的结果集标识成已使用状态
                    paperService.AsUpdateable().SetColumns(it => it.State == 1).Where(it => it.State == 0).ExecuteCommand();

                    //取最近的一次实际材质记录
                    var newlist = list.OrderBy(it => it.MachineID).OrderByDescending(it => it.CreateTime).ToList();
                    var ls0Info = newlist.FirstOrDefault(it => it.MachineID == "LS0");
                    if (ls0Info != null)
                    {

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ls0Info.Brand && paper.SPC_Code == ls0Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ls0.BrandLS0 != ls0Info.Brand)
                            isBrandChange = true;
                        #endregion

                        PubGetRealPaper(ls0Info);

                        if (ls0Info.ErpPaper != stateInfo_ls0.CurCode || ls0Info.ErpWidth != stateInfo_ls0.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用面纸
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[0] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[0] != '-')
                                    isUse = true;
                            }
                            //如果机台楞型是空，那么重新获取一下楞型
                            if (string.IsNullOrEmpty(stateInfo_ls0.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();

                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果面纸没有楞型
                                    var ls0NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[0] != "-");
                                    if (ls0NotNull != null)
                                    {
                                        stateInfo_ls0.CurFlute = ls0NotNull.WO_Wave;
                                        stateInfo_ls0.CurCode = ls0NotNull.WO_PaperCode.Split('.')[0];
                                        stateInfo_ls0.CurWidth = ls0NotNull.WO_Width;
                                        stateInfo_ls0.CodeALl = ls0NotNull.WO_PaperCode;
                                        stateInfo_ls0.NextBachCode = ls0NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果面纸没有楞型
                                    var ls0NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[0] != '-');
                                    if (ls0NotNull != null)
                                    {
                                        stateInfo_ls0.CurFlute = ls0NotNull.WO_Wave;
                                        stateInfo_ls0.CurCode = ls0NotNull.WO_PaperCode.ToCharArray()[0].ToString();
                                        stateInfo_ls0.CurWidth = ls0NotNull.WO_Width;
                                        stateInfo_ls0.CodeALl = ls0NotNull.WO_PaperCode;
                                        stateInfo_ls0.NextBachCode = ls0NotNull.WO_PaperCode;
                                    }
                                }

                            }

                            stateInfo_ls0.LastCode = stateInfo_ls0.CurCode;
                            stateInfo_ls0.LastWidth = stateInfo_ls0.CurWidth;
                            stateInfo_ls0.CurCode = ls0Info.ErpPaper;
                            stateInfo_ls0.CurWidth = Convert.ToInt32(ls0Info.ErpWidth);
                            stateInfo_ls0.BrandLS0 = ls0Info.Brand;

                            //如果当前糊机材质有面纸，则单独把面纸材质替换一下
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                var codes = stateInfo_gu.CurCode.Split('.');
                                if (codes[0] != "-")
                                {
                                    codes[0] = stateInfo_ls0.CurCode;
                                }
                                stateInfo_gu.CurCode = string.Join(".", codes);
                            }
                            else
                            {
                                List<string> codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                                if (codes[0] != "-")
                                {
                                    codes[0] = stateInfo_ls0.CurCode;
                                    stateInfo_gu.CurCode = string.Join("", codes);
                                }
                            }
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS0 实际材质={ls0Info.ErpPaper},实际门幅={ls0Info.ErpWidth}");
                            sb.AppendLine($"当前 LS0 材质={stateInfo_ls0.LastCode},门幅={stateInfo_ls0.LastWidth},楞型={stateInfo_ls0.CurFlute},订单材质={stateInfo_ls0.CodeALl}");
                            sb.AppendLine($"准备进入 HandleChangePaperLS0 处理LS0换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperLS0(true);

                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS0 实际材质={ls0Info.ErpPaper},实际门幅={ls0Info.ErpWidth}");
                            sb.AppendLine($"当前 LS0 材质={stateInfo_ls0.CurCode},门幅={stateInfo_ls0.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ms1Info = newlist.FirstOrDefault(it => it.MachineID == "MS1");
                    if (ms1Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "MS1");
                        if (spInfo != null)
                        {
                            spInfo.Code = ms1Info.ErpPaper;
                            spInfo.Brand = ms1Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("MS1"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ms1Info.Brand && paper.SPC_Code == ms1Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ms1.BrandMS1 != ms1Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ms1Info);
                        if (ms1Info.ErpPaper != stateInfo_ms1.CurCode || ms1Info.ErpWidth != stateInfo_ms1.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF1
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[1] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[1] != '-')
                                    isUse = true;
                            }
                            //如果MS1的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms1.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                                    var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[1] != "-");
                                    if (ms1NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                        stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                        stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[1];
                                        stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[2];
                                        stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                        stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                        stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                        stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                                    var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[1] != '-');
                                    if (ms1NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                        stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                        stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[1].ToString();
                                        stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[2].ToString();
                                        stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                        stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                        stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                        stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ms1.LastCode = stateInfo_ms1.CurCode;
                            stateInfo_ms1.LastWidth = stateInfo_ms1.CurWidth;
                            stateInfo_ms1.CurCode = ms1Info.ErpPaper;
                            stateInfo_ms1.CurWidth = Convert.ToInt32(ms1Info.ErpWidth);
                            stateInfo_ms1.BrandMS1 = ms1Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS1 实际材质={ms1Info.ErpPaper},实际门幅={ms1Info.ErpWidth}");
                            sb.AppendLine($"当前 MS1 材质={stateInfo_ms1.LastCode},门幅={stateInfo_ms1.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperMS1 处理MS1换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperMS1(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS1 实际材质={ms1Info.ErpPaper},实际门幅={ms1Info.ErpWidth}");
                            sb.AppendLine($"当前 MS1 材质={stateInfo_ms1.CurCode},门幅={stateInfo_ms1.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ls1Info = newlist.FirstOrDefault(it => it.MachineID == "LS1");
                    if (ls1Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "LS1");
                        if (spInfo != null)
                        {
                            spInfo.Code = ls1Info.ErpPaper;
                            spInfo.Brand = ls1Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("LS1"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ls1Info.Brand && paper.SPC_Code == ls1Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ls1.BrandLS1 != ls1Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ls1Info);
                        if (ls1Info.ErpPaper != stateInfo_ls1.CurCode || ls1Info.ErpWidth != stateInfo_ls1.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF1
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[1] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[1] != '-')
                                    isUse = true;
                            }
                            //如果MS1的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms1.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                                    var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[1] != "-");
                                    if (ms1NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                        stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                        stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[1];
                                        stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.Split('.')[2];
                                        stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                        stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                        stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                        stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果1芯没有楞型，则找到最近的一笔1芯不为-的订单，取其楞型赋值给 1芯楞型
                                    var ms1NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[1] != '-');
                                    if (ms1NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms1, ms1NotNull.WO_PaperCode, ms1NotNull.WO_Wave, "MS1");
                                        stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                        stateInfo_ms1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[1].ToString();
                                        stateInfo_ls1.CurCode = ms1NotNull.WO_PaperCode.ToCharArray()[2].ToString();
                                        stateInfo_ms1.CurWidth = ms1NotNull.WO_Width;
                                        stateInfo_ls1.CurWidth = ms1NotNull.WO_Width;

                                        stateInfo_ms1.CodeALl = ms1NotNull.WO_PaperCode;
                                        stateInfo_ls1.CodeALl = ms1NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ls1.LastCode = stateInfo_ls1.CurCode;
                            stateInfo_ls1.LastWidth = stateInfo_ls1.CurWidth;
                            stateInfo_ls1.CurCode = ls1Info.ErpPaper;
                            stateInfo_ls1.CurWidth = Convert.ToInt32(ls1Info.ErpWidth);
                            stateInfo_ls1.BrandLS1 = ls1Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS1 实际材质={ls1Info.ErpPaper},实际门幅={ls1Info.ErpWidth}");
                            sb.AppendLine($"当前 LS1 材质={stateInfo_ls1.LastCode},门幅={stateInfo_ls1.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperMS1 处理LS1换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperLS1(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS1 实际材质={ls1Info.ErpPaper},实际门幅={ls1Info.ErpWidth}");
                            sb.AppendLine($"当前 LS1 材质={stateInfo_ls1.CurCode},门幅={stateInfo_ls1.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ms2Info = newlist.FirstOrDefault(it => it.MachineID == "MS2");
                    if (ms2Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "MS2");
                        if (spInfo != null)
                        {
                            spInfo.Code = ms2Info.ErpPaper;
                            spInfo.Brand = ms2Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("MS2"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ms2Info.Brand && paper.SPC_Code == ms2Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ms2.BrandMS2 != ms2Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ms2Info);
                        if (ms2Info.ErpPaper != stateInfo_ms2.CurCode || ms2Info.ErpWidth != stateInfo_ms2.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF2
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[3] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[3] != '-')
                                    isUse = true;
                            }
                            //如果MS2的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms2.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                                    var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[3] != "-");
                                    if (ms2NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                        stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                        stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[3];
                                        stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[4];
                                        stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                        stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                        stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                        stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                                    var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[3] != '-');
                                    if (ms2NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                        stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                        stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[3].ToString();
                                        stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[4].ToString();
                                        stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                        stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                        stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                        stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ms2.LastCode = stateInfo_ms2.CurCode;
                            stateInfo_ms2.LastWidth = stateInfo_ms2.CurWidth;
                            stateInfo_ms2.CurCode = ms2Info.ErpPaper;
                            stateInfo_ms2.CurWidth = Convert.ToInt32(ms2Info.ErpWidth);
                            stateInfo_ms2.BrandMS2 = ms2Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS2 实际材质={ms2Info.ErpPaper},实际门幅={ms2Info.ErpWidth}");
                            sb.AppendLine($"当前 MS2 材质={stateInfo_ms2.LastCode},门幅={stateInfo_ms2.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperMS2 处理MS2换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperMS2(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS2 实际材质={ms2Info.ErpPaper},实际门幅={ms2Info.ErpWidth}");
                            sb.AppendLine($"当前 MS2 材质={stateInfo_ms2.CurCode},门幅={stateInfo_ms2.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ls2Info = newlist.FirstOrDefault(it => it.MachineID == "LS2");
                    if (ls2Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "LS2");
                        if (spInfo != null)
                        {
                            spInfo.Code = ls2Info.ErpPaper;
                            spInfo.Brand = ls2Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("LS2"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ls2Info.Brand && paper.SPC_Code == ls2Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ls2.BrandLS2 != ls2Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ls2Info);
                        if (ls2Info.ErpPaper != stateInfo_ls2.CurCode || ls2Info.ErpWidth != stateInfo_ls2.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF2
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[3] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[3] != '-')
                                    isUse = true;
                            }
                            //如果MS2的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms2.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                                    var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[3] != "-");
                                    if (ms2NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                        stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                        stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[3];
                                        stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.Split('.')[4];
                                        stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                        stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                        stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                        stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果2芯没有楞型，则找到最近的一笔2芯不为-的订单，取其楞型赋值给 2芯楞型
                                    var ms2NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[3] != '-');
                                    if (ms2NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms2, ms2NotNull.WO_PaperCode, ms2NotNull.WO_Wave, "MS2");
                                        stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                                        stateInfo_ms2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[3].ToString();
                                        stateInfo_ls2.CurCode = ms2NotNull.WO_PaperCode.ToCharArray()[4].ToString();
                                        stateInfo_ms2.CurWidth = ms2NotNull.WO_Width;
                                        stateInfo_ls2.CurWidth = ms2NotNull.WO_Width;

                                        stateInfo_ms2.CodeALl = ms2NotNull.WO_PaperCode;
                                        stateInfo_ls2.CodeALl = ms2NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ls2.LastCode = stateInfo_ls2.CurCode;
                            stateInfo_ls2.LastWidth = stateInfo_ls2.CurWidth;
                            stateInfo_ls2.CurCode = ls2Info.ErpPaper;
                            stateInfo_ls2.CurWidth = Convert.ToInt32(ls2Info.ErpWidth);
                            stateInfo_ls2.BrandLS2 = ls2Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS2 实际材质={ls2Info.ErpPaper},实际门幅={ls2Info.ErpWidth}");
                            sb.AppendLine($"当前 LS2 材质={stateInfo_ls2.LastCode},门幅={stateInfo_ls2.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperLS2 处理LS2换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperLS2(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS2 实际材质={ls2Info.ErpPaper},实际门幅={ls2Info.ErpWidth}");
                            sb.AppendLine($"当前 LS2 材质={stateInfo_ls2.CurCode},门幅={stateInfo_ls2.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ms3Info = newlist.FirstOrDefault(it => it.MachineID == "MS3");
                    if (ms3Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "MS3");
                        if (spInfo != null)
                        {
                            spInfo.Code = ms3Info.ErpPaper;
                            spInfo.Brand = ms3Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("MS3"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ms3Info.Brand && paper.SPC_Code == ms3Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ms3.BrandMS3 != ms3Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ms3Info);
                        if (ms3Info.ErpPaper != stateInfo_ms3.CurCode || ms3Info.ErpWidth != stateInfo_ms3.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF3
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[5] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[5] != '-')
                                    isUse = true;
                            }
                            //如果MS3的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms3.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                                    var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[5] != "-");
                                    if (ms3NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                        stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                        stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[5];
                                        stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[6];
                                        stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                        stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                        stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                        stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                                    var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[5] != '-');
                                    if (ms3NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                        stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                        stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[5].ToString();
                                        stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[6].ToString();
                                        stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                        stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                        stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                        stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ms3.LastCode = stateInfo_ms3.CurCode;
                            stateInfo_ms3.LastWidth = stateInfo_ms3.CurWidth;
                            stateInfo_ms3.CurCode = ms3Info.ErpPaper;
                            stateInfo_ms3.CurWidth = Convert.ToInt32(ms3Info.ErpWidth);
                            stateInfo_ms3.BrandMS3 = ms3Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS3 实际材质={ms3Info.ErpPaper},实际门幅={ms3Info.ErpWidth}");
                            sb.AppendLine($"当前 MS3 材质={stateInfo_ms3.LastCode},门幅={stateInfo_ms3.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperMS3 处理MS3换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperMS3(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 MS3 实际材质={ms3Info.ErpPaper},实际门幅={ms3Info.ErpWidth}");
                            sb.AppendLine($"当前 MS3 材质={stateInfo_ms3.CurCode},门幅={stateInfo_ms3.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }

                    var ls3Info = newlist.FirstOrDefault(it => it.MachineID == "LS3");
                    if (ls3Info != null)
                    {
                        #region 存储接纸机实材对象
                        var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == "LS3");
                        if (spInfo != null)
                        {
                            spInfo.Code = ls3Info.ErpPaper;
                            spInfo.Brand = ls3Info.Brand;
                        }
                        Task.Run(async () => { await GetSPRealPaperToChangeGUPaper("LS3"); });
                        #endregion

                        #region 判断这个原纸是否属于原纸供应商资料内
                        bool isBrandPaper = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == ls3Info.Brand && paper.SPC_Code == ls3Info.ErpPaper).Any();
                        #endregion

                        #region 判断本次拿到的实际材质品牌和上次的是否一致
                        bool isBrandChange = false;
                        if (stateInfo_ls3.BrandLS3 != ls3Info.Brand)
                            isBrandChange = true;
                        #endregion
                        PubGetRealPaper(ls3Info);
                        if (ls3Info.ErpPaper != stateInfo_ls3.CurCode || ls3Info.ErpWidth != stateInfo_ls3.CurWidth || isBrandPaper || isBrandChange)
                        {
                            bool isUse = false;//当前生产的订单是否使用SF3
                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                if (stateInfo_gu.CurCode.Split('.')[5] != "-")
                                    isUse = true;
                            }
                            else
                            {
                                if (stateInfo_gu.CurCode.ToCharArray()[5] != '-')
                                    isUse = true;
                            }
                            //如果MS3的机台楞型为空，则重新赋值
                            if (string.IsNullOrEmpty(stateInfo_ms3.CurFlute) || !isUse)
                            {
                                var info = oService.GetFirstByWorkNo();
                                var allOrders = oService.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                                if (info.WO_PaperCode.Contains("."))
                                {
                                    //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                                    var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[5] != "-");
                                    if (ms3NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                        stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                        stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[5];
                                        stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.Split('.')[6];
                                        stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                        stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                        stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                        stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                                    }
                                }
                                else
                                {
                                    //如果3芯没有楞型，则找到最近的一笔3芯不为-的订单，取其楞型赋值给 3芯楞型
                                    var ms3NotNull = allOrders.FirstOrDefault(it => it.WO_PaperCode.ToCharArray()[5] != '-');
                                    if (ms3NotNull != null)
                                    {
                                        GetSPFlute(ref stateInfo_ms3, ms3NotNull.WO_PaperCode, ms3NotNull.WO_Wave, "MS3");
                                        stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                                        stateInfo_ms3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[5].ToString();
                                        stateInfo_ls3.CurCode = ms3NotNull.WO_PaperCode.ToCharArray()[6].ToString();
                                        stateInfo_ms3.CurWidth = ms3NotNull.WO_Width;
                                        stateInfo_ls3.CurWidth = ms3NotNull.WO_Width;

                                        stateInfo_ms3.CodeALl = ms3NotNull.WO_PaperCode;
                                        stateInfo_ls3.CodeALl = ms3NotNull.WO_PaperCode;
                                    }
                                }
                            }

                            stateInfo_ls3.LastCode = stateInfo_ls3.CurCode;
                            stateInfo_ls3.LastWidth = stateInfo_ls3.CurWidth;
                            stateInfo_ls3.CurCode = ls3Info.ErpPaper;
                            stateInfo_ls3.CurWidth = Convert.ToInt32(ls3Info.ErpWidth);
                            stateInfo_ls3.BrandLS3 = ls3Info.Brand;
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS3 实际材质={ls3Info.ErpPaper},实际门幅={ls3Info.ErpWidth}");
                            sb.AppendLine($"当前 LS3 材质={stateInfo_ls3.LastCode},门幅={stateInfo_ls3.LastWidth}");
                            sb.AppendLine($"准备进入 HandleChangePaperLS3 处理LS3换材");
                            logger.Info(sb.ToString(), module);

                            //拿到不一样的材质，进行接纸机换材处理
                            HandleChangePaperLS3(true);
                        }
                        else
                        {
                            StringBuilder sb = new StringBuilder();
                            sb.AppendLine($"拿到了 LS3 实际材质={ls3Info.ErpPaper},实际门幅={ls3Info.ErpWidth}");
                            sb.AppendLine($"当前 LS3 材质={stateInfo_ls3.CurCode},门幅={stateInfo_ls3.CurWidth}");
                            sb.AppendLine($"材质一样，不需要赋值");
                            logger.Info(sb.ToString(), module);
                        }
                    }
                }
                catch (Exception)
                {
                }
                finally { await Task.Delay(1000); }
            }
        }

        /// <summary>
        /// 接纸机换材监听任务
        /// 接纸机换材的前提一定是换卷
        /// </summary>
        private async Task SPChangePaper()
        {
            while (true)
            {
                try
                {
                    if (comm == null)
                        continue;
                    if (comm.PointVars == null)
                        continue;
                    var changeRollInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.PLCChangeRoll.ToString());
                    if (changeRollInfo == null)
                        continue;
                    string states = changeRollInfo.VarValue;

                    #region 换卷信号异常处理 7层和5层这边要修改数字
                    if (states.Length < floor)
                        continue;
                    char[] rightStates = { '1', '2', '3', '4' };
                    bool isOK = true;
                    foreach (var c in states.ToCharArray())
                    {
                        if (!rightStates.Contains(c))
                        {
                            isOK = false;
                            break;
                        }
                    }
                    if (!isOK)
                    {
                        logger.Warn($"换卷信号={states},判定为异常，不继续往下执行,等待正确的PLC信号", module);
                        continue;
                    }

                    #endregion

                    if (begin == 0)
                    {
                        #region 服务端刚启动时，对state对象赋值，并且向机器上发送数据
                        //把每个接纸机的状态值变成从PLC里面读取到的值
                        char[] chars = states.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            switch (i)
                            {
                                case 0:
                                    stateInfo_ls0.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 1:
                                    stateInfo_ms1.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 2:
                                    stateInfo_ls1.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 3:
                                    stateInfo_ms2.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 4:
                                    stateInfo_ls2.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 5:
                                    stateInfo_ms3.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                case 6:
                                    stateInfo_ls3.LastPlcChangeRoll_Part = chars[i].ToString();
                                    break;
                                default:
                                    break;
                            }
                        }
                        logger.Info("服务端刚启动，执行 HandleFirstAll 对所有设备部位进行点位赋值", module);
                        //通知所有相关类进行赋值处理
                        HandleFirstAll();
                        begin++;
                        #endregion
                    }
                    else
                    {
                        #region 判断是否换卷，如果换卷，再判断是否换材，如果换材则执行相应动作
                        /**
                         * 判断每个接纸机是否换卷
                         * 如果换卷，按照原来的逻辑：先从数据库把已使用过的纸卷删除，然后取未使用过的纸卷作为当前正在用的实际材质
                         * 因为此步骤原来的逻辑代码里面有处理，这边防止冲突不再进行处理
                         * 而是在数据库里面增加一个触发器，把ERP传入的条码信息插入到另外的一个表中，我们直接取另外的表的记录
                         * 找表中最近的一条没有使用过的记录：
                         * --如果本次拿到的实际材质和上一次使用的材质不一样，则判断为换材
                         * --如果拿不到实际材质，判断同材剩余米数（是否已经落在换材准备区间内），若已判定换材准备中，则判断为换材，取下一批材质作为当前材质
                         * 如果MS和LS有一个是换材，那么整个SF都是换材，触发SF换材赋值事件
                         * 如果LS0判定为换材，则 DF换材，触发DF换材赋值事件
                         */
                        List<string> statesNow = new List<string>();
                        foreach (var item in states.ToCharArray())
                        {
                            statesNow.Add(item.ToString());
                        }
                        for (int i = 0; i < statesNow.Count; i++)
                        {
                            switch (i)
                            {
                                case 0:
                                    stateInfo_ls0.IsChangeRoll = IsChangeRoll(stateInfo_ls0.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ls0.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 1:
                                    stateInfo_ms1.IsChangeRoll = IsChangeRoll(stateInfo_ms1.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ms1.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 2:
                                    stateInfo_ls1.IsChangeRoll = IsChangeRoll(stateInfo_ls1.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ls1.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 3:
                                    stateInfo_ms2.IsChangeRoll = IsChangeRoll(stateInfo_ms2.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ms2.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 4:
                                    stateInfo_ls2.IsChangeRoll = IsChangeRoll(stateInfo_ls2.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ls2.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 5:
                                    stateInfo_ms3.IsChangeRoll = IsChangeRoll(stateInfo_ms3.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ms3.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                case 6:
                                    stateInfo_ls3.IsChangeRoll = IsChangeRoll(stateInfo_ls3.LastPlcChangeRoll_Part, statesNow[i]);
                                    stateInfo_ls3.LastPlcChangeRoll_Part = statesNow[i];
                                    break;
                                default:
                                    break;
                            }

                        }
                        if (stateInfo_ls0.IsChangeRoll)
                        {
                            stateInfo_ls0.IsChangeRoll = false;
                            logger.Info("LS0 判定为换卷，准备进入 HandleChangeRollLS0 函数", module);
                            HandleChangeRollLS0();
                        }

                        if (stateInfo_ms1.IsChangeRoll)
                        {
                            stateInfo_ms1.IsChangeRoll = false;
                            logger.Info("MS1 判定为换卷，准备进入 HandleChangeRollMS1 函数", module);
                            HandleChangeRollMS1();
                        }

                        if (stateInfo_ls1.IsChangeRoll)
                        {
                            stateInfo_ls1.IsChangeRoll = false;
                            logger.Info("LS1 判定为换卷，准备进入 HandleChangeRollLS1 函数", module);
                            HandleChangeRollLS1();
                        }

                        if (stateInfo_ms2.IsChangeRoll)
                        {
                            stateInfo_ms2.IsChangeRoll = false;
                            logger.Info("MS2 判定为换卷，准备进入 HandleChangeRollMS2 函数", module);
                            HandleChangeRollMS2();
                        }

                        if (stateInfo_ls2.IsChangeRoll)
                        {
                            stateInfo_ls2.IsChangeRoll = false;
                            logger.Info("LS2 判定为换卷，准备进入 HandleChangeRollLS2 函数", module);
                            HandleChangeRollLS2();
                        }

                        if (stateInfo_ms3.IsChangeRoll)
                        {
                            stateInfo_ms3.IsChangeRoll = false;
                            logger.Info("MS3 判定为换卷，准备进入 HandleChangeRollMS3 函数", module);
                            HandleChangeRollMS3();
                        }

                        if (stateInfo_ls3.IsChangeRoll)
                        {
                            stateInfo_ls3.IsChangeRoll = false;
                            logger.Info("LS3 判定为换卷，准备进入 HandleChangeRollLS3 函数", module);
                            HandleChangeRollLS3();
                        }

                        #endregion
                    }


                }
                catch (Exception ex)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine("SPChangePaper--监听接纸机换材过程中发生异常：");
                    sb.AppendLine(ex.ToString());
                    logger.Error(sb.ToString(), module);
                }
                finally
                {
                    await Task.Delay(500);
                }
            }
        }

        /// <summary>
        /// 按照同材剩余米数判断接纸机是否进入换材准备中
        /// </summary>
        private async Task SPChangePaperReady()
        {
            while (true)
            {
                try
                {
                    var ls0 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.LS0_Remaining_mm.ToString());
                    var ms1 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.MS1_Remaining_mm.ToString());
                    var ls1 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.LS1_Remaining_mm.ToString());
                    var ms2 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.MS2_Remaining_mm.ToString());
                    var ls2 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.LS2_Remaining_mm.ToString());
                    var ms3 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.MS3_Remaining_mm.ToString());
                    var ls3 = comm?.PointVars.Find(it => it.VarCode == PointVarEnum.LS3_Remaining_mm.ToString());

                    if (ls0 != null)
                    {
                        decimal remain = ls0.VarValue.ToDecimal();//同材剩余米数从MDI拿到的数值单位是mm
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ls0.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ls0.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ls0.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ls0.SPRange1 = 0;
                            stateInfo_ls0.SPRange2 = 0;
                            stateInfo_ls0.SPRange3 = 0;
                        }
                    }

                    if (ms1 != null)
                    {
                        decimal remain = ms1.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ms1.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ms1.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ms1.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ms1.SPRange1 = 0;
                            stateInfo_ms1.SPRange2 = 0;
                            stateInfo_ms1.SPRange3 = 0;
                        }
                    }

                    if (ls1 != null)
                    {
                        decimal remain = ls1.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ls1.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ls1.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ls1.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ls1.SPRange1 = 0;
                            stateInfo_ls1.SPRange2 = 0;
                            stateInfo_ls1.SPRange3 = 0;
                        }
                    }

                    if (ms2 != null)
                    {
                        decimal remain = ms2.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ms2.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ms2.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ms2.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ms2.SPRange1 = 0;
                            stateInfo_ms2.SPRange2 = 0;
                            stateInfo_ms2.SPRange3 = 0;
                        }
                    }

                    if (ls2 != null)
                    {
                        decimal remain = ls2.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ls2.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ls2.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ls2.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ls2.SPRange1 = 0;
                            stateInfo_ls2.SPRange2 = 0;
                            stateInfo_ls2.SPRange3 = 0;
                        }
                    }

                    if (ms3 != null)
                    {
                        decimal remain = ms3.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ms3.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ms3.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ms3.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ms3.SPRange1 = 0;
                            stateInfo_ms3.SPRange2 = 0;
                            stateInfo_ms3.SPRange3 = 0;
                        }
                    }

                    if (ls3 != null)
                    {
                        decimal remain = ls3.VarValue.ToDecimal();
                        if (remain > 0 && remain <= 20000)
                        {
                            stateInfo_ls3.SPRange1++;
                        }
                        else if (remain > 20000 && remain <= 30000)
                        {
                            stateInfo_ls3.SPRange2++;
                        }
                        else if (remain > 30000 && remain <= 40000)
                        {
                            stateInfo_ls3.SPRange3++;
                        }
                        else
                        {
                            stateInfo_ls3.SPRange1 = 0;
                            stateInfo_ls3.SPRange2 = 0;
                            stateInfo_ls3.SPRange3 = 0;
                        }
                    }

                    int cntls0 = Math.Max(stateInfo_ls0.SPRange1, Math.Max(stateInfo_ls0.SPRange2, stateInfo_ls0.SPRange3));
                    int cntls1 = Math.Max(stateInfo_ls1.SPRange1, Math.Max(stateInfo_ls1.SPRange2, stateInfo_ls1.SPRange3));
                    int cntls2 = Math.Max(stateInfo_ls2.SPRange1, Math.Max(stateInfo_ls2.SPRange2, stateInfo_ls2.SPRange3));
                    int cntls3 = Math.Max(stateInfo_ls3.SPRange1, Math.Max(stateInfo_ls3.SPRange2, stateInfo_ls3.SPRange3));
                    int cntms1 = Math.Max(stateInfo_ms1.SPRange1, Math.Max(stateInfo_ms1.SPRange2, stateInfo_ms1.SPRange3));
                    int cntms2 = Math.Max(stateInfo_ms2.SPRange1, Math.Max(stateInfo_ms2.SPRange2, stateInfo_ms2.SPRange3));
                    int cntms3 = Math.Max(stateInfo_ms3.SPRange1, Math.Max(stateInfo_ms3.SPRange2, stateInfo_ms3.SPRange3));

                    if (cntls0 >= 5 && !stateInfo_ls0.VChangePaper)
                    {
                        stateInfo_ls0.SPRange1 = 0;
                        stateInfo_ls0.SPRange2 = 0;
                        stateInfo_ls0.SPRange3 = 0;
                        stateInfo_ls0.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls0, orders, "LS0");
                    }
                    else if (cntls0 >= 5 && stateInfo_ls0.VChangePaper)
                    {
                        stateInfo_ls0.SPRange1 = 0;
                        stateInfo_ls0.SPRange2 = 0;
                        stateInfo_ls0.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls0, orders, "LS0");
                    }

                    if (cntms1 >= 5 && !stateInfo_ms1.VChangePaper)
                    {
                        stateInfo_ms1.SPRange1 = 0;
                        stateInfo_ms1.SPRange2 = 0;
                        stateInfo_ms1.SPRange3 = 0;
                        stateInfo_ms1.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms1, orders, "MS1");
                    }
                    else if (cntms1 >= 5 && stateInfo_ms1.VChangePaper)
                    {
                        stateInfo_ms1.SPRange1 = 0;
                        stateInfo_ms1.SPRange2 = 0;
                        stateInfo_ms1.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms1, orders, "MS1");
                    }

                    if (cntls1 >= 5 && !stateInfo_ls1.VChangePaper)
                    {
                        stateInfo_ls1.SPRange1 = 0;
                        stateInfo_ls1.SPRange2 = 0;
                        stateInfo_ls1.SPRange3 = 0;
                        stateInfo_ls1.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls1, orders, "LS1");
                    }
                    else if (cntls1 >= 5 && stateInfo_ls1.VChangePaper)
                    {
                        stateInfo_ls1.SPRange1 = 0;
                        stateInfo_ls1.SPRange2 = 0;
                        stateInfo_ls1.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls1, orders, "LS1");
                    }

                    if (cntms2 >= 5 && !stateInfo_ms2.VChangePaper)
                    {
                        stateInfo_ms2.SPRange1 = 0;
                        stateInfo_ms2.SPRange2 = 0;
                        stateInfo_ms2.SPRange3 = 0;
                        stateInfo_ms2.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms2, orders, "MS2");
                    }
                    else if (cntms2 >= 5 && stateInfo_ms2.VChangePaper)
                    {
                        stateInfo_ms2.SPRange1 = 0;
                        stateInfo_ms2.SPRange2 = 0;
                        stateInfo_ms2.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms2, orders, "MS2");
                    }

                    if (cntls2 >= 5 && !stateInfo_ls2.VChangePaper)
                    {
                        stateInfo_ls2.SPRange1 = 0;
                        stateInfo_ls2.SPRange2 = 0;
                        stateInfo_ls2.SPRange3 = 0;
                        stateInfo_ls2.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls2, orders, "LS2");
                    }
                    else if (cntls2 >= 5 && stateInfo_ls2.VChangePaper)
                    {
                        stateInfo_ls2.SPRange1 = 0;
                        stateInfo_ls2.SPRange2 = 0;
                        stateInfo_ls2.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls2, orders, "LS2");
                    }

                    if (cntms3 >= 5 && !stateInfo_ms3.VChangePaper)
                    {
                        stateInfo_ms3.SPRange1 = 0;
                        stateInfo_ms3.SPRange2 = 0;
                        stateInfo_ms3.SPRange3 = 0;
                        stateInfo_ms3.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms3, orders, "MS3");
                    }
                    else if (cntms3 >= 5 && stateInfo_ms3.VChangePaper)
                    {
                        stateInfo_ms3.SPRange1 = 0;
                        stateInfo_ms3.SPRange2 = 0;
                        stateInfo_ms3.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ms3, orders, "MS3");
                    }

                    if (cntls3 >= 5 && !stateInfo_ls3.VChangePaper)
                    {
                        stateInfo_ls3.SPRange1 = 0;
                        stateInfo_ls3.SPRange2 = 0;
                        stateInfo_ls3.SPRange3 = 0;
                        stateInfo_ls3.VChangePaper = true;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls3, orders, "LS3");
                    }
                    else if (cntls3 >= 5 && stateInfo_ls3.VChangePaper)
                    {
                        stateInfo_ls3.SPRange1 = 0;
                        stateInfo_ls3.SPRange2 = 0;
                        stateInfo_ls3.SPRange3 = 0;
                        var orders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        CalSPNextBatchInfo(ref stateInfo_ls3, orders, "LS3");
                    }
                }
                catch (Exception ex)
                {
                    logger.Error($"SPChangePaperReady--判断同材剩余米数是否进入换材等待区错误:{ex}", module);
                }
                finally
                {
                    await Task.Delay(300);
                }
            }
        }

        /// <summary>
        /// 糊机换材监听任务
        /// </summary>
        private async Task DFChangePaper()
        {
            //糊机换材不需要考虑提前蒸汽赋值的问题，蒸汽赋值由单独的业务类实现了。
            while (true)
            {

                try
                {
                    decimal dfRemain = 0;
                    dfRemain = GlobalControl.dfSameMaterilaLeftMeter;//这边其实是MDI同材剩余米数-糊机到横切的距离

                    var curOrder = oService.GetFirstByWorkNo();//当前正在生产的订单
                    string curCode = "";
                    int curWidth = 0;
                    if (curOrder != null)
                    {
                        curCode = curOrder.WO_PaperCode;
                        curWidth = curOrder.WO_Width;
                    }

                    if (dfRemain == 0 || dfRemain < -50 || dfRemain > 100)
                    {
                        continue;
                    }

                    if (dfRemain > 0 && dfRemain <= 30)
                    {
                        stateInfo_gu.GuRange1++;
                    }
                    else if (dfRemain > 30 && dfRemain <= 60)
                    {
                        stateInfo_gu.GuRange2++;
                    }
                    else if (dfRemain > 60 && dfRemain <= 90)
                    {
                        stateInfo_gu.GuRange3++;
                    }

                    if (dfRemain > 0 && dfRemain <= 90)
                    {
                        logger.Info($"DFChangePaper--dfRemain={dfRemain},GuRange1={stateInfo_gu.GuRange1},GuRange2={stateInfo_gu.GuRange2},GuRange3={stateInfo_gu.GuRange3}", module);
                    }

                    if (stateInfo_gu.GuRange1 >= 5 || stateInfo_gu.GuRange2 >= 5 || stateInfo_gu.GuRange3 >= 5)
                    {
                        List<string> PreSetInfo = GlobalControl.GetDictItems(DictTypesEnum.PreHQMeter.ToString());
                        int preHQMeter = PreSetInfo[0].ToInt32();
                        if (dfRemain <= preHQMeter)
                        {
                            logger.Info($"DFChangePaper--已进入换材处理:糊机同材剩余-横切到糊机的距离={dfRemain},糊机到横切距离设定值={preHQMeter}", module);

                            //取下批材质，下批门幅，下批楞型
                            var nextBachFirstOrder = BLLFactory<OrderInfoManage>.Instance.AsQueryable().Where(x => x.WO_PaperCode != curCode || x.WO_Width != curWidth).OrderBy(x => x.WO_WorkNo).First();

                            if (nextBachFirstOrder != null)
                            {
                                //用取到的 下批材质，下批门幅，下批楞型 替换 本批材质，本批门幅，本批楞型 
                                string nextCode = nextBachFirstOrder.WO_PaperCode;
                                int nextWidth = nextBachFirstOrder.WO_Width;
                                string nextFlute = nextBachFirstOrder.WO_Wave;

                                if (stateInfo_gu.CurCode != nextCode || nextWidth != stateInfo_gu.CurWidth)
                                {
                                    stateInfo_gu.LastWidth = stateInfo_gu.CurWidth;
                                    stateInfo_gu.LastFlute = stateInfo_gu.CurFlute;
                                    stateInfo_gu.LastCode = stateInfo_gu.CurCode;

                                    //获取当前的各接纸机真实材质，判断是用理论材质还是真实材质（判断依据为克重差，如果超过30则用理论的）
                                    //GetRealPaperCode(nextCode, out string realPaperCode);
                                    stateInfo_gu.CurCode = nextCode;
                                    stateInfo_gu.CurWidth = nextWidth;
                                    stateInfo_gu.CurFlute = nextFlute;
                                    stateInfo_gu.NextBachCode = nextCode;
                                    //糊机判定为换材的时候，需要把LS0的当前楞型更改 ,糊机换材还是应该完全独立，不和面纸混在一起搞
                                    //stateInfo_ls0.LastFlute = stateInfo_ls0.CurFlute;
                                    //stateInfo_ls0.CurFlute = nextFlute;
                                    StringBuilder sb = new StringBuilder();
                                    sb.AppendLine("DFChangePaper--糊机判定为换材");
                                    sb.AppendLine($"上次使用的：材质={stateInfo_gu.LastCode}，楞型={stateInfo_gu.LastFlute}，门幅={stateInfo_gu.LastWidth}");
                                    sb.AppendLine($"即将使用的：材质={stateInfo_gu.CurCode}，楞型={stateInfo_gu.CurFlute}，门幅={stateInfo_gu.CurWidth};下批材质={nextCode}");
                                    sb.AppendLine("准备进入HandleGuChangePaper具体执行函数");
                                    logger.Info(sb.ToString(), module);
                                    //判定为糊机换材，发送相关部位赋值消息
                                    HandleGuChangePaper();
                                    stateInfo_gu.GuRange1 = 0;
                                    stateInfo_gu.GuRange2 = 0;
                                    stateInfo_gu.GuRange3 = 0;
                                    while (true)
                                    {
                                        try
                                        {
                                            var curOrderNew = oService.GetFirstByWorkNo();
                                            if (curOrderNew.WO_PaperCode == curCode && curOrderNew.WO_Width == curWidth)
                                            {
                                                continue;
                                            }
                                            else
                                            {
                                                //添加换材分析任务
                                                await BLLFactory<JobInfoManager>.Instance.AddPaperAnalysisJob();
                                                break;
                                            }
                                        }
                                        catch (Exception)
                                        {

                                        }
                                        finally
                                        {
                                            await Task.Delay(1000);
                                        }
                                    }
                                }
                            }

                        }

                    }
                }
                catch (Exception ex)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine("DFChangePaper--监听糊机换材过程中发生异常：");
                    sb.AppendLine(ex.Message);
                    logger.Error(sb.ToString(), module);
                }
                finally
                {
                    await Task.Delay(500);
                }
            }
        }

        /// <summary>
        /// 判断是否换卷
        /// </summary>
        /// <param name="oldValue">之前的状态值</param>
        /// <param name="newValue">当前拿到的状态值</param>
        /// <returns></returns>
        private bool IsChangeRoll(string oldValue, string newValue)
        {
            bool result = false;
            try
            {
                if ((oldValue == "1" || oldValue == "2") && (newValue == "3" || newValue == "4"))
                    result = true;
                else if ((oldValue == "3" || oldValue == "4") && (newValue == "1" || newValue == "2"))
                    result = true;
            }
            catch (Exception ex)
            {
                logger.Error($"IsChangeRoll-判断是否换卷出现异常：{ex}", module);
            }

            return result;
        }

        /// <summary>
        /// LS0换卷业务处理函数
        /// </summary>
        private void HandleChangeRollLS0()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollLS0");
                logger.Info(sb.ToString(), module);
                sb.Clear();

                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "LS0";

                #region 新逻辑
                //判定为换卷换材的时候并不需要判断实际材质，因为实际材质一直都在轮询判断处理，并且换卷后给到的实际材质会晚于换卷信号
                if (stateInfo_ls0.VChangePaper)
                {
                    string curCode = stateInfo_ls0.NextBatchTheoryCode;
                    int curWidth = stateInfo_ls0.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        //同材剩余到0的时候，并没有找到下批订单，只能重新按照当前订单找对应的下批订单
                        var curOrder = oService.GetFirstByWorkNo();
                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[0];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[0].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string ls0Code = "";
                            if (pCode.Contains("."))
                            {
                                ls0Code = pCode.Split('.')[0];
                            }
                            else
                            {
                                ls0Code = pCode.ToCharArray()[0].ToString();
                            }
                            if (ls0Code != "-" && (ls0Code != curCode || width != curWidth))
                            {
                                stateInfo_ls0.LastFlute = stateInfo_ls0.CurFlute;
                                stateInfo_ls0.CurFlute = item.WO_Wave;
                                stateInfo_ls0.LastCode = stateInfo_ls0.CurCode;
                                stateInfo_ls0.LastWidth = stateInfo_ls0.CurWidth;
                                stateInfo_ls0.CurCode = ls0Code;
                                stateInfo_ls0.CurWidth = width;
                                stateInfo_ls0.NextBachCode = pCode;//下批的订单全材质
                                stateInfo_ls0.CodeALl = pCode;//对应订单材质
                                //如果当前糊机材质有面纸，则单独把面纸材质替换一下
                                if (stateInfo_gu.CurCode.Contains("."))
                                {
                                    var codes = stateInfo_gu.CurCode.Split('.');
                                    if (codes[0] != "-")
                                    {
                                        codes[0] = stateInfo_ls0.CurCode;
                                    }
                                    stateInfo_gu.CurCode = string.Join(".", codes);
                                }
                                else
                                {
                                    List<string> codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                                    if (codes[0] != "-")
                                    {
                                        codes[0] = stateInfo_ls0.CurCode;
                                        stateInfo_gu.CurCode = string.Join("", codes);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ls0.NextBatchTheoryCode != "-" && (stateInfo_ls0.NextBatchTheoryCode != stateInfo_ls0.CurCode || stateInfo_ls0.NextBatchTheoryWidth != stateInfo_ls0.CurWidth))
                        {
                            stateInfo_ls0.LastFlute = stateInfo_ls0.CurFlute;
                            stateInfo_ls0.LastCode = stateInfo_ls0.CurCode;
                            stateInfo_ls0.LastWidth = stateInfo_ls0.CurWidth;
                            stateInfo_ls0.CurFlute = stateInfo_ls0.NextBatchTheoryFlute;
                            stateInfo_ls0.CurCode = stateInfo_ls0.NextBatchTheoryCode;
                            stateInfo_ls0.CurWidth = stateInfo_ls0.NextBatchTheoryWidth;
                            stateInfo_ls0.NextBachCode = stateInfo_ls0.NextBatchTheoryCodeAll;
                            stateInfo_ls0.CodeALl = stateInfo_ls0.NextBatchTheoryCodeAll;

                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                var codes = stateInfo_gu.CurCode.Split('.');
                                if (codes[0] != "-")
                                {
                                    codes[0] = stateInfo_ls0.CurCode;
                                }
                                stateInfo_gu.CurCode = string.Join(".", codes);
                            }
                            else
                            {
                                List<string> codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                                if (codes[0] != "-")
                                {
                                    codes[0] = stateInfo_ls0.CurCode;
                                    stateInfo_gu.CurCode = string.Join("", codes);
                                }
                            }
                        }
                    }

                    //清理下批理论材质，品牌，以及换材标志位
                    stateInfo_ls0.NextBatchTheoryCodeAll = "";
                    stateInfo_ls0.NextBatchTheoryCode = "";
                    stateInfo_ls0.NextBatchTheoryWidth = 0;
                    stateInfo_ls0.NextBatchTheoryFlute = "";
                    stateInfo_ls0.BrandLS0 = "";
                    stateInfo_ls0.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollLS0--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ls0.LastCode}，门幅={stateInfo_ls0.LastWidth}，楞型={stateInfo_ls0.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ls0.CurCode}，门幅={stateInfo_ls0.CurWidth}，楞型={stateInfo_ls0.CurFlute}，下批订单全材质={stateInfo_ls0.NextBachCode}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();
                    HandleChangePaperLS0();

                    PubChangePaper(new PartPaperCode { Part = "LS0", PaperCode = stateInfo_ls0.CurCode });

                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion
                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollLS0 过程异常：{ex.Message}", module);
            }

        }

        /// <summary>
        /// MS1换卷业务处理函数
        /// </summary>
        private void HandleChangeRollMS1()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollMS1");
                logger.Info(sb.ToString(), module);
                sb.Clear();

                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "MS1";
                #region 新逻辑
                //判定为换卷换材的时候并不需要判断实际材质，因为实际材质一直都在轮询判断处理，并且换卷后给到的实际材质会晚于换卷信号
                if (stateInfo_ms1.VChangePaper)
                {
                    string curCode = stateInfo_ms1.NextBatchTheoryCode;
                    int curWidth = stateInfo_ms1.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();
                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[1];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[1].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[1];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[1].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ms1, item.WO_PaperCode, item.WO_Wave, "MS1");
                                stateInfo_ms1.LastCode = stateInfo_ms1.CurCode;
                                stateInfo_ms1.LastWidth = stateInfo_ms1.CurWidth;
                                stateInfo_ms1.CurCode = code;
                                stateInfo_ms1.CurWidth = width;
                                stateInfo_ms1.NextBachCode = pCode;
                                stateInfo_ms1.CodeALl = pCode;

                                stateInfo_ls1.LastFlute = stateInfo_ls1.CurFlute;
                                stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ms1.NextBatchTheoryCode != "-" && (stateInfo_ms1.NextBatchTheoryCode != stateInfo_ms1.CurCode || stateInfo_ms1.NextBatchTheoryWidth != stateInfo_ms1.CurWidth))
                        {
                            stateInfo_ms1.LastFlute = stateInfo_ms1.CurFlute;
                            stateInfo_ms1.LastCode = stateInfo_ms1.CurCode;
                            stateInfo_ms1.LastWidth = stateInfo_ms1.CurWidth;
                            stateInfo_ls1.CurFlute = stateInfo_ms1.NextBatchTheoryFlute;
                            stateInfo_ms1.CurCode = stateInfo_ms1.NextBatchTheoryCode;
                            stateInfo_ms1.CurWidth = stateInfo_ms1.NextBatchTheoryWidth;
                            stateInfo_ms1.NextBachCode = stateInfo_ms1.NextBatchTheoryCodeAll;
                            stateInfo_ms1.CodeALl = stateInfo_ms1.NextBatchTheoryCodeAll;

                            stateInfo_ls1.LastFlute = stateInfo_ls1.CurFlute;
                            stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;
                        }
                    }

                    stateInfo_ms1.NextBatchTheoryFlute = "";
                    stateInfo_ms1.NextBatchTheoryCode = "";
                    stateInfo_ms1.NextBatchTheoryWidth = 0;
                    stateInfo_ms1.NextBatchTheoryCodeAll = "";
                    stateInfo_ms1.BrandMS1 = "";
                    stateInfo_ms1.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollMS1--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ms1.LastCode}，门幅={stateInfo_ms1.LastWidth}，楞型={stateInfo_ms1.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ms1.CurCode}，门幅={stateInfo_ms1.CurWidth}，楞型={stateInfo_ms1.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();
                    HandleChangePaperMS1();
                    PubChangePaper(new PartPaperCode { Part = "MS1", PaperCode = stateInfo_ms1.CurCode });

                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion

                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollMS1 过程异常：{ex.Message}", module);
            }

        }

        /// <summary>
        /// LS1换卷业务处理函数
        /// </summary>
        private void HandleChangeRollLS1()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollLS1");
                logger.Info(sb.ToString(), module);
                sb.Clear();

                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "LS1";
                #region 新逻辑
                if (stateInfo_ls1.VChangePaper)
                {
                    string curCode = stateInfo_ls1.NextBatchTheoryCode;
                    int curWidth = stateInfo_ls1.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();
                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[2];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[2].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[2];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[2].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ls1, item.WO_PaperCode, item.WO_Wave, "LS1");
                                stateInfo_ls1.LastCode = stateInfo_ls1.CurCode;
                                stateInfo_ls1.LastWidth = stateInfo_ls1.CurWidth;
                                stateInfo_ls1.CurCode = code;
                                stateInfo_ls1.CurWidth = width;
                                stateInfo_ls1.NextBachCode = pCode;
                                stateInfo_ls1.CodeALl = pCode;

                                stateInfo_ms1.LastFlute = stateInfo_ms1.CurFlute;
                                stateInfo_ms1.CurFlute = stateInfo_ls1.CurFlute;

                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ls1.NextBatchTheoryCode != "-" && (stateInfo_ls1.NextBatchTheoryCode != stateInfo_ls1.CurCode || stateInfo_ls1.NextBatchTheoryWidth != stateInfo_ls1.CurWidth))
                        {
                            stateInfo_ls1.LastFlute = stateInfo_ls1.CurFlute;
                            stateInfo_ls1.LastCode = stateInfo_ls1.CurCode;
                            stateInfo_ls1.LastWidth = stateInfo_ls1.CurWidth;
                            stateInfo_ls1.CurFlute = stateInfo_ls1.NextBatchTheoryFlute;
                            stateInfo_ls1.CurCode = stateInfo_ls1.NextBatchTheoryCode;
                            stateInfo_ls1.CurWidth = stateInfo_ls1.NextBatchTheoryWidth;
                            stateInfo_ls1.NextBachCode = stateInfo_ls1.NextBatchTheoryCodeAll;
                            stateInfo_ls1.CodeALl = stateInfo_ls1.NextBatchTheoryCodeAll;

                            stateInfo_ms1.LastFlute = stateInfo_ms1.CurFlute;
                            stateInfo_ms1.CurFlute = stateInfo_ls1.CurFlute;
                        }
                    }
                    stateInfo_ls1.NextBatchTheoryFlute = "";
                    stateInfo_ls1.NextBatchTheoryCode = "";
                    stateInfo_ls1.NextBatchTheoryWidth = 0;
                    stateInfo_ls1.NextBatchTheoryCodeAll = "";
                    stateInfo_ls1.BrandLS1 = "";
                    stateInfo_ls1.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollLS1--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ls1.LastCode}，门幅={stateInfo_ls1.LastWidth}，楞型={stateInfo_ls1.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ls1.CurCode}，门幅={stateInfo_ls1.CurWidth}，楞型={stateInfo_ls1.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();

                    HandleChangePaperLS1();
                    PubChangePaper(new PartPaperCode { Part = "LS1", PaperCode = stateInfo_ls1.CurCode });

                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion

                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollMS1 过程异常：{ex.Message}", module);
            }

        }

        /// <summary>
        /// MS2换卷业务处理函数
        /// </summary>
        private void HandleChangeRollMS2()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollMS2");
                logger.Info(sb.ToString(), module);
                sb.Clear();

                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "MS2";
                #region 新逻辑
                if (stateInfo_ms2.VChangePaper)
                {
                    string curCode = stateInfo_ms2.NextBatchTheoryCode;
                    int curWidth = stateInfo_ms2.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();

                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[3];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[3].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[3];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[3].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ms2, item.WO_PaperCode, item.WO_Wave, "MS2");
                                stateInfo_ms2.LastCode = stateInfo_ms2.CurCode;
                                stateInfo_ms2.LastWidth = stateInfo_ms2.CurWidth;
                                stateInfo_ms2.CurCode = code;
                                stateInfo_ms2.CurWidth = width;
                                stateInfo_ms2.CodeALl = pCode;
                                stateInfo_ms2.NextBachCode = pCode;

                                stateInfo_ls2.LastFlute = stateInfo_ls2.CurFlute;
                                stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;

                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ms2.NextBatchTheoryCode != "-" && (stateInfo_ms2.NextBatchTheoryCode != stateInfo_ms2.CurCode || stateInfo_ms2.NextBatchTheoryWidth != stateInfo_ms2.CurWidth))
                        {
                            stateInfo_ms2.LastFlute = stateInfo_ms2.CurFlute;
                            stateInfo_ms2.LastCode = stateInfo_ms2.CurCode;
                            stateInfo_ms2.LastWidth = stateInfo_ms2.CurWidth;
                            stateInfo_ms2.CurFlute = stateInfo_ms2.NextBatchTheoryFlute;
                            stateInfo_ms2.CurCode = stateInfo_ms2.NextBatchTheoryCode;
                            stateInfo_ms2.CurWidth = stateInfo_ms2.NextBatchTheoryWidth;
                            stateInfo_ms2.CodeALl = stateInfo_ms2.NextBatchTheoryCodeAll;
                            stateInfo_ms2.NextBachCode = stateInfo_ms2.NextBatchTheoryCodeAll;

                            stateInfo_ls2.LastFlute = stateInfo_ls2.CurFlute;
                            stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;
                        }
                    }
                    stateInfo_ms2.NextBatchTheoryFlute = "";
                    stateInfo_ms2.NextBatchTheoryCode = "";
                    stateInfo_ms2.NextBatchTheoryWidth = 0;
                    stateInfo_ms2.NextBatchTheoryCodeAll = "";
                    stateInfo_ms2.BrandMS2 = "";
                    stateInfo_ms2.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollMS2--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ms2.LastCode}，门幅={stateInfo_ms2.LastWidth}，楞型={stateInfo_ms2.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ms2.CurCode}，门幅={stateInfo_ms2.CurWidth}，楞型={stateInfo_ms2.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();

                    HandleChangePaperMS2();
                    PubChangePaper(new PartPaperCode { Part = "MS2", PaperCode = stateInfo_ms2.CurCode });
                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion
                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollMS2 过程异常：{ex.Message}", module);
            }

        }

        /// <summary>
        /// LS2换卷业务处理函数
        /// </summary>
        private void HandleChangeRollLS2()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollLS2");
                logger.Info(sb.ToString(), module);
                sb.Clear();
                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "LS2";
                #region 新逻辑
                if (stateInfo_ls2.VChangePaper)
                {
                    string curCode = stateInfo_ls2.NextBatchTheoryCode;
                    int curWidth = stateInfo_ls2.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();

                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[4];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[4].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[4];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[4].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ls2, item.WO_PaperCode, item.WO_Wave, "LS2");
                                stateInfo_ls2.LastCode = stateInfo_ls2.CurCode;
                                stateInfo_ls2.LastWidth = stateInfo_ls2.CurWidth;
                                stateInfo_ls2.CurCode = code;
                                stateInfo_ls2.CurWidth = width;
                                stateInfo_ls2.CodeALl = pCode;
                                stateInfo_ls2.NextBachCode = pCode;

                                stateInfo_ms2.LastFlute = stateInfo_ms2.CurFlute;
                                stateInfo_ms2.CurFlute = stateInfo_ls2.CurFlute;

                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ls2.NextBatchTheoryCode != "-" && (stateInfo_ls2.NextBatchTheoryCode != stateInfo_ls2.CurCode || stateInfo_ls2.NextBatchTheoryWidth != stateInfo_ls2.CurWidth))
                        {
                            stateInfo_ls2.LastFlute = stateInfo_ls2.CurFlute;
                            stateInfo_ls2.LastCode = stateInfo_ls2.CurCode;
                            stateInfo_ls2.LastWidth = stateInfo_ls2.CurWidth;
                            stateInfo_ls2.CurFlute = stateInfo_ls2.NextBatchTheoryFlute;
                            stateInfo_ls2.CurCode = stateInfo_ls2.NextBatchTheoryCode;
                            stateInfo_ls2.CurWidth = stateInfo_ls2.NextBatchTheoryWidth;
                            stateInfo_ls2.CodeALl = stateInfo_ls2.NextBatchTheoryCodeAll;
                            stateInfo_ls2.NextBachCode = stateInfo_ls2.NextBatchTheoryCodeAll;

                            stateInfo_ms2.LastFlute = stateInfo_ms2.CurFlute;
                            stateInfo_ms2.CurFlute = stateInfo_ls2.CurFlute;

                        }
                    }
                    stateInfo_ls2.NextBatchTheoryFlute = "";
                    stateInfo_ls2.NextBatchTheoryCode = "";
                    stateInfo_ls2.NextBatchTheoryWidth = 0;
                    stateInfo_ls2.NextBatchTheoryCodeAll = "";
                    stateInfo_ls2.BrandLS2 = "";
                    stateInfo_ls2.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollLS2--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ls2.LastCode}，门幅={stateInfo_ls2.LastWidth}，楞型={stateInfo_ls2.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ls2.CurCode}，门幅={stateInfo_ls2.CurWidth}，楞型={stateInfo_ls2.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();

                    HandleChangePaperLS2();
                    PubChangePaper(new PartPaperCode { Part = "LS2", PaperCode = stateInfo_ls2.CurCode });
                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion
                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollLS2 过程异常：{ex.Message}", module);
            }


        }

        /// <summary>
        /// MS3换卷业务处理函数
        /// </summary>
        private void HandleChangeRollMS3()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollMS3");
                logger.Info(sb.ToString(), module);
                sb.Clear();
                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "MS3";
                #region 新逻辑
                if (stateInfo_ms3.VChangePaper)
                {
                    string curCode = stateInfo_ms3.NextBatchTheoryCode;
                    int curWidth = stateInfo_ms3.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();

                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[5];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[5].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[5];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[5].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ms3, item.WO_PaperCode, item.WO_Wave, "MS3");
                                stateInfo_ms3.LastCode = stateInfo_ms3.CurCode;
                                stateInfo_ms3.LastWidth = stateInfo_ms3.CurWidth;
                                stateInfo_ms3.CurCode = code;
                                stateInfo_ms3.CurWidth = width;
                                stateInfo_ms3.CodeALl = pCode;
                                stateInfo_ms3.NextBachCode = pCode;

                                stateInfo_ls3.LastFlute = stateInfo_ls3.CurFlute;
                                stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;

                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ms3.NextBatchTheoryCode != "-" && (stateInfo_ms3.NextBatchTheoryCode != stateInfo_ms3.CurCode || stateInfo_ms3.NextBatchTheoryWidth != stateInfo_ms3.CurWidth))
                        {
                            stateInfo_ms3.LastFlute = stateInfo_ms3.CurFlute;
                            stateInfo_ms3.LastCode = stateInfo_ms3.CurCode;
                            stateInfo_ms3.LastWidth = stateInfo_ms3.CurWidth;
                            stateInfo_ms3.CurFlute = stateInfo_ms3.NextBatchTheoryFlute;
                            stateInfo_ms3.CurCode = stateInfo_ms3.NextBatchTheoryCode;
                            stateInfo_ms3.CurWidth = stateInfo_ms3.NextBatchTheoryWidth;
                            stateInfo_ms3.CodeALl = stateInfo_ms3.NextBatchTheoryCodeAll;
                            stateInfo_ms3.NextBachCode = stateInfo_ms3.NextBatchTheoryCodeAll;

                            stateInfo_ls3.LastFlute = stateInfo_ls3.CurFlute;
                            stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;
                        }
                    }
                    stateInfo_ms3.NextBatchTheoryFlute = "";
                    stateInfo_ms3.NextBatchTheoryCode = "";
                    stateInfo_ms3.NextBatchTheoryWidth = 0;
                    stateInfo_ms3.NextBatchTheoryCodeAll = "";
                    stateInfo_ms3.BrandMS3 = "";
                    stateInfo_ms3.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollMS3--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ms3.LastCode}，门幅={stateInfo_ms3.LastWidth}，楞型={stateInfo_ms3.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ms3.CurCode}，门幅={stateInfo_ms3.CurWidth}，楞型={stateInfo_ms3.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();

                    HandleChangePaperMS3();
                    PubChangePaper(new PartPaperCode { Part = "MS3", PaperCode = stateInfo_ms3.CurCode });
                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion
                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollMS3 过程异常：{ex.Message}", module);
            }

        }

        /// <summary>
        /// LS3换卷业务处理函数
        /// </summary>
        private void HandleChangeRollLS3()
        {
            try
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("进入到换卷处理函数 HandleChangeRollLS3");
                logger.Info(sb.ToString(), module);
                sb.Clear();
                ChangeRollRemainEventModel eInfo = new ChangeRollRemainEventModel();
                eInfo.MachineID = "LS3";
                #region 新逻辑
                if (stateInfo_ls3.VChangePaper)
                {
                    string curCode = stateInfo_ls3.NextBatchTheoryCode;
                    int curWidth = stateInfo_ls3.NextBatchTheoryWidth;
                    if (string.IsNullOrEmpty(curCode))
                    {
                        var curOrder = oService.GetFirstByWorkNo();

                        if (curOrder != null)
                        {
                            curWidth = curOrder.WO_Width;
                            if (curOrder.WO_PaperCode.Contains("."))
                            {
                                curCode = curOrder.WO_PaperCode.Split('.')[6];
                            }
                            else
                            {
                                curCode = curOrder.WO_PaperCode.ToCharArray()[6].ToString();
                            }
                        }
                        //获取下批同材的首笔订单的楞型 赋值给当前LS0的楞型
                        var allorders = BLLFactory<OrderInfoManage>.Instance.AsQueryable().OrderBy(it => it.WO_WorkNo).ToList();
                        foreach (var item in allorders)
                        {
                            List<string> codes = new List<string>();
                            int width = item.WO_Width;
                            string pCode = item.WO_PaperCode;
                            string code = "";
                            string fluteAll = item.WO_Wave;
                            if (pCode.Contains("."))
                            {
                                codes = pCode.Split('.').ToList();
                                code = pCode.Split('.')[6];
                            }
                            else
                            {
                                code = pCode.ToCharArray()[6].ToString();
                                foreach (var c in pCode)
                                {
                                    codes.Add(c.ToString());
                                }
                            }
                            if (code != "-" && (code != curCode || width != curWidth))
                            {
                                GetSPFlute(ref stateInfo_ls3, item.WO_PaperCode, item.WO_Wave, "LS3");
                                stateInfo_ls3.LastCode = stateInfo_ls3.CurCode;
                                stateInfo_ls3.LastWidth = stateInfo_ls3.CurWidth;
                                stateInfo_ls3.CurCode = code;
                                stateInfo_ls3.CurWidth = width;
                                stateInfo_ls3.CodeALl = pCode;
                                stateInfo_ls3.NextBachCode = pCode;

                                stateInfo_ms3.LastFlute = stateInfo_ms3.CurFlute;
                                stateInfo_ms3.CurFlute = stateInfo_ls3.CurFlute;

                                break;
                            }
                        }
                    }
                    else
                    {
                        if (stateInfo_ls3.NextBatchTheoryCode != "-" && (stateInfo_ls3.NextBatchTheoryCode != stateInfo_ls3.CurCode || stateInfo_ls3.NextBatchTheoryWidth != stateInfo_ls3.CurWidth))
                        {
                            stateInfo_ls3.LastFlute = stateInfo_ls3.CurFlute;
                            stateInfo_ls3.LastCode = stateInfo_ls3.CurCode;
                            stateInfo_ls3.LastWidth = stateInfo_ls3.CurWidth;
                            stateInfo_ls3.CurFlute = stateInfo_ls3.NextBatchTheoryFlute;
                            stateInfo_ls3.CurCode = stateInfo_ls3.NextBatchTheoryCode;
                            stateInfo_ls3.CurWidth = stateInfo_ls3.NextBatchTheoryWidth;
                            stateInfo_ls3.CodeALl = stateInfo_ls3.NextBatchTheoryCodeAll;
                            stateInfo_ls3.NextBachCode = stateInfo_ls3.NextBatchTheoryCodeAll;

                            stateInfo_ms3.LastFlute = stateInfo_ms3.CurFlute;
                            stateInfo_ms3.CurFlute = stateInfo_ls3.CurFlute;
                        }
                    }
                    stateInfo_ls3.NextBatchTheoryFlute = "";
                    stateInfo_ls3.NextBatchTheoryCode = "";
                    stateInfo_ls3.NextBatchTheoryWidth = 0;
                    stateInfo_ls3.NextBatchTheoryCodeAll = "";
                    stateInfo_ls3.BrandLS3 = "";
                    stateInfo_ls3.VChangePaper = false;

                    sb.AppendLine($"HandleChangeRollLS3--已经进入换材准备中状态了，用下批理论材质进行赋值操作");
                    sb.AppendLine($"当前：材质={stateInfo_ls3.LastCode}，门幅={stateInfo_ls3.LastWidth}，楞型={stateInfo_ls3.LastFlute}");
                    sb.AppendLine($"下批理论：材质={stateInfo_ls3.CurCode}，门幅={stateInfo_ls3.CurWidth}，楞型={stateInfo_ls3.CurFlute}");
                    logger.Info(sb.ToString(), module);
                    sb.Clear();

                    HandleChangePaperLS3();
                    PubChangePaper(new PartPaperCode { Part = "LS3", PaperCode = stateInfo_ls3.CurCode });
                    eInfo.Flag = 1;
                }
                else
                {
                    eInfo.Flag = 2;
                }
                #endregion
                ChangeRollRemain(eInfo);
            }
            catch (Exception ex)
            {
                logger.Error($"执行 HandleChangeRollLS3 过程异常：{ex.Message}", module);
            }


        }

        /// <summary>
        /// 服务端刚启动，首次全部给机器写一遍点位数据 立刻赋值
        /// </summary>
        private void HandleFirstAll(List<string> flags = null)
        {
            try
            {
                List<PubChangeNowInfo> list = new List<PubChangeNowInfo>();
                if (flags == null || flags.Contains("LS0"))
                {
                    string dfCode = stateInfo_gu.CurCode;
                    string flute = stateInfo_gu.CurFlute;
                    if (dfCode.Contains("."))
                    {
                        var papers = dfCode.Split('.').Where(it => it != "-").ToList();
                        dfCode = string.Join(".", papers);
                    }
                    else
                    {
                        dfCode = dfCode.Replace("-", "");
                    }
                    QdmCtrl.GetQdmDFCoef(dfCode, flute);

                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.GlueGu,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.WrapGu,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.WrapGu_Add2,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.BridgeTension,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.PressGroupQty,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.HotPress,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.CodePress,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.RidingRoll,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });

                    list.Add(new PubChangeNowInfo
                    {
                        Part = IPSHandlePart.DFDualBoost,
                        IsFirst = true,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.CurCode,
                        LastFlute = stateInfo_gu.CurFlute,
                        LastWidth = stateInfo_gu.CurWidth
                    });
                }
                if (flags == null || flags.Contains("LS0"))
                {
                    if (!string.IsNullOrEmpty(stateInfo_ls0.CurCode) && stateInfo_ls0.CurCode != "-")
                    {
                        string allcode = stateInfo_ls0.CodeALl;
                        if (stateInfo_ls0.CodeALl.Contains("."))
                        {
                            var papers = stateInfo_ls0.CodeALl.Split('.').Where(it => it != "-").ToList();
                            allcode = string.Join(".", papers);
                        }
                        else
                        {
                            allcode = stateInfo_ls0.CodeALl.Replace("-", "");
                        }
                        QdmCtrl.GetQdmDFCoef(allcode, stateInfo_ls0.CurFlute);

                        //面纸包角
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS0,
                            IsFirst = true,
                            Code = stateInfo_ls0.CurCode + "/" + stateInfo_ls0.CodeALl,
                            Width = stateInfo_ls0.CurWidth,
                            Flute = stateInfo_ls0.CurFlute,
                            LastCode = stateInfo_ls0.CurCode,
                            LastWidth = stateInfo_ls0.CurWidth,
                            LastFlute = stateInfo_ls0.CurFlute,
                            BrandLS0 = stateInfo_ls0.BrandLS0
                        });
                        //面纸张力
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionLS0,
                            IsFirst = true,
                            Code = stateInfo_ls0.CurCode + "/" + stateInfo_gu.CurCode,
                            Width = stateInfo_ls0.CurWidth,
                            Flute = stateInfo_ls0.CurFlute,
                            LastCode = stateInfo_ls0.CurCode,
                            LastWidth = stateInfo_ls0.CurWidth,
                            LastFlute = stateInfo_ls0.CurFlute,
                            BrandLS0 = stateInfo_ls0.BrandLS0
                        });
                    }
                }
                if (flags == null || flags.Contains("MS1") || flags.Contains("LS1"))
                {
                    if (!string.IsNullOrEmpty(stateInfo_ms1.CurCode) && stateInfo_ms1.CurCode != "-")
                    {
                        string sfCodeMS = stateInfo_ms1.CurCode;
                        string sfCodeLS = stateInfo_ls1.CurCode;
                        string sfFlute = stateInfo_ms1.CurFlute;
                        QdmCtrl.GetQdmCoefSFInfo(sfCodeMS, sfCodeLS, sfFlute);

                        //SF1糊间隙重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.GlueSF1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //MS1包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //MS1ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS1_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //LS1包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ls1.CurWidth,
                            Flute = stateInfo_ls1.CurFlute,
                            LastCode = stateInfo_ls1.CurCode,
                            LastWidth = stateInfo_ls1.CurWidth,
                            LastFlute = stateInfo_ls1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //LS1ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS1_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ls1.CurWidth,
                            Flute = stateInfo_ls1.CurFlute,
                            LastCode = stateInfo_ls1.CurCode,
                            LastWidth = stateInfo_ls1.CurWidth,
                            LastFlute = stateInfo_ls1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //MS1接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionMS1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //LS1接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionLS1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ls1.CurWidth,
                            Flute = stateInfo_ls1.CurFlute,
                            LastCode = stateInfo_ls1.CurCode,
                            LastWidth = stateInfo_ls1.CurWidth,
                            LastFlute = stateInfo_ls1.CurFlute,
                            BrandMS1 = stateInfo_ms1.BrandMS1,
                            BrandLS1 = stateInfo_ls1.BrandLS1,
                        });
                        //SF1压力辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.PressRollSF1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute
                        });
                        //SF1瓦楞辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.CorrugatedRollSF1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute
                        });
                        //SF1热喷雾
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.HotSpraySF1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.CurCode,
                            LastWidth = stateInfo_ms1.CurWidth,
                            LastFlute = stateInfo_ms1.CurFlute
                        });
                        //SF1真空泵重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.VacuumBlowerSF1,
                            IsFirst = true,
                            Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                            Width = stateInfo_ms1.CurWidth,
                            Flute = stateInfo_ms1.CurFlute,
                            LastCode = stateInfo_ms1.LastCode,
                            LastWidth = stateInfo_ms1.LastWidth,
                            LastFlute = stateInfo_ms1.LastFlute
                        });
                    }
                }
                if (flags == null || flags.Contains("MS2") || flags.Contains("LS2"))
                {
                    if (!string.IsNullOrEmpty(stateInfo_ms2.CurCode) && stateInfo_ms2.CurCode != "-")
                    {
                        string sfCodeMS = stateInfo_ms2.CurCode;
                        string sfCodeLS = stateInfo_ls2.CurCode;
                        string sfFlute = stateInfo_ms2.CurFlute;
                        QdmCtrl.GetQdmCoefSFInfo(sfCodeMS, sfCodeLS, sfFlute);

                        //SF2糊间隙重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.GlueSF2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //MS2包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //MS2ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS2_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //LS2包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ls2.CurWidth,
                            Flute = stateInfo_ls2.CurFlute,
                            LastCode = stateInfo_ls2.CurCode,
                            LastWidth = stateInfo_ls2.CurWidth,
                            LastFlute = stateInfo_ls2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //LS2ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS2_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ls2.CurWidth,
                            Flute = stateInfo_ls2.CurFlute,
                            LastCode = stateInfo_ls2.CurCode,
                            LastWidth = stateInfo_ls2.CurWidth,
                            LastFlute = stateInfo_ls2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //MS2接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionMS2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //LS2接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionLS2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ls2.CurWidth,
                            Flute = stateInfo_ls2.CurFlute,
                            LastCode = stateInfo_ls2.CurCode,
                            LastWidth = stateInfo_ls2.CurWidth,
                            LastFlute = stateInfo_ls2.CurFlute,
                            BrandMS2 = stateInfo_ms2.BrandMS2,
                            BrandLS2 = stateInfo_ls2.BrandLS2,
                        });
                        //SF2压力辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.PressRollSF2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute
                        });
                        //SF2瓦楞辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.CorrugatedRollSF2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute
                        });
                        //SF2热喷雾
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.HotSpraySF2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.CurCode,
                            LastWidth = stateInfo_ms2.CurWidth,
                            LastFlute = stateInfo_ms2.CurFlute
                        });
                        //SF2真空泵重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.VacuumBlowerSF2,
                            IsFirst = true,
                            Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                            Width = stateInfo_ms2.CurWidth,
                            Flute = stateInfo_ms2.CurFlute,
                            LastCode = stateInfo_ms2.LastCode,
                            LastWidth = stateInfo_ms2.LastWidth,
                            LastFlute = stateInfo_ms2.LastFlute
                        });
                    }
                }
                if (flags == null || flags.Contains("MS3") || flags.Contains("LS3"))
                {
                    if (!string.IsNullOrEmpty(stateInfo_ms3.CurCode) && stateInfo_ms3.CurCode != "-")
                    {
                        string sfCodeMS = stateInfo_ms3.CurCode;
                        string sfCodeLS = stateInfo_ls3.CurCode;
                        string sfFlute = stateInfo_ms3.CurFlute;
                        QdmCtrl.GetQdmCoefSFInfo(sfCodeMS, sfCodeLS, sfFlute);

                        //SF3糊间隙重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.GlueSF3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //MS3包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //MS3ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapMS3_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //LS3包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ls3.CurWidth,
                            Flute = stateInfo_ls3.CurFlute,
                            LastCode = stateInfo_ls3.CurCode,
                            LastWidth = stateInfo_ls3.CurWidth,
                            LastFlute = stateInfo_ls3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //LS3ext包角重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.WrapLS3_Ext,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ls3.CurWidth,
                            Flute = stateInfo_ls3.CurFlute,
                            LastCode = stateInfo_ls3.CurCode,
                            LastWidth = stateInfo_ls3.CurWidth,
                            LastFlute = stateInfo_ls3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //MS3接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionMS3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //LS3接纸机张力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.TensionLS3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ls3.CurWidth,
                            Flute = stateInfo_ls3.CurFlute,
                            LastCode = stateInfo_ls3.CurCode,
                            LastWidth = stateInfo_ls3.CurWidth,
                            LastFlute = stateInfo_ls3.CurFlute,
                            BrandMS3 = stateInfo_ms3.BrandMS3,
                            BrandLS3 = stateInfo_ls3.BrandLS3,
                        });
                        //SF3压力辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.PressRollSF3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute
                        });
                        //SF3瓦楞辊压力重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.CorrugatedRollSF3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute
                        });
                        //SF3热喷雾
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.HotSpraySF3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.CurCode,
                            LastWidth = stateInfo_ms3.CurWidth,
                            LastFlute = stateInfo_ms3.CurFlute
                        });
                        //SF1真空泵重新赋值
                        list.Add(new PubChangeNowInfo
                        {
                            Part = IPSHandlePart.VacuumBlowerSF3,
                            IsFirst = true,
                            Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                            Width = stateInfo_ms3.CurWidth,
                            Flute = stateInfo_ms3.CurFlute,
                            LastCode = stateInfo_ms3.LastCode,
                            LastWidth = stateInfo_ms3.LastWidth,
                            LastFlute = stateInfo_ms3.LastFlute
                        });
                    }
                }

                if (list.Count > 0)
                {
                    logger.Info($"HandleFirstAll 构造通知消息报文完成，准备发送......", module);
                    foreach (var info in list)
                    {
                        PubChangeNow(info);
                    }
                    logger.Info($"HandleFirstAll 发送完毕", module);
                }
            }
            catch (Exception ex)
            {
                logger.Error($"HandleFirstAll执行过程中异常报错：{ex}", module);
            }
        }

        /// <summary>
        /// 糊机换材处理函数
        /// </summary>
        private void HandleGuChangePaper()
        {
            try
            {
                string currentGuCode = stateInfo_gu.CurCode;
                List<string> allCodes = new List<string>();

                //检查是否有历史QDM DF系数，没有则生成
                List<string> codes = new List<string>();
                string code = "";
                if (stateInfo_gu.CurCode.Contains("."))
                {
                    codes = stateInfo_gu.CurCode.Split('.').Where(it => it != "-").ToList();
                    code = string.Join(".", codes);

                    allCodes = stateInfo_gu.CurCode.Split('.').ToList();
                }
                else
                {
                    foreach (var item in stateInfo_gu.CurCode.ToCharArray())
                    {
                        if (item != '-')
                        {
                            codes.Add(item.ToString());
                        }
                    }
                    code = string.Join("", codes);
                    allCodes = stateInfo_gu.CurCode.ToCharArray().Select(it => it.ToString()).ToList();
                }

                var dictDatas = BLLFactory<DictDataInfoManager>.Instance.Context.Queryable<DictDataInfo>()
                      .LeftJoin<DictTypeInfo>((data, type) => data.PD_TypeID == type.PD_ID)
                      .Where((data, type) => type.PD_Code == "DistanceToGU")
                      .ToList();
                string brandLS0 = "";
                string brandMS1 = "";
                string brandLS1 = "";
                string brandMS2 = "";
                string brandLS2 = "";
                string brandMS3 = "";
                string brandLS3 = "";
                if (dictDatas != null && dictDatas.Count > 0)
                {
                    var isUseInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "IsUseDistanceToGU");
                    if (isUseInfo != null && isUseInfo.PD_Value.ToLower() == "true")
                    {

                        //对糊机材质进行特殊处理，如果启用了糊机实在处理逻辑，则需要结合糊机实材对糊机材质特殊处理
                        for (int i = 0; i < allCodes.Count; i++)
                        {
                            switch (i)
                            {
                                case 0:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS0))
                                    {
                                        allCodes[i] = _temp_GU.Code_LS0;
                                        brandLS0 = _temp_GU.Brand_LS0;
                                    }
                                    break;
                                case 1:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS1))
                                    {
                                        allCodes[i] = _temp_GU.Code_MS1;
                                        brandMS1 = _temp_GU.Brand_MS1;
                                    }
                                    break;
                                case 2:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS1))
                                    {
                                        allCodes[i] = _temp_GU.Code_LS1;
                                        brandLS1 = _temp_GU.Brand_LS1;
                                    }
                                    break;

                                case 3:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS2))
                                    {
                                        allCodes[i] = _temp_GU.Code_MS2;
                                        brandMS2 = _temp_GU.Brand_MS2;
                                    }
                                    break;
                                case 4:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS2))
                                    {
                                        allCodes[i] = _temp_GU.Code_LS2;
                                        brandLS2 = _temp_GU.Brand_LS2;
                                    }
                                    break;

                                case 5:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS3))
                                    {
                                        allCodes[i] = _temp_GU.Code_MS3;
                                        brandMS3 = _temp_GU.Brand_MS3;
                                    }
                                    break;
                                case 6:
                                    if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS3))
                                    {
                                        allCodes[i] = _temp_GU.Code_LS3;
                                        brandLS3 = _temp_GU.Brand_LS3;
                                    }
                                    break;
                                default:
                                    break;
                            }
                        }

                        var newList = allCodes.Where(t => t != "-").ToList();
                        if (stateInfo_gu.CurCode.Contains("."))
                        {
                            code = string.Join(".", newList);
                            currentGuCode = string.Join(".", allCodes);
                        }
                        else
                        {
                            code = string.Join("", newList);
                            currentGuCode = string.Join("", allCodes);
                        }

                        logger.Info($"开启糊机实材处理，本次糊机理论材质={stateInfo_gu.CurCode}，实际赋值使用实际材质={currentGuCode}", module);
                    }
                }



                //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                if (codes.Count != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                {
                    logger.Warn($"HandleGuChangePaper--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}，stateInfo_gu.CurCode={stateInfo_gu.CurCode}", module);
                    return;
                }
                QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);

                List<PublishInfo> list = new List<PublishInfo>
                {
                    //糊机糊间隙重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.GlueGu,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth,
                        BrandLS0 = brandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    },
                    //糊机包角重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth,
                        BrandLS0 = brandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    },
                    new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu_Add2,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth,
                        BrandLS0 = brandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    },
                    //压板组数重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.PressGroupQty,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth
                    },
                    //天桥张力重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.BridgeTension,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth,
                        BrandLS0 = brandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    },
                    //热板压力重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.HotPress,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth
                    },
                    //冷板压力重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.CodePress,
                        Code = stateInfo_gu.CurCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth
                    },
                    //糊机骑辊重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.RidingRoll,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth
                    },
                    //糊机楞尖加速器重新赋值
                    new PublishInfo
                    {
                        Part = IPSHandlePart.DFDualBoost,
                        Code = currentGuCode,
                        Flute = stateInfo_gu.CurFlute,
                        Width = stateInfo_gu.CurWidth,
                        LastCode = stateInfo_gu.LastCode,
                        LastFlute = stateInfo_gu.LastFlute,
                        LastWidth = stateInfo_gu.LastWidth
                    },
                };
                Publish(list);
                logger.Info($"HandleGuChangePaper 已发送糊机换材消息，通知各执行类", module);
            }
            catch (Exception ex)
            {
                logger.Error($"HandleGuChangePaper执行过程中异常报错：{ex.Message}", module);
            }

        }

        /// <summary>
        /// 面纸接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperLS0(bool isReal = false)
        {
            try
            {
                List<PublishInfo> list = new List<PublishInfo>();
                if (isReal)
                {
                    string brandMS1 = stateInfo_ms1.BrandMS1;
                    string brandLS1 = stateInfo_ls1.BrandLS1;
                    string brandMS2 = stateInfo_ms2.BrandMS2;
                    string brandLS2 = stateInfo_ls2.BrandLS2;
                    string brandMS3 = stateInfo_ms3.BrandMS3;
                    string brandLS3 = stateInfo_ls3.BrandLS3;

                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成; BM-621 要求只替换面纸材质，其他的保持不变
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    string codeAll = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        //for (int i = 0; i < codes.Count; i++)
                        //{
                        //    if (codes[i] == "-")
                        //        continue;
                        //    switch (i)
                        //    {
                        //        case 0:
                        //            codes[i] = stateInfo_ls0.CurCode;
                        //            break;
                        //        case 1:
                        //            codes[i] = stateInfo_ms1.CurCode;
                        //            break;
                        //        case 2:
                        //            codes[i] = stateInfo_ls1.CurCode;
                        //            break;
                        //        case 3:
                        //            codes[i] = stateInfo_ms2.CurCode;
                        //            break;
                        //        case 4:
                        //            codes[i] = stateInfo_ls2.CurCode;
                        //            break;
                        //        case 5:
                        //            codes[i] = stateInfo_ms3.CurCode;
                        //            break;
                        //        case 6:
                        //            codes[i] = stateInfo_ls3.CurCode;
                        //            break;
                        //        default:
                        //            break;
                        //    }
                        //}

                        codes[0] = stateInfo_ls0.CurCode;
                        code = string.Join(".", codes.Where(it => it != "-"));
                        codeAll = string.Join(".", codes);
                    }
                    else
                    {
                        //var chars = stateInfo_gu.CurCode.ToCharArray();
                        //for (int i = 0; i < chars.Length; i++)
                        //{
                        //    if (chars[i] == '-')
                        //    {
                        //        codes.Add("-");
                        //        continue;
                        //    }

                        //    switch (i)
                        //    {
                        //        case 0:
                        //            codes.Add(stateInfo_ls0.CurCode);
                        //            break;
                        //        case 1:
                        //            codes.Add(stateInfo_ms1.CurCode);
                        //            break;
                        //        case 2:
                        //            codes.Add(stateInfo_ls1.CurCode);
                        //            break;
                        //        case 3:
                        //            codes.Add(stateInfo_ms2.CurCode);
                        //            break;
                        //        case 4:
                        //            codes.Add(stateInfo_ls2.CurCode);
                        //            break;
                        //        case 5:
                        //            codes.Add(stateInfo_ms3.CurCode);
                        //            break;
                        //        case 6:
                        //            codes.Add(stateInfo_ls3.CurCode);
                        //            break;
                        //        default:
                        //            break;
                        //    }
                        //}

                        codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                        code = string.Join("", codes.Where(it => it != "-"));
                        codeAll = string.Join("", codes);
                    }


                    var dictDatas = BLLFactory<DictDataInfoManager>.Instance.Context.Queryable<DictDataInfo>()
                     .LeftJoin<DictTypeInfo>((data, type) => data.PD_TypeID == type.PD_ID).Where((data, type) => type.PD_Code == "DistanceToGU").ToList();
                    if (dictDatas != null && dictDatas.Count > 0)
                    {
                        var isUseInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "IsUseDistanceToGU");
                        if (isUseInfo != null && isUseInfo.PD_Value.ToLower() == "true")
                        {
                            for (int i = 0; i < codes.Count; i++)
                            {
                                if (codes[i] != "-")
                                {
                                    switch (i)
                                    {
                                        case 1:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS1))
                                            {
                                                codes[i] = _temp_GU.Code_MS1;
                                                brandMS1 = _temp_GU.Brand_MS1;
                                            }
                                            break;
                                        case 2:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS1))
                                            {
                                                codes[i] = _temp_GU.Code_LS1;
                                                brandLS1 = _temp_GU.Brand_LS1;
                                            }
                                            break;
                                        case 3:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS2))
                                            {
                                                codes[i] = _temp_GU.Code_MS2;
                                                brandMS2 = _temp_GU.Brand_MS2;
                                            }
                                            break;
                                        case 4:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS2))
                                            {
                                                codes[i] = _temp_GU.Code_LS2;
                                                brandLS2 = _temp_GU.Brand_LS2;
                                            }
                                            break;
                                        case 5:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS3))
                                            {
                                                codes[i] = _temp_GU.Code_MS3;
                                                brandMS3 = _temp_GU.Brand_MS3;
                                            }
                                            break;
                                        case 6:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS3))
                                            {
                                                codes[i] = _temp_GU.Code_LS3;
                                                brandLS3 = _temp_GU.Brand_LS3;
                                            }
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }

                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                code = string.Join(".", codes.Where(it => it != "-"));
                                codeAll = string.Join(".", codes);
                            }
                            else
                            {
                                code = string.Join("", codes.Where(it => it != "-"));
                                codeAll = string.Join("", codes);
                            }
                        }
                    }


                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperLS0--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                    logger.Info($"面纸换材，实际材质{codeAll},BrandLS0{stateInfo_ls0.BrandLS0},brandMS1{brandMS1},brandLS1{brandLS1},brandMS2{brandMS2},brandLS2{brandLS2},brandMS3{brandMS3},brandLS3{brandLS3}", module);
                    //面纸包角
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapLS0,
                        Code = stateInfo_ls0.CurCode + "/" + stateInfo_ls0.CodeALl,
                        Width = stateInfo_ls0.CurWidth,
                        Flute = stateInfo_ls0.CurFlute,
                        LastCode = stateInfo_ls0.LastCode,
                        LastWidth = stateInfo_ls0.LastWidth,
                        LastFlute = stateInfo_ls0.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0
                    });
                    //面纸张力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.TensionLS0,
                        Code = stateInfo_ls0.CurCode + "/" + stateInfo_gu.CurCode,
                        Width = stateInfo_ls0.CurWidth,
                        Flute = stateInfo_ls0.CurFlute,
                        LastCode = stateInfo_ls0.LastCode,
                        LastWidth = stateInfo_ls0.LastWidth,
                        LastFlute = stateInfo_ls0.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0
                    });
                    //糊机糊间隙
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.GlueGu,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机包角
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机包角
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu_Add2,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机天桥张力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.BridgeTension,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //热板压力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.HotPress,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                    //冷板压力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.CodePress,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                    //压板组数
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.PressGroupQty,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                }
                else
                {
                    _temp_GU.Code_LS0 = "";
                    _temp_GU.Brand_LS0 = "";
                    //正常换卷换材
                    List<string> codes = new List<string>();
                    string code = "";
                    string codeAll = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        code = string.Join(".", codes.Where(it => it != "-"));
                        codeAll = stateInfo_gu.CurCode;
                    }
                    else
                    {
                        codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                        code = stateInfo_gu.CurCode.Replace("-", "");
                        codeAll = stateInfo_gu.CurCode;
                    }

                    string brandMS1 = stateInfo_ms1.BrandMS1;
                    string brandLS1 = stateInfo_ls1.BrandLS1;
                    string brandMS2 = stateInfo_ms2.BrandMS2;
                    string brandLS2 = stateInfo_ls2.BrandLS2;
                    string brandMS3 = stateInfo_ms3.BrandMS3;
                    string brandLS3 = stateInfo_ls3.BrandLS3;

                    var dictDatas = BLLFactory<DictDataInfoManager>.Instance.Context.Queryable<DictDataInfo>()
                    .LeftJoin<DictTypeInfo>((data, type) => data.PD_TypeID == type.PD_ID).Where((data, type) => type.PD_Code == "DistanceToGU").ToList();
                    if (dictDatas != null && dictDatas.Count > 0)
                    {
                        var isUseInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "IsUseDistanceToGU");
                        if (isUseInfo != null && isUseInfo.PD_Value.ToLower() == "true")
                        {
                            for (int i = 0; i < codes.Count; i++)
                            {
                                if (codes[i] != "-")
                                {
                                    switch (i)
                                    {
                                        case 1:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS1))
                                            {
                                                codes[i] = _temp_GU.Code_MS1;
                                                brandMS1 = _temp_GU.Brand_MS1;
                                            }
                                            break;
                                        case 2:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS1))
                                            {
                                                codes[i] = _temp_GU.Code_LS1;
                                                brandLS1 = _temp_GU.Brand_LS1;
                                            }
                                            break;
                                        case 3:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS2))
                                            {
                                                codes[i] = _temp_GU.Code_MS2;
                                                brandMS2 = _temp_GU.Brand_MS2;
                                            }
                                            break;
                                        case 4:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS2))
                                            {
                                                codes[i] = _temp_GU.Code_LS2;
                                                brandLS2 = _temp_GU.Brand_LS2;
                                            }
                                            break;
                                        case 5:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_MS3))
                                            {
                                                codes[i] = _temp_GU.Code_MS3;
                                                brandMS3 = _temp_GU.Brand_MS3;
                                            }
                                            break;
                                        case 6:
                                            if (!string.IsNullOrEmpty(_temp_GU.Code_LS3))
                                            {
                                                codes[i] = _temp_GU.Code_LS3;
                                                brandLS3 = _temp_GU.Brand_LS3;
                                            }
                                            break;
                                        default:
                                            break;
                                    }
                                }
                            }

                            if (stateInfo_gu.CurCode.Contains("."))
                            {
                                code = string.Join(".", codes.Where(it => it != "-"));
                                codeAll = string.Join(".", codes);
                            }
                            else
                            {
                                code = string.Join("", codes.Where(it => it != "-"));
                                codeAll = string.Join("", codes);
                            }

                        }
                    }

                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperLS0--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    logger.Info($"面纸换材，实际材质{codeAll},BrandLS0{stateInfo_ls0.BrandLS0},brandMS1{brandMS1},brandLS1{brandLS1},brandMS2{brandMS2},brandLS2{brandLS2},brandMS3{brandMS3},brandLS3{brandLS3}", module);

                    //面纸包角
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapLS0,
                        Code = stateInfo_ls0.CurCode + "/" + stateInfo_ls0.CodeALl,
                        Width = stateInfo_ls0.CurWidth,
                        Flute = stateInfo_ls0.CurFlute,
                        LastCode = stateInfo_ls0.LastCode,
                        LastWidth = stateInfo_ls0.LastWidth,
                        LastFlute = stateInfo_ls0.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0
                    });
                    //面纸张力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.TensionLS0,
                        Code = stateInfo_ls0.CurCode + "/" + stateInfo_ls0.CodeALl,
                        Width = stateInfo_ls0.CurWidth,
                        Flute = stateInfo_ls0.CurFlute,
                        LastCode = stateInfo_ls0.LastCode,
                        LastWidth = stateInfo_ls0.LastWidth,
                        LastFlute = stateInfo_ls0.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0
                    });
                    //糊机糊间隙
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.GlueGu,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机包角
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机包角 附加烘缸
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.WrapGu_Add2,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //糊机天桥张力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.BridgeTension,
                        Code = codeAll,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute,
                        BrandLS0 = stateInfo_ls0.BrandLS0,
                        BrandMS1 = brandMS1,
                        BrandLS1 = brandLS1,
                        BrandMS2 = brandMS2,
                        BrandLS2 = brandLS2,
                        BrandMS3 = brandMS3,
                        BrandLS3 = brandLS3
                    });
                    //热板压力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.HotPress,
                        Code = stateInfo_gu.CurCode,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                    //冷板压力
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.CodePress,
                        Code = stateInfo_gu.CurCode,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                    //压板组数
                    list.Add(new PublishInfo
                    {
                        Part = IPSHandlePart.PressGroupQty,
                        Code = stateInfo_gu.CurCode,
                        Width = stateInfo_gu.CurWidth,
                        Flute = stateInfo_gu.CurFlute,
                        LastCode = stateInfo_gu.LastCode,
                        LastWidth = stateInfo_gu.LastWidth,
                        LastFlute = stateInfo_gu.LastFlute
                    });
                }
                if (isReal)
                {
                    logger.Info($"HandleChangePaperLS0 准备发送LS0换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperLS0 准备发送LS0换材消息(正常换卷换材)，通知各执行类", module);
                }
                Publish(list);

            }
            catch (Exception ex)
            {

                logger.Error($"HandleChangePaperLS0 执行过程中异常报错：{ex.Message}", module);
            }
        }

        /// <summary>
        /// 1芯接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperMS1(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成(按照糊机材质部位拼实际材质)
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperMS1--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_MS1 = "";
                    _temp_GU.Code_MS1 = "";
                }

                //以当前的真实材质 芯纸+里纸 到SFqdm表中找记录，没有的话就生成
                if (stateInfo_ms1.CurCode == "-" || stateInfo_ls1.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ms1.CurFlute))
                {
                    logger.Warn($"HandleChangePaperMS1--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms1.CurCode},ls={stateInfo_ls1.CurCode},flute={stateInfo_ms1.CurFlute}", module);
                    return;
                }
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms1.CurCode, stateInfo_ls1.CurCode, stateInfo_ms1.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF1糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode + "/" + stateInfo_ls1.CurCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //MS1包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //MS1ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS1_Ext,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //LS1包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //LS1ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS1_Ext,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //SF1压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode + "/" + stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute
                });
                //SF1瓦楞辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.CorrugatedRollSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute
                });
                //SF1真空泵重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute
                });
                //MS1接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionMS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //SF1热喷雾
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.HotSpraySF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //SF1真空泵重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                if (isReal)
                {
                    logger.Info($"HandleChangePaperMS1 准备发送MS1换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperMS1 准备发送MS1换材消息(正常换卷换材)，通知各执行类", module);
                }
                Publish(list);
            }
            catch (Exception ex)
            {

                logger.Error($"HandleChangePaperMS1 执行过程中异常报错：{ex.Message}", module);
            }
        }

        /// <summary>
        /// 1里接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperLS1(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperLS1--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_LS1 = "";
                    _temp_GU.Code_LS1 = "";
                }

                if (stateInfo_ms1.CurCode == "-" || stateInfo_ls1.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ls1.CurFlute))
                {
                    logger.Warn($"HandleChangePaperLS1--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms1.CurCode},ls={stateInfo_ls1.CurCode},flute={stateInfo_ls1.CurFlute}", module);
                    return;
                }
                //以当前的真实材质 芯纸+里纸 到SFqdm表中找记录，没有的话就生成
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms1.CurCode, stateInfo_ls1.CurCode, stateInfo_ls1.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF1糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //MS1包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //MS1ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS1_Ext,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //LS1包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //LS1ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS1_Ext,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });
                //SF1压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ms1.CurWidth,
                    Flute = stateInfo_ms1.CurFlute,
                    LastCode = stateInfo_ms1.LastCode + "/" + stateInfo_ms1.LastCode,
                    LastWidth = stateInfo_ms1.LastWidth,
                    LastFlute = stateInfo_ms1.LastFlute
                });
                //SF1瓦楞辊压力重新赋值
                //LS1接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionLS1,
                    Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode,
                    Width = stateInfo_ls1.CurWidth,
                    Flute = stateInfo_ls1.CurFlute,
                    LastCode = stateInfo_ls1.LastCode,
                    LastWidth = stateInfo_ls1.LastWidth,
                    LastFlute = stateInfo_ls1.LastFlute,
                    BrandMS1 = stateInfo_ms1.BrandMS1,
                    BrandLS1 = stateInfo_ls1.BrandLS1
                });

                if (isReal)
                {
                    logger.Info($"HandleChangePaperLS1 准备发送LS1换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperLS1 准备发送LS1换材消息(正常换卷换材)，通知各执行类", module);
                }

                Publish(list);
            }
            catch (Exception ex)
            {

                logger.Error($"HandleChangePaperLS1 执行过程中异常报错：{ex.Message}", module);
            }

        }

        /// <summary>
        /// 2芯接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperMS2(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperMS2--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_MS2 = "";
                    _temp_GU.Code_MS2 = "";
                }
                if (stateInfo_ms2.CurCode == "-" || stateInfo_ls2.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ms2.CurFlute))
                {
                    logger.Warn($"HandleChangePaperMS2--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms2.CurCode},ls={stateInfo_ls2.CurCode},flute={stateInfo_ms2.CurFlute}", module);
                    return;
                }
                // 以当前的真实材质 芯纸 + 里纸 到SFqdm表中找记录，没有的话就生成
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms2.CurCode, stateInfo_ls2.CurCode, stateInfo_ms2.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF2糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode + "/" + stateInfo_ls2.CurCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //MS2包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //MS2ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS2_Ext,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //LS2包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //LS2包角ext重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS2_Ext,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //SF2压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode + "/" + stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute
                });
                //SF2瓦楞辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.CorrugatedRollSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute
                });
                //SF2真空泵重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute
                });
                //MS2接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionMS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //SF2热喷雾
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.HotSpraySF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //SF2真空泵重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                if (isReal)
                {
                    logger.Info($"HandleChangePaperMS2 准备发送MS2换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperMS2 准备发送MS2换材消息(正常换卷换材)，通知各执行类", module);
                }

                Publish(list);
            }
            catch (Exception ex)
            {
                logger.Error($"HandleChangePaperMS2 执行过程中异常报错：{ex.Message}", module);
            }

        }

        /// <summary>
        /// 2里接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperLS2(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperLS2--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_LS2 = "";
                    _temp_GU.Code_LS2 = "";
                }

                if (stateInfo_ms2.CurCode == "-" || stateInfo_ls2.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ls2.CurFlute))
                {
                    logger.Warn($"HandleChangePaperLS2--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms2.CurCode},ls={stateInfo_ls2.CurCode},flute={stateInfo_ls2.CurFlute}", module);
                    return;
                }
                //以当前的真实材质 芯纸+里纸 到SFqdm表中找记录，没有的话就生成
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms2.CurCode, stateInfo_ls2.CurCode, stateInfo_ls2.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF2糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //MS2包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //MS2ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS2_Ext,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //LS2包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //LS2包角ext重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS2_Ext,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });
                //SF2压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ms2.CurWidth,
                    Flute = stateInfo_ms2.CurFlute,
                    LastCode = stateInfo_ms2.LastCode + "/" + stateInfo_ms2.LastCode,
                    LastWidth = stateInfo_ms2.LastWidth,
                    LastFlute = stateInfo_ms2.LastFlute
                });
                //SF2瓦楞辊压力重新赋值
                //LS2接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionLS2,
                    Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode,
                    Width = stateInfo_ls2.CurWidth,
                    Flute = stateInfo_ls2.CurFlute,
                    LastCode = stateInfo_ls2.LastCode,
                    LastWidth = stateInfo_ls2.LastWidth,
                    LastFlute = stateInfo_ls2.LastFlute,
                    BrandMS2 = stateInfo_ms2.BrandMS2,
                    BrandLS2 = stateInfo_ls2.BrandLS2
                });

                if (isReal)
                {
                    logger.Info($"HandleChangePaperLS2 准备发送LS2换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperLS2 准备发送LS2换材消息(正常换卷换材)，通知各执行类", module);
                }

                Publish(list);
            }
            catch (Exception ex)
            {
                logger.Error($"HandleChangePaperLS2 执行过程中异常报错：{ex.Message}", module);
            }
        }

        /// <summary>
        /// 3芯接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperMS3(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperMS3--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_MS3 = "";
                    _temp_GU.Code_MS3 = "";
                }

                if (stateInfo_ms3.CurCode == "-" || stateInfo_ls3.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ms3.CurFlute))
                {
                    logger.Warn($"HandleChangePaperMS3--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms3.CurCode},ls={stateInfo_ls3.CurCode},flute={stateInfo_ms3.CurFlute}", module);
                    return;
                }
                //以当前的真实材质 芯纸 + 里纸 到SFqdm表中找记录，没有的话就生成
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms3.CurCode, stateInfo_ls3.CurCode, stateInfo_ms3.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF3糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode + "/" + stateInfo_ls3.CurCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //MS3包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //MS3ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS3_Ext,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //LS3包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //LS3包角ext重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS3_Ext,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //SF3压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode + "/" + stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute
                });
                //SF3瓦楞辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.CorrugatedRollSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute
                });
                //SF3瓦楞辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute
                });
                //MS3接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionMS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //SF3热喷雾
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.HotSpraySF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //SF3真空泵重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.VacuumBlowerSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });

                if (isReal)
                {
                    logger.Info($"HandleChangePaperMS3 准备发送MS3换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperMS3 准备发送MS3换材消息(正常换卷换材)，通知各执行类", module);
                }

                Publish(list);
            }
            catch (Exception ex)
            {
                logger.Error($"HandleChangePaperMS3 执行过程中异常报错：{ex.Message}", module);
            }

        }

        /// <summary>
        /// 3里接纸机换材处理函数
        /// </summary>
        private void HandleChangePaperLS3(bool isReal = false)
        {
            try
            {
                if (isReal)
                {
                    #region 获取到真实材质的时候，把每个部位的当前材质拼起来，从DF QDM里面找一下，没有就生成
                    //检查是否有历史QDM DF系数，没有则生成
                    List<string> codes = new List<string>();
                    string code = "";
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        codes = stateInfo_gu.CurCode.Split('.').ToList();
                        for (int i = 0; i < codes.Count; i++)
                        {
                            if (codes[i] == "-")
                                continue;
                            switch (i)
                            {
                                case 0:
                                    codes[i] = stateInfo_ls0.CurCode;
                                    break;
                                case 1:
                                    codes[i] = stateInfo_ms1.CurCode;
                                    break;
                                case 2:
                                    codes[i] = stateInfo_ls1.CurCode;
                                    break;
                                case 3:
                                    codes[i] = stateInfo_ms2.CurCode;
                                    break;
                                case 4:
                                    codes[i] = stateInfo_ls2.CurCode;
                                    break;
                                case 5:
                                    codes[i] = stateInfo_ms3.CurCode;
                                    break;
                                case 6:
                                    codes[i] = stateInfo_ls3.CurCode;
                                    break;
                                default:
                                    break;
                            }
                        }
                        code = string.Join(".", codes.Where(it => it != "-"));
                    }
                    else
                    {
                        var chars = stateInfo_gu.CurCode.ToCharArray();
                        for (int i = 0; i < chars.Length; i++)
                        {
                            if (chars[i] != '-')
                            {
                                switch (i)
                                {
                                    case 0:
                                        codes.Add(stateInfo_ls0.CurCode);
                                        break;
                                    case 1:
                                        codes.Add(stateInfo_ms1.CurCode);
                                        break;
                                    case 2:
                                        codes.Add(stateInfo_ls1.CurCode);
                                        break;
                                    case 3:
                                        codes.Add(stateInfo_ms2.CurCode);
                                        break;
                                    case 4:
                                        codes.Add(stateInfo_ls2.CurCode);
                                        break;
                                    case 5:
                                        codes.Add(stateInfo_ms3.CurCode);
                                        break;
                                    case 6:
                                        codes.Add(stateInfo_ls3.CurCode);
                                        break;
                                    default:
                                        break;
                                }
                            }
                        }
                        code = string.Join("", codes.Where(it => it != "-"));
                    }
                    //这边的作用在于判断表内是否有记录，没有则生成，统一一个地方生成，避免多个函数都生成QDM系数造成资源竞争和重复的问题
                    if (codes.Where(it => it != "-").Count() != stateInfo_gu.CurFlute.Substring(0, 1).ToInt32())
                    {
                        logger.Warn($"HandleChangePaperLS3--QdmCtrl.GetQdmDFCoef 无法创建，任务跳出，因为code={code},flute = {stateInfo_gu.CurFlute}", module);
                        return;
                    }
                    QdmCtrl.GetQdmDFCoef(code, stateInfo_gu.CurFlute);
                    #endregion
                }
                else
                {
                    _temp_GU.Brand_LS3 = "";
                    _temp_GU.Code_MS3 = "";
                }

                if (stateInfo_ms3.CurCode == "-" || stateInfo_ls3.CurCode == "-" || string.IsNullOrEmpty(stateInfo_ls3.CurFlute))
                {
                    logger.Warn($"HandleChangePaperLS3--QdmCtrl.GetQdmCoefSFInfo,参数有误：ms={stateInfo_ms3.CurCode},ls={stateInfo_ls3.CurCode},flute={stateInfo_ls3.CurFlute}", module);
                    return;
                }
                //以当前的真实材质 芯纸+里纸 到SFqdm表中找记录，没有的话就生成
                QdmCtrl.GetQdmCoefSFInfo(stateInfo_ms3.CurCode, stateInfo_ls3.CurCode, stateInfo_ls3.CurFlute);
                List<PublishInfo> list = new List<PublishInfo>();
                //SF3糊间隙重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.GlueSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //MS3包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //MS3ext包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapMS3_Ext,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //LS3包角重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //LS3包角ext重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.WrapLS3_Ext,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });
                //SF3压力辊压力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.PressRollSF3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ms3.CurWidth,
                    Flute = stateInfo_ms3.CurFlute,
                    LastCode = stateInfo_ms3.LastCode + "/" + stateInfo_ms3.LastCode,
                    LastWidth = stateInfo_ms3.LastWidth,
                    LastFlute = stateInfo_ms3.LastFlute
                });
                //SF3瓦楞辊压力重新赋值
                //LS3接纸机张力重新赋值
                list.Add(new PublishInfo
                {
                    Part = IPSHandlePart.TensionLS3,
                    Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode,
                    Width = stateInfo_ls3.CurWidth,
                    Flute = stateInfo_ls3.CurFlute,
                    LastCode = stateInfo_ls3.LastCode,
                    LastWidth = stateInfo_ls3.LastWidth,
                    LastFlute = stateInfo_ls3.LastFlute,
                    BrandMS3 = stateInfo_ms3.BrandMS3,
                    BrandLS3 = stateInfo_ls3.BrandLS3
                });

                if (isReal)
                {
                    logger.Info($"HandleChangePaperLS3 准备发送LS3换材消息(取到数据库实际材质)，通知各执行类", module);
                }
                else
                {
                    logger.Info($"HandleChangePaperLS3 准备发送LS3换材消息(正常换卷换材)，通知各执行类", module);
                }

                Publish(list);
            }
            catch (Exception ex)
            {
                logger.Error($"HandleChangePaperLS3 执行过程中异常报错：{ex.Message}", module);
            }
        }

        /// <summary>
        /// 得到接纸机楞型
        /// </summary>
        /// <param name="info"></param>
        /// <param name="paperCode">全材质</param>
        /// <param name="flute">订单楞型</param>
        /// <param name="machineID">接纸机编号</param>
        private void GetSPFlute(ref DriveStateInfo info, string paperCode, string flute, string machineID)
        {
            try
            {
                info.LastFlute = info.CurFlute;
                List<string> papers = new List<string>();
                if (paperCode.Contains("."))
                {
                    papers = paperCode.Split('.').ToList();
                }
                else
                {
                    foreach (var c in paperCode.ToCharArray())
                    {
                        papers.Add(c.ToString());
                    }
                }
                int index = 1;
                for (int i = 0; i < papers.Count; i++)
                {
                    switch (i)
                    {
                        case 1:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS1" || machineID == "LS1")
                                {
                                    info.CurFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        case 3:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS2" || machineID == "LS2")
                                {
                                    info.CurFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        case 5:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS3" || machineID == "LS3")
                                {
                                    info.CurFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
            catch (Exception ex)
            {

                throw ex;
            }

        }

        /// <summary>
        /// 实时发送Ips赋值的值和QDM系数以及界面系数
        /// </summary>
        private async Task SendIpsSetValue()
        {
            while (true)
            {
                try
                {
                    SendIpsValueInfo info = new SendIpsValueInfo
                    {
                        Data = new List<IpsValueInfo>()
                    };

                    GlobalControl.ipsValueInfos.ForEach(it =>
                    {

                        if (it.Position == IpsDriverPositionEnum.GlueGU1
                            || it.Position == IpsDriverPositionEnum.GlueGU2
                            || it.Position == IpsDriverPositionEnum.GlueGU2
                            || it.Position == IpsDriverPositionEnum.GlueSF1 || it.Position == IpsDriverPositionEnum.GlueSF2 || it.Position == IpsDriverPositionEnum.GlueSF2)
                        {
                            it.OffSetValue = it.OffSetValue + it.BrandOffSetValue;
                        }

                        info.Data.Add(it);
                    });

                    if (info.Data.Count > 0)
                    {
                        await GlobalInfos.SendMsg("I001", JsonConvert.SerializeObject(info));
                    }

                }
                catch (Exception ex)
                {
                    logger.Error($"发送IPS相关设置数据异常出错：{ex.Message}");
                }
                finally
                {
                    await Task.Delay(500);
                }
            }
        }


        /// <summary>
        /// 发送蒸汽
        /// </summary>
        /// <returns></returns>
        private async Task SendSteamValue()
        {
            while (true)
            {
                try
                {
                    List<SteamToClientModel> steamToClientModels = GlobalControl.ipsSteamValueInfos.Values.ToList();
                    if (steamToClientModels.Count > 0)
                    {
                        string strMsg = JsonConvert.SerializeObject(steamToClientModels.AsReadOnly());
                        await GlobalInfos.SendMsg("M121", strMsg);
                    }

                }
                catch (Exception ex)
                {
                    logger.Error($"SteamBiz-SendM121方法出错{ex.StackTrace}");
                }
                finally
                {
                    await Task.Delay(500);
                }
            }

        }

        /// <summary>
        /// 重新初始化内存变量，重新给各部位点位赋值
        /// </summary>
        public void Refresh()
        {
            try
            {
                InitInfos();
                _temp_GU = new GuRealInfo();
                HandleFirstAll();
                logger.Info("Refresh--重新初始化内存变量，重新给各部位点位赋值,通知各执行类完成", module);
            }
            catch (Exception ex)
            {
                logger.Error($"Refresh--重新初始化赋点位异常失败：{ex.Message}", module);
            }

        }

        /// <summary>
        /// 基础设置改变或者QDM系数改变，立即生效
        /// </summary>
        public void ChangeBaseDataToReSet()
        {
            HandleFirstAll();
            logger.Info("ChangeBaseDataToReSet--基础设置改变或者QDM系数改变，通知各执行类重新计算点位值完毕", module);
        }


        /// <summary>
        /// 前台蒸汽点位调整
        /// </summary>
        /// <param name="message"></param>
        public void HandleM120Task(string message)
        {
            List<PubChangeNowInfo> publishInfos = new List<PubChangeNowInfo>();
            List<string> paperCodes = new List<string>();
            if (GlobalControl.curOrder.WO_PaperCode.Contains("."))
            {
                paperCodes = GlobalControl.curOrder.WO_PaperCode.Split(".").ToList();
            }
            else
            {
                paperCodes = GlobalControl.curOrder.WO_PaperCode.ToCharArray().Select(a => a.ToString()).ToList();
            }

            if (!string.IsNullOrWhiteSpace(message))
            {
                PubChangeNowInfo publishInfo = new PubChangeNowInfo();
                var formFactor = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(a => 1 == 1);
                switch (message)
                {
                    case "F_PreheaterSteam_SF1_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamSF1;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.SF1Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_SF1_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);
                        break;
                    case "F_PreheaterSteam_SF2_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamSF2;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.SF2Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_SF2_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);
                        break;
                    case "F_PreheaterSteam_SF3_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamSF3;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.SF3Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_SF3_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);
                        break;
                    case "F_PreheaterSteam_GU_0_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamHPH;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.HPH0Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_GU_0_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_PreheaterSteam_GU_1st_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamHPH;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.HPH1Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_GU_1st_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_PreheaterSteam_GU_2nd_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamHPH;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.HPH2Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_GU_2nd_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_PreheaterSteam_GU_3rd_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamHPH;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.HPH3Steam, new IpsValueInfo() { FormCoef = formFactor.F_PreheaterSteam_GU_3rd_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_HotPlateSteam_DF_1Part_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamDF;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.DFSteamPart1, new IpsValueInfo() { FormCoef = formFactor.F_HotPlateSteam_DF_1Part_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_HotPlateSteam_DF_2Part_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamDF;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.DFSteamPart2, new IpsValueInfo() { FormCoef = formFactor.F_HotPlateSteam_DF_2Part_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    case "F_HotPlateSteam_DF_3Part_Form_Factor":
                        publishInfo.Part = IPSHandlePart.SteamDF;
                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.DFSteamPart3, new IpsValueInfo() { FormCoef = formFactor.F_HotPlateSteam_DF_3Part_Form_Factor, RealQdmCoef = 1, FormQdmCoef = 1 }, string.Join(".", paperCodes), GlobalControl.curOrder.WO_Wave);

                        break;
                    default:
                        break;
                }

                PubChangeNow(publishInfo);
            }
            else
            {
                publishInfos.Add(new PubChangeNowInfo() { Part = IPSHandlePart.SteamDF });
                publishInfos.Add(new PubChangeNowInfo() { Part = IPSHandlePart.SteamHPH });
                publishInfos.Add(new PubChangeNowInfo() { Part = IPSHandlePart.SteamSF1 });
                publishInfos.Add(new PubChangeNowInfo() { Part = IPSHandlePart.SteamSF2 });
                publishInfos.Add(new PubChangeNowInfo() { Part = IPSHandlePart.SteamSF3 });

                foreach (var item in publishInfos)
                {
                    PubChangeNow(item);
                }
            }


        }

        ///// <summary>
        ///// 弯翘调整细节处理
        ///// </summary>
        ///// <param name="floor">当前订单层数</param>
        ///// <param name="cmd">上弯还是下弯命令
        ///// UP1----上弯轻
        ///// UP2----上弯中
        ///// UP3----上弯重
        ///// DOWN1---下弯轻
        ///// DOWN2---下弯中
        ///// DOWN3---下弯重
        ///// RESET_UP1----上弯轻复位
        ///// RESET_UP2----上弯中复位
        ///// RESET_UP3----上弯重复位
        ///// RESET_DOWN1---下弯轻复位
        ///// RESET_DOWN2---下弯中复位
        ///// RESET_DOWN3---下弯重复位
        ///// </param>
        ///// <param name="curvedWarpSetInfos">弯翘调整设置集合</param>
        ///// <param name="isRest">是否复位标识</param>
        ///// </summary>

        //private void HandleCurvedWarpDetail(int floor, string cmd, List<WarpedWrapInfo> curvedWarpSetInfos, bool isRest)
        //{
        //    //包角调整
        //    //压板组数调整
        //    //冷板压力调整
        //    decimal value = 0;
        //    PubChangeNowInfo pubInfo = null;
        //    switch (floor)
        //    {
        //        case 3:
        //            if (cmd == "UP1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpLight3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpModerate3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpSevere3rd ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            break;
        //        case 5:
        //            if (cmd == "UP1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2")?.F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpLight5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpModerate5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpSevere5ve ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            break;
        //        case 7:
        //            if (cmd == "UP1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "UP3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_UpWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN1")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpLight7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN2")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpModerate7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            else if (cmd == "DOWN3")
        //            {
        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS1").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS1;
        //                    pubInfo.Width = stateInfo_ms1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms1.CurFlute;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS1").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS1;
        //                    pubInfo.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
        //                    pubInfo.Width = stateInfo_ls1.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls1.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls1.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls1.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls1.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS2").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS2;
        //                    pubInfo.Width = stateInfo_ms2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms2.CurFlute;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS2").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS2;
        //                    pubInfo.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
        //                    pubInfo.Width = stateInfo_ls2.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls2.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls2.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls2.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls2.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "MS3").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapMS3;
        //                    pubInfo.Width = stateInfo_ms3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ms3.CurFlute;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.LastCode = stateInfo_ms3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ms3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ms3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }

        //                if (curvedWarpSetInfos.Exists(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3"))
        //                {
        //                    value = curvedWarpSetInfos.FirstOrDefault(it => it.F_Equipment == "WrapInfo" && it.F_Position == "LS3").F_DownWarpSevere7en ?? 0;
        //                    pubInfo = new PubChangeNowInfo();
        //                    pubInfo.Part = IPSHandlePart.WrapLS3;
        //                    pubInfo.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
        //                    pubInfo.Width = stateInfo_ls3.CurWidth;
        //                    pubInfo.Flute = stateInfo_ls3.CurFlute;
        //                    pubInfo.LastCode = stateInfo_ls3.LastCode;
        //                    pubInfo.LastWidth = stateInfo_ls3.LastWidth;
        //                    pubInfo.LastFlute = stateInfo_ls3.LastFlute;
        //                    if (isRest)
        //                    {
        //                        value = 0;
        //                    }
        //                    pubInfo.OffSetValue = value;
        //                    PubChangeNow(pubInfo);
        //                }
        //            }
        //            break;
        //        default:
        //            break;
        //    }
        //}

        ///// <summary>
        ///// 弯翘调整
        ///// </summary>
        ///// <param name="msg">
        ///// 调整内容 
        ///// UP1----上弯轻
        ///// UP2----上弯中
        ///// UP3----上弯重
        ///// DOWN1---下弯轻
        ///// DOWN2---下弯中
        ///// DOWN3---下弯重
        ///// RESET_UP1----上弯轻复位
        ///// RESET_UP2----上弯中复位
        ///// RESET_UP3----上弯重复位
        ///// RESET_DOWN1---下弯轻复位
        ///// RESET_DOWN2---下弯中复位
        ///// RESET_DOWN3---下弯重复位
        ///// </param>
        //public void HandleCurvedWarp(string msg)
        //{
        //    //获取弯翘调整设置
        //    //按照设置得到需要调整的项目
        //    //发布调整消息，带入偏移量，此类消息当业务处理类订阅到之后立刻执行，没有延迟处理
        //    string cmd = msg.ToUpper();
        //    //查询弯翘调整项目
        //    List<WarpedWrapInfo> curvedWarpSetInfos = BLLFactory<WarpedWrapInfoManager>.Instance.GetList();
        //    //当前正在生产的订单
        //    var curOrderInfo = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();
        //    //判断是几层
        //    int floor = curOrderInfo.WO_Wave.Substring(0, 1).ToInt16();
        //    if (cmd.Contains("RESET"))
        //    {
        //        string spl = cmd.Split('_')[1];

        //        HandleCurvedWarpDetail(floor, spl, curvedWarpSetInfos, true);
        //    }
        //    else
        //    {
        //        HandleCurvedWarpDetail(floor, cmd, curvedWarpSetInfos, false);
        //    }
        //}


        /// <summary>
        /// 客户端界面系数或开关变化处理函数
        /// </summary>
        /// <param name="msg">客户端发送的消息</param>
        public void HandleM107(string msg)
        {
            try
            {
                string guCode = stateInfo_gu.CurCode;
                var dictDatas = BLLFactory<DictDataInfoManager>.Instance.Context.Queryable<DictDataInfo>()
               .LeftJoin<DictTypeInfo>((d, t) => d.PD_TypeID == t.PD_ID).Where((d, t) => t.PD_Code == "DistanceToGU").ToList();
                var isUseInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "IsUseDistanceToGU");
                if (isUseInfo != null && isUseInfo.PD_Value.ToLower() == "true")
                {
                    string joinStr = "";
                    List<string> allCodes = new List<string>();
                    if (stateInfo_gu.CurCode.Contains("."))
                    {
                        joinStr = ".";

                        allCodes = stateInfo_gu.CurCode.Split('.').ToList();
                    }
                    else
                    {
                        allCodes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                    }
                    for (int i = 0; i < allCodes.Count; i++)
                    {
                        switch (i)
                        {
                            case 0:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS0))
                                {
                                    allCodes[i] = _temp_GU.Code_LS0;
                                }
                                break;
                            case 1:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS1))
                                {
                                    allCodes[i] = _temp_GU.Code_MS1;
                                }
                                break;
                            case 2:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS1))
                                {
                                    allCodes[i] = _temp_GU.Code_LS1;
                                }
                                break;

                            case 3:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS2))
                                {
                                    allCodes[i] = _temp_GU.Code_MS2;
                                }
                                break;
                            case 4:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS2))
                                {
                                    allCodes[i] = _temp_GU.Code_LS2;
                                }
                                break;

                            case 5:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_MS3))
                                {
                                    allCodes[i] = _temp_GU.Code_MS3;
                                }
                                break;
                            case 6:
                                if (allCodes[i] != "-" && !string.IsNullOrEmpty(_temp_GU.Code_LS3))
                                {
                                    allCodes[i] = _temp_GU.Code_LS3;
                                }
                                break;
                            default:
                                break;
                        }
                    }

                    guCode = string.Join(joinStr, allCodes);
                }


                if (string.IsNullOrEmpty(msg))
                {
                    //用户勾选糊机糊间隙设备部位和糊机包角设备部位
                    logger.Info("HandleM107-用户勾选糊机糊间隙设备部位和糊机包角设备部位,准备通知改变糊机糊间隙和糊机包角", module);
                    //对糊机糊间隙和糊机包角重新赋值
                    PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                    infoGlue.OffSetValue = 0;
                    infoGlue.Part = IPSHandlePart.GlueGu;
                    infoGlue.Width = stateInfo_gu.CurWidth;
                    infoGlue.Flute = stateInfo_gu.CurFlute;
                    //infoGlue.Code = stateInfo_gu.CurCode;
                    infoGlue.Code = guCode;
                    infoGlue.LastCode = stateInfo_gu.LastCode;
                    infoGlue.LastWidth = stateInfo_gu.LastWidth;
                    infoGlue.LastFlute = stateInfo_gu.LastFlute;
                    infoGlue.BrandLS0 = _temp_GU.Brand_LS0;
                    infoGlue.BrandMS1 = _temp_GU.Brand_MS1;
                    infoGlue.BrandLS1 = _temp_GU.Brand_LS1;
                    infoGlue.BrandMS2 = _temp_GU.Brand_MS2;
                    infoGlue.BrandLS2 = _temp_GU.Brand_LS2;
                    infoGlue.BrandMS3 = _temp_GU.Brand_MS3;
                    infoGlue.BrandLS3 = _temp_GU.Brand_LS3;
                    PubChangeNow(infoGlue);

                    PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                    infoWrap.OffSetValue = 0;
                    infoWrap.Part = IPSHandlePart.WrapGu;
                    infoWrap.Width = stateInfo_gu.CurWidth;
                    infoWrap.Flute = stateInfo_gu.CurFlute;
                    //infoWrap.Code = stateInfo_gu.CurCode;
                    infoWrap.Code = guCode;
                    infoWrap.LastCode = stateInfo_gu.LastCode;
                    infoWrap.LastWidth = stateInfo_gu.LastWidth;
                    infoWrap.LastFlute = stateInfo_gu.LastFlute;
                    infoWrap.BrandLS0 = _temp_GU.Brand_LS0;
                    infoWrap.BrandMS1 = _temp_GU.Brand_MS1;
                    infoWrap.BrandLS1 = _temp_GU.Brand_LS1;
                    infoWrap.BrandMS2 = _temp_GU.Brand_MS2;
                    infoWrap.BrandLS2 = _temp_GU.Brand_LS2;
                    infoWrap.BrandMS3 = _temp_GU.Brand_MS3;
                    infoWrap.BrandLS3 = _temp_GU.Brand_LS3;
                    PubChangeNow(infoWrap);

                    PubChangeNowInfo infoWrap_Ext = new PubChangeNowInfo();
                    infoWrap_Ext.OffSetValue = 0;
                    infoWrap_Ext.Part = IPSHandlePart.WrapGu_Add2;
                    infoWrap_Ext.Width = stateInfo_gu.CurWidth;
                    infoWrap_Ext.Flute = stateInfo_gu.CurFlute;
                    //infoWrap_Ext.Code = stateInfo_gu.CurCode;
                    infoWrap_Ext.Code = guCode;
                    infoWrap_Ext.LastCode = stateInfo_gu.LastCode;
                    infoWrap_Ext.LastWidth = stateInfo_gu.LastWidth;
                    infoWrap_Ext.LastFlute = stateInfo_gu.LastFlute;
                    infoWrap_Ext.BrandLS0 = _temp_GU.Brand_LS0;
                    infoWrap_Ext.BrandMS1 = _temp_GU.Brand_MS1;
                    infoWrap_Ext.BrandLS1 = _temp_GU.Brand_LS1;
                    infoWrap_Ext.BrandMS2 = _temp_GU.Brand_MS2;
                    infoWrap_Ext.BrandLS2 = _temp_GU.Brand_LS2;
                    infoWrap_Ext.BrandMS3 = _temp_GU.Brand_MS3;
                    infoWrap_Ext.BrandLS3 = _temp_GU.Brand_LS3;
                    PubChangeNow(infoWrap_Ext);
                }
                else
                {
                    //客户端修改界面系数以及开关变量
                    logger.Info($"HandleM107-用户修改系数或者开关，报文：{msg}", module);
                    //msg解析
                    WritePointValueModel info = JsonConvert.DeserializeObject<WritePointValueModel>(msg);
                    if (info != null)
                    {
                        if (info.VarIsOpen != null)
                        {
                            if (info.VarIsOpen == "FALSE")
                            {
                                return;
                            }
                        }

                        string type = info.VarType;
                        string position = info.VarPosition;
                        if (type == "GlueInfo")
                        {
                            if (position == "GU_1st" || position == "GU_2nd" || position == "GU_3rd")
                            {
                                //糊机糊间隙系数修改
                                PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                                infoGlue.OffSetValue = 0;
                                infoGlue.Part = IPSHandlePart.GlueGu;
                                infoGlue.Width = stateInfo_gu.CurWidth;
                                infoGlue.Flute = stateInfo_gu.CurFlute;
                                //infoGlue.Code = stateInfo_gu.CurCode;
                                infoGlue.Code = guCode;
                                infoGlue.LastCode = stateInfo_gu.LastCode;
                                infoGlue.LastWidth = stateInfo_gu.LastWidth;
                                infoGlue.LastFlute = stateInfo_gu.LastFlute;
                                infoGlue.BrandLS0 = stateInfo_ls0.BrandLS0;
                                infoGlue.BrandMS1 = _temp_GU.Brand_MS1;
                                infoGlue.BrandLS1 = _temp_GU.Brand_LS1;
                                infoGlue.BrandMS2 = _temp_GU.Brand_MS2;
                                infoGlue.BrandLS2 = _temp_GU.Brand_LS2;
                                infoGlue.BrandMS3 = _temp_GU.Brand_MS3;
                                infoGlue.BrandLS3 = _temp_GU.Brand_LS3;
                                PubChangeNow(infoGlue);
                            }
                            else if (position == "SF1")
                            {
                                PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                                infoGlue.OffSetValue = 0;
                                infoGlue.Part = IPSHandlePart.GlueSF1;
                                infoGlue.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoGlue.Width = stateInfo_ms1.CurWidth;
                                infoGlue.Flute = stateInfo_ms1.CurFlute;
                                infoGlue.LastCode = stateInfo_ms1.LastCode + "/" + stateInfo_ls1.LastCode;
                                infoGlue.LastWidth = stateInfo_ms1.LastWidth;
                                infoGlue.LastFlute = stateInfo_ms1.LastFlute;
                                infoGlue.BrandMS1 = stateInfo_ms1.BrandMS1;
                                infoGlue.BrandLS1 = stateInfo_ls1.BrandLS1;
                                PubChangeNow(infoGlue);
                            }
                            else if (position == "SF2")
                            {
                                PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                                infoGlue.OffSetValue = 0;
                                infoGlue.Part = IPSHandlePart.GlueSF2;
                                infoGlue.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoGlue.Width = stateInfo_ms2.CurWidth;
                                infoGlue.Flute = stateInfo_ms2.CurFlute;
                                infoGlue.LastCode = stateInfo_ms2.LastCode + "/" + stateInfo_ls2.LastCode;
                                infoGlue.LastWidth = stateInfo_ms2.LastWidth;
                                infoGlue.LastFlute = stateInfo_ms2.LastFlute;
                                infoGlue.BrandMS2 = stateInfo_ms2.BrandMS2;
                                infoGlue.BrandLS2 = stateInfo_ls2.BrandLS2;
                                PubChangeNow(infoGlue);
                            }
                            else if (position == "SF3")
                            {
                                PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                                infoGlue.OffSetValue = 0;
                                infoGlue.Part = IPSHandlePart.GlueSF3;
                                infoGlue.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoGlue.Width = stateInfo_ms3.CurWidth;
                                infoGlue.Flute = stateInfo_ms3.CurFlute;
                                infoGlue.LastCode = stateInfo_ms3.LastCode + "/" + stateInfo_ls3.LastCode;
                                infoGlue.LastWidth = stateInfo_ms3.LastWidth;
                                infoGlue.LastFlute = stateInfo_ms3.LastFlute;
                                infoGlue.BrandMS3 = stateInfo_ms3.BrandMS3;
                                infoGlue.BrandLS3 = stateInfo_ls3.BrandLS3;
                                PubChangeNow(infoGlue);
                            }
                        }
                        else if (type == "WrapInfo")
                        {
                            if (position == "GU_1st" || position == "GU_2nd" || position == "GU_3rd")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapGu;
                                infoWrap.Width = stateInfo_gu.CurWidth;
                                infoWrap.Flute = stateInfo_gu.CurFlute;
                                //infoWrap.Code = stateInfo_gu.CurCode;
                                infoWrap.Code = guCode;
                                infoWrap.LastCode = stateInfo_gu.LastCode;
                                infoWrap.LastWidth = stateInfo_gu.LastWidth;
                                infoWrap.LastFlute = stateInfo_gu.LastFlute;
                                infoWrap.BrandLS0 = stateInfo_ls0.BrandLS0;
                                infoWrap.BrandMS1 = _temp_GU.Brand_MS1;
                                infoWrap.BrandLS1 = _temp_GU.Brand_LS1;
                                infoWrap.BrandMS2 = _temp_GU.Brand_MS2;
                                infoWrap.BrandLS2 = _temp_GU.Brand_LS2;
                                infoWrap.BrandMS3 = _temp_GU.Brand_MS3;
                                infoWrap.BrandLS3 = _temp_GU.Brand_LS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "GU_0")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS0;
                                infoWrap.Code = stateInfo_ls0.CurCode + "/" + stateInfo_gu.CurCode;
                                infoWrap.Width = stateInfo_ls0.CurWidth;
                                infoWrap.Flute = stateInfo_ls0.CurFlute;
                                infoWrap.LastCode = stateInfo_ls0.LastCode;
                                infoWrap.LastWidth = stateInfo_ls0.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls0.LastFlute;
                                infoWrap.BrandLS0 = stateInfo_ls0.BrandLS0;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "GU_Ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapGu_Add2;
                                infoWrap.Width = stateInfo_gu.CurWidth;
                                infoWrap.Flute = stateInfo_gu.CurFlute;
                                infoWrap.Code = stateInfo_gu.CurCode;
                                infoWrap.LastCode = stateInfo_gu.LastCode;
                                infoWrap.LastWidth = stateInfo_gu.LastWidth;
                                infoWrap.LastFlute = stateInfo_gu.LastFlute;
                                infoWrap.BrandLS0 = stateInfo_ls0.BrandLS0;
                                infoWrap.BrandMS1 = _temp_GU.Brand_MS1;
                                infoWrap.BrandLS1 = _temp_GU.Brand_LS1;
                                infoWrap.BrandMS2 = _temp_GU.Brand_MS2;
                                infoWrap.BrandLS2 = _temp_GU.Brand_LS2;
                                infoWrap.BrandMS3 = _temp_GU.Brand_MS3;
                                infoWrap.BrandLS3 = _temp_GU.Brand_LS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                infoWrap.BrandMS1 = stateInfo_ms1.BrandMS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS1ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS1_Ext;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                infoWrap.BrandMS1 = stateInfo_ms1.BrandMS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ls1.CurWidth;
                                infoWrap.Flute = stateInfo_ls1.CurFlute;
                                infoWrap.LastCode = stateInfo_ls1.LastCode;
                                infoWrap.LastWidth = stateInfo_ls1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls1.LastFlute;
                                infoWrap.BrandLS1 = stateInfo_ls1.BrandLS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS1ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS1_Ext;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ls1.CurWidth;
                                infoWrap.Flute = stateInfo_ls1.CurFlute;
                                infoWrap.LastCode = stateInfo_ls1.LastCode;
                                infoWrap.LastWidth = stateInfo_ls1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls1.LastFlute;
                                infoWrap.BrandLS1 = stateInfo_ls1.BrandLS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                infoWrap.BrandMS2 = stateInfo_ms2.BrandMS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS2ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS2_Ext;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                infoWrap.BrandMS2 = stateInfo_ms2.BrandMS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ls2.CurWidth;
                                infoWrap.Flute = stateInfo_ls2.CurFlute;
                                infoWrap.LastCode = stateInfo_ls2.LastCode;
                                infoWrap.LastWidth = stateInfo_ls2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls2.LastFlute;
                                infoWrap.BrandLS2 = stateInfo_ls2.BrandLS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS2ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS2_Ext;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ls2.CurWidth;
                                infoWrap.Flute = stateInfo_ls2.CurFlute;
                                infoWrap.LastCode = stateInfo_ls2.LastCode;
                                infoWrap.LastWidth = stateInfo_ls2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls2.LastFlute;
                                infoWrap.BrandLS2 = stateInfo_ls2.BrandLS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                infoWrap.BrandMS3 = stateInfo_ms3.BrandMS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS3ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS3_Ext;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                infoWrap.BrandMS3 = stateInfo_ms3.BrandMS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ls3.CurWidth;
                                infoWrap.Flute = stateInfo_ls3.CurFlute;
                                infoWrap.LastCode = stateInfo_ls3.LastCode;
                                infoWrap.LastWidth = stateInfo_ls3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls3.LastFlute;
                                infoWrap.BrandLS3 = stateInfo_ls3.BrandLS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS3ext")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS3_Ext;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ls3.CurWidth;
                                infoWrap.Flute = stateInfo_ls3.CurFlute;
                                infoWrap.LastCode = stateInfo_ls3.LastCode;
                                infoWrap.LastWidth = stateInfo_ls3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls3.LastFlute;
                                infoWrap.BrandLS3 = stateInfo_ls3.BrandLS3;
                                PubChangeNow(infoWrap);
                            }
                            //温度包角模式
                            else if (position.Contains("GU"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapGu;
                                infoWrap.Width = stateInfo_gu.CurWidth;
                                infoWrap.Flute = stateInfo_gu.CurFlute;
                                infoWrap.Code = stateInfo_gu.CurCode;
                                infoWrap.LastCode = stateInfo_gu.LastCode;
                                infoWrap.LastWidth = stateInfo_gu.LastWidth;
                                infoWrap.LastFlute = stateInfo_gu.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("LS0"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS0;
                                infoWrap.Code = stateInfo_ls0.CurCode + "/" + stateInfo_gu.CurCode;
                                infoWrap.Width = stateInfo_ls0.CurWidth;
                                infoWrap.Flute = stateInfo_ls0.CurFlute;
                                infoWrap.LastCode = stateInfo_ls0.LastCode;
                                infoWrap.LastWidth = stateInfo_ls0.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls0.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("SF1"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);

                                PubChangeNowInfo infoWrap1 = new PubChangeNowInfo();
                                infoWrap1.OffSetValue = 0;
                                infoWrap1.Part = IPSHandlePart.WrapLS1;
                                infoWrap1.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap1.Width = stateInfo_ls1.CurWidth;
                                infoWrap1.Flute = stateInfo_ls1.CurFlute;
                                infoWrap1.LastCode = stateInfo_ls1.LastCode;
                                infoWrap1.LastWidth = stateInfo_ls1.LastWidth;
                                infoWrap1.LastFlute = stateInfo_ls1.LastFlute;
                                PubChangeNow(infoWrap1);
                            }
                            else if (position.Contains("MS1"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("LS1"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ls1.CurWidth;
                                infoWrap.Flute = stateInfo_ls1.CurFlute;
                                infoWrap.LastCode = stateInfo_ls1.LastCode;
                                infoWrap.LastWidth = stateInfo_ls1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("SF2"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);

                                PubChangeNowInfo infoWrap1 = new PubChangeNowInfo();
                                infoWrap1.OffSetValue = 0;
                                infoWrap1.Part = IPSHandlePart.WrapLS2;
                                infoWrap1.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap1.Width = stateInfo_ls2.CurWidth;
                                infoWrap1.Flute = stateInfo_ls2.CurFlute;
                                infoWrap1.LastCode = stateInfo_ls2.LastCode;
                                infoWrap1.LastWidth = stateInfo_ls2.LastWidth;
                                infoWrap1.LastFlute = stateInfo_ls2.LastFlute;
                                PubChangeNow(infoWrap1);
                            }
                            else if (position.Contains("MS2"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("LS2"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ls2.CurWidth;
                                infoWrap.Flute = stateInfo_ls2.CurFlute;
                                infoWrap.LastCode = stateInfo_ls2.LastCode;
                                infoWrap.LastWidth = stateInfo_ls2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("SF3"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);

                                PubChangeNowInfo infoWrap1 = new PubChangeNowInfo();
                                infoWrap1.OffSetValue = 0;
                                infoWrap1.Part = IPSHandlePart.WrapLS3;
                                infoWrap1.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap1.Width = stateInfo_ls3.CurWidth;
                                infoWrap1.Flute = stateInfo_ls3.CurFlute;
                                infoWrap1.LastCode = stateInfo_ls3.LastCode;
                                infoWrap1.LastWidth = stateInfo_ls3.LastWidth;
                                infoWrap1.LastFlute = stateInfo_ls3.LastFlute;
                                PubChangeNow(infoWrap1);
                            }
                            else if (position.Contains("MS3"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapMS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position.Contains("LS3"))
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.WrapLS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ls3.CurWidth;
                                infoWrap.Flute = stateInfo_ls3.CurFlute;
                                infoWrap.LastCode = stateInfo_ls3.LastCode;
                                infoWrap.LastWidth = stateInfo_ls3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "BridgeTensionInfo")
                        {
                            PubChangeNowInfo infoBt = new PubChangeNowInfo();
                            infoBt.OffSetValue = 0;
                            infoBt.Part = IPSHandlePart.BridgeTension;
                            //infoBt.Code = stateInfo_gu.CurCode;
                            infoBt.Code = guCode;
                            infoBt.Flute = stateInfo_gu.CurFlute;
                            infoBt.Width = stateInfo_gu.CurWidth;
                            infoBt.LastCode = stateInfo_gu.LastCode;
                            infoBt.LastFlute = stateInfo_gu.LastFlute;
                            infoBt.LastWidth = stateInfo_gu.LastWidth;
                            infoBt.BrandLS0 = stateInfo_ls0.BrandLS0;
                            infoBt.BrandMS1 = _temp_GU.Brand_MS1;
                            infoBt.BrandLS1 = _temp_GU.Brand_LS1;
                            infoBt.BrandMS2 = _temp_GU.Brand_MS2;
                            infoBt.BrandLS2 = _temp_GU.Brand_LS2;
                            infoBt.BrandMS3 = _temp_GU.Brand_MS3;
                            infoBt.BrandLS3 = _temp_GU.Brand_LS3;
                            PubChangeNow(infoBt);
                        }
                        else if (type == "ColdPlatePressInfo")
                        {
                            PubChangeNowInfo infoCpp = new PubChangeNowInfo();
                            infoCpp.OffSetValue = 0;
                            infoCpp.Part = IPSHandlePart.CodePress;
                            infoCpp.Code = stateInfo_gu.CurCode;
                            infoCpp.Flute = stateInfo_gu.CurFlute;
                            infoCpp.Width = stateInfo_gu.CurWidth;
                            infoCpp.LastCode = stateInfo_gu.LastCode;
                            infoCpp.LastFlute = stateInfo_gu.LastFlute;
                            infoCpp.LastWidth = stateInfo_gu.LastWidth;
                            PubChangeNow(infoCpp);
                        }
                        else if (type == "HotPlatePressInfo" || type == "HotPlatePress1Info" || type == "HotPlatePress2Info")
                        {
                            PubChangeNowInfo infoHpp = new PubChangeNowInfo();
                            infoHpp.OffSetValue = 0;
                            infoHpp.Part = IPSHandlePart.HotPress;
                            infoHpp.Code = stateInfo_gu.CurCode;
                            infoHpp.Flute = stateInfo_gu.CurFlute;
                            infoHpp.Width = stateInfo_gu.CurWidth;
                            infoHpp.LastCode = stateInfo_gu.LastCode;
                            infoHpp.LastFlute = stateInfo_gu.LastFlute;
                            infoHpp.LastWidth = stateInfo_gu.LastWidth;
                            PubChangeNow(infoHpp);
                        }
                        else if (type == "HotLoadGroupQtyInfo")
                        {
                            PubChangeNowInfo infoHgp = new PubChangeNowInfo();
                            infoHgp.OffSetValue = 0;
                            infoHgp.Part = IPSHandlePart.PressGroupQty;
                            infoHgp.Code = stateInfo_gu.CurCode;
                            infoHgp.Flute = stateInfo_gu.CurFlute;
                            infoHgp.Width = stateInfo_gu.CurWidth;
                            infoHgp.LastCode = stateInfo_gu.LastCode;
                            infoHgp.LastFlute = stateInfo_gu.LastFlute;
                            infoHgp.LastWidth = stateInfo_gu.LastWidth;
                            PubChangeNow(infoHgp);
                        }
                        else if (type == "PressRollInfo")
                        {
                            if (position == "SF1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.PressRollSF1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode + "/" + stateInfo_ls1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.PressRollSF2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode + "/" + stateInfo_ls2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.PressRollSF3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode + "/" + stateInfo_ls3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "CorrugatedRollInfo")
                        {
                            if (position == "SF1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.CorrugatedRollSF1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.CorrugatedRollSF2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.CorrugatedRollSF3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "SplicerTensionInfo")
                        {
                            if (position == "LS0")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionLS0;
                                infoWrap.Code = stateInfo_ls0.CurCode + "/" + stateInfo_gu.CurCode;
                                infoWrap.Width = stateInfo_ls0.CurWidth;
                                infoWrap.Flute = stateInfo_ls0.CurFlute;
                                infoWrap.LastCode = stateInfo_ls0.LastCode;
                                infoWrap.LastWidth = stateInfo_ls0.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls0.LastFlute;
                                infoWrap.BrandLS0 = stateInfo_ls0.BrandLS0;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionMS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                infoWrap.BrandMS1 = stateInfo_ms1.BrandMS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionLS1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ls1.CurWidth;
                                infoWrap.Flute = stateInfo_ls1.CurFlute;
                                infoWrap.LastCode = stateInfo_ls1.LastCode;
                                infoWrap.LastWidth = stateInfo_ls1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls1.LastFlute;
                                infoWrap.BrandLS1 = stateInfo_ls1.BrandLS1;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionMS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                infoWrap.BrandMS2 = stateInfo_ms2.BrandMS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionLS2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ls2.CurWidth;
                                infoWrap.Flute = stateInfo_ls2.CurFlute;
                                infoWrap.LastCode = stateInfo_ls2.LastCode;
                                infoWrap.LastWidth = stateInfo_ls2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls2.LastFlute;
                                infoWrap.BrandLS2 = stateInfo_ls2.BrandLS2;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "MS3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionMS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                infoWrap.BrandMS3 = stateInfo_ms3.BrandMS3;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "LS3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.TensionLS3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ls3.CurWidth;
                                infoWrap.Flute = stateInfo_ls3.CurFlute;
                                infoWrap.LastCode = stateInfo_ls3.LastCode;
                                infoWrap.LastWidth = stateInfo_ls3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ls3.LastFlute;
                                infoWrap.BrandLS3 = stateInfo_ls3.BrandLS3;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "SprayInfo")
                        {
                            if (position == "SF1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.HotSpraySF1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.HotSpraySF2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.HotSpraySF3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "VacuumBlowerInfo")
                        {
                            if (position == "SF1")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.VacuumBlowerSF1;
                                infoWrap.Code = stateInfo_ms1.CurCode + "/" + stateInfo_ls1.CurCode;
                                infoWrap.Width = stateInfo_ms1.CurWidth;
                                infoWrap.Flute = stateInfo_ms1.CurFlute;
                                infoWrap.LastCode = stateInfo_ms1.LastCode;
                                infoWrap.LastWidth = stateInfo_ms1.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms1.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF2")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.VacuumBlowerSF2;
                                infoWrap.Code = stateInfo_ms2.CurCode + "/" + stateInfo_ls2.CurCode;
                                infoWrap.Width = stateInfo_ms2.CurWidth;
                                infoWrap.Flute = stateInfo_ms2.CurFlute;
                                infoWrap.LastCode = stateInfo_ms2.LastCode;
                                infoWrap.LastWidth = stateInfo_ms2.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms2.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                            else if (position == "SF3")
                            {
                                PubChangeNowInfo infoWrap = new PubChangeNowInfo();
                                infoWrap.OffSetValue = 0;
                                infoWrap.Part = IPSHandlePart.VacuumBlowerSF3;
                                infoWrap.Code = stateInfo_ms3.CurCode + "/" + stateInfo_ls3.CurCode;
                                infoWrap.Width = stateInfo_ms3.CurWidth;
                                infoWrap.Flute = stateInfo_ms3.CurFlute;
                                infoWrap.LastCode = stateInfo_ms3.LastCode;
                                infoWrap.LastWidth = stateInfo_ms3.LastWidth;
                                infoWrap.LastFlute = stateInfo_ms3.LastFlute;
                                PubChangeNow(infoWrap);
                            }
                        }
                        else if (type == "RidingRollInfo")
                        {
                            //糊机骑辊系数修改
                            PubChangeNowInfo infoGlue = new PubChangeNowInfo();
                            infoGlue.OffSetValue = 0;
                            infoGlue.Part = IPSHandlePart.RidingRoll;
                            infoGlue.Width = stateInfo_gu.CurWidth;
                            infoGlue.Flute = stateInfo_gu.CurFlute;
                            //infoGlue.Code = stateInfo_gu.CurCode;
                            infoGlue.Code = guCode;
                            infoGlue.LastCode = stateInfo_gu.LastCode;
                            infoGlue.LastWidth = stateInfo_gu.LastWidth;
                            infoGlue.LastFlute = stateInfo_gu.LastFlute;
                            infoGlue.BrandLS0 = stateInfo_ls0.BrandLS0;
                            infoGlue.BrandMS1 = _temp_GU.Brand_MS1;
                            infoGlue.BrandLS1 = _temp_GU.Brand_LS1;
                            infoGlue.BrandMS2 = _temp_GU.Brand_MS2;
                            infoGlue.BrandLS2 = _temp_GU.Brand_LS2;
                            infoGlue.BrandMS3 = _temp_GU.Brand_MS3;
                            infoGlue.BrandLS3 = _temp_GU.Brand_LS3;
                            PubChangeNow(infoGlue);
                        }
                        else if (type == "SecondSteam")
                        {
                            if (comm.PointVars.Exists(a => a.VarCode == PointVarEnum.PLC_2ndSteamPressure.ToString()))
                            {
                                //蒸汽二次备压
                                comm.WriteVar(PointVarEnum.PLC_2ndSteamPressure.ToString(), info.VarValue);
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error($"处理M107异常：{ex.Message}---报文信息：{msg}", module);
            }
        }

        // <summary>
        /// 根据巡航调整接纸机张力
        /// </summary>
        /// <param name="position">接纸机机台编号</param>
        /// <param name="rate">系数</param>
        public void SetSPTensionByCruise(string position, decimal rate)
        {
            switch (position)
            {
                case "LS0":
                    GlobalControl.tensionPercent_LS0 = rate;
                    break;
                case "LS1":
                    GlobalControl.tensionPercent_LS1 = rate;
                    break;
                case "LS2":
                    GlobalControl.tensionPercent_LS2 = rate;
                    break;
                case "LS3":
                    GlobalControl.tensionPercent_LS3 = rate;
                    break;
                case "MS1":
                    GlobalControl.tensionPercent_MS1 = rate;
                    break;
                case "MS2":
                    GlobalControl.tensionPercent_MS2 = rate;
                    break;
                case "MS3":
                    GlobalControl.tensionPercent_MS3 = rate;
                    break;
                default:
                    break;
            }

            var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
            if (position == "LS0")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS0);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS0);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_LS0_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.LS0_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "MS1")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS1);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS1);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_MS1_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.MS1_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "LS1")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS1);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS1);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_LS1_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.LS1_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "MS2")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS2);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS2);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_MS2_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.MS2_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "LS2")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS2);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS2);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_LS2_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.LS2_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "MS3")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS3);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionMS3);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_MS3_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.MS3_NominalTension_daN.ToString(), setValue);
                }
            }
            else if (position == "LS3")
            {
                //从全局变量里面获取当前的张力设定值，然后在乘以系数，给设备写值
                var info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS3);
                if (info == null)
                {
                    Thread.Sleep(1000);
                    info = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.TensionLS3);
                    if (info == null)
                        return;
                }
                decimal setValue = Math.Round(info.OriginalValue * rate, 0);
                info.SetValue = setValue;
                if (formSetInfo.F_SplicerTension_LS3_Form_IsOpen)
                {
                    comm.WriteVar(PointVarEnum.LS3_NominalTension_daN.ToString(), setValue);
                }
            }
        }

        /// <summary>
        /// 获取接纸机下批理论信息
        /// </summary>
        /// <param name="info">存储信息变量</param>
        /// <param name="orders">当前全部订单</param>
        /// <param name="position">部位：LS0，MS1，LS1，DF</param>
        private void CalSPNextBatchInfo(ref DriveStateInfo info, List<OrderInfo> orders, string position)
        {
            var firstInfo = orders.FirstOrDefault();
            if (firstInfo == null) return;
            string code = firstInfo.WO_PaperCode;
            int width = firstInfo.WO_Width;
            string[] codes = code.Split('.');
            OrderInfo nextInfo = null;
            switch (position)
            {
                case "LS0":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[0] != "-" && (it.WO_PaperCode.Split('.')[0] != codes[0] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[0];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        info.NextBatchTheoryFlute = nextInfo.WO_Wave;
                    }
                    break;
                case "MS1":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[1] != "-" && (it.WO_PaperCode.Split('.')[1] != codes[1] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[1];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "MS1", ref info);
                    }
                    break;
                case "LS1":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[2] != "-" && (it.WO_PaperCode.Split('.')[2] != codes[2] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[2];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "LS1", ref info);
                    }
                    break;
                case "MS2":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[3] != "-" && (it.WO_PaperCode.Split('.')[3] != codes[3] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[3];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "MS2", ref info);
                    }
                    break;
                case "LS2":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[4] != "-" && (it.WO_PaperCode.Split('.')[4] != codes[4] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[4];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "LS2", ref info);
                    }
                    break;
                case "MS3":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[5] != "-" && (it.WO_PaperCode.Split('.')[5] != codes[5] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[5];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "MS3", ref info);
                    }
                    break;
                case "LS3":
                    nextInfo = orders.FirstOrDefault(it => it.WO_PaperCode.Split('.')[6] != "-" && (it.WO_PaperCode.Split('.')[6] != codes[6] || it.WO_Width != width));
                    if (nextInfo != null)
                    {
                        info.NextBatchTheoryCodeAll = nextInfo.WO_PaperCode;
                        info.NextBatchTheoryCode = nextInfo.WO_PaperCode.Split('.')[6];
                        info.NextBatchTheoryWidth = nextInfo.WO_Width;
                        CalSPFluteByCodeAndWave(nextInfo.WO_PaperCode, nextInfo.WO_Wave, "LS3", ref info);
                    }
                    break;
                default:
                    break;
            }
            logger.Info($"CalSPNextBatchInfo--{position}-下批材质={info.NextBatchTheoryCode},下批门幅={info.NextBatchTheoryWidth},下批楞型={info.NextBatchTheoryFlute},下批全材质={info.NextBatchTheoryCodeAll}", module);
        }

        private void CalSPFluteByCodeAndWave(string paperCode, string flute, string machineID, ref DriveStateInfo info)
        {
            try
            {
                List<string> papers = new List<string>();
                if (paperCode.Contains("."))
                {
                    papers = paperCode.Split('.').ToList();
                }
                else
                {
                    papers = paperCode.ToCharArray().Select(c => c.ToString()).ToList();
                }
                int index = 1;
                for (int i = 0; i < papers.Count; i++)
                {
                    switch (i)
                    {
                        case 1:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS1" || machineID == "LS1")
                                {
                                    info.NextBatchTheoryFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        case 3:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS2" || machineID == "LS2")
                                {
                                    info.NextBatchTheoryFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        case 5:
                            if (papers[i] != "-")
                            {
                                if (machineID == "MS3" || machineID == "LS3")
                                {
                                    info.NextBatchTheoryFlute = flute.Substring(index, 1);
                                }
                                index++;
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
            catch (Exception ex)
            {

                throw ex;
            }

        }


        /// <summary>
        /// 横切换单判断是否换材，校准接纸机材质
        /// </summary>
        public void HQChangeOrder()
        {
            try
            {
                //从数据字典中，拿控制开关信息
                var dictdataInfo = BLLFactory<DictDataInfoManager>.Instance.GetFirst(it => it.PD_Property == "UseHQChangeOrder");
                if (dictdataInfo == null || dictdataInfo.PD_Value == "0")//0:不启用  1：启用
                {
                    return;
                }

                StringBuilder sb = new StringBuilder();
                sb.AppendLine("横切换单或者强换，触发判定是否横切换材");
                List<string> flags = new List<string>();
                //取当前首笔订单
                var curInfo = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();
                if (curInfo == null)
                {
                    return;
                }
                //取刚刚完工的订单
                var lastInfo = BLLFactory<OrderFinishInfoManage>.Instance.AsQueryable().OrderByDescending(it => it.WOF_ID).First();
                if (lastInfo == null)
                {
                    return;
                }
                string curPaper = curInfo.WO_PaperCode;
                int curWidth = curInfo.WO_Width;
                decimal curWidthInch = curInfo.WO_WidthInch ?? 0;
                string lastPaper = lastInfo.WOF_PaperCode;
                int lastWidth = lastInfo.WOF_Width;
                decimal lastWidthInch = lastInfo.WOF_WidthInch ?? 0;
                List<string> curPapers = new List<string>();
                List<string> lastPapers = new List<string>();
                if (curPaper.Contains("."))
                {
                    curPapers = curPaper.Split('.').ToList();
                }
                else
                {
                    curPapers = curPaper.ToCharArray().Select(c => c.ToString()).ToList();
                }
                if (lastPaper.Contains("."))
                {
                    lastPapers = lastPaper.Split('.').ToList();
                }
                else
                {
                    lastPapers = lastPaper.ToCharArray().Select(c => c.ToString()).ToList();
                }

                //循环判断各机台在横切位置是否换材
                for (int i = 0; i < curPapers.Count; i++)
                {
                    //判定为换材
                    if ((curPapers[i] != lastPapers[i] || curWidth != lastWidth) && curPapers[i] != "-")
                    {
                        //数据库是否有正在使用的实际材质，有实际材质则优先使用实际材质，那么这边就不需要再校准
                        bool hasDbPaper = false;
                        CorrCodeInfo corrCodeInfo = new CorrCodeInfo();
                        switch (i)
                        {
                            case 0:
                                sb.AppendLine($"LS0横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("LS0", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "LS0").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"LS0有实际材质，不需要校准");
                                    continue;
                                }

                                //当前使用的材质和本批理论材质不一致时，进行材质校准
                                if (stateInfo_ls0.CurCode != curPapers[i] || stateInfo_ls0.CurWidth != curWidth)
                                {
                                    stateInfo_ls0.LastCode = stateInfo_ls0.CurCode;
                                    stateInfo_ls0.LastWidth = stateInfo_ls0.CurWidth;
                                    stateInfo_ls0.LastFlute = stateInfo_ls0.CurFlute;
                                    stateInfo_ls0.CurCode = curPapers[i];
                                    stateInfo_ls0.CurWidth = curWidth;
                                    stateInfo_ls0.CurFlute = curInfo.WO_Wave;
                                    stateInfo_ls0.NextBachCode = curPaper;
                                    stateInfo_ls0.CodeALl = curPaper;
                                    if (stateInfo_gu.CurCode.Contains("."))
                                    {
                                        var codes = stateInfo_gu.CurCode.Split('.');
                                        if (codes[0] != "-")
                                        {
                                            codes[0] = stateInfo_ls0.CurCode;
                                        }
                                        stateInfo_gu.CurCode = string.Join(".", codes);
                                    }
                                    else
                                    {
                                        List<string> codes = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
                                        if (codes[0] != "-")
                                        {
                                            codes[0] = stateInfo_ls0.CurCode;
                                            stateInfo_gu.CurCode = string.Join("", codes);
                                        }
                                    }
                                    //立即发送面纸换材消息给各业务执行类
                                    flags.Add("LS0");
                                    sb.AppendLine($"LS0材质校准：当前正在用的材质={stateInfo_ls0.LastCode}，门幅={stateInfo_ls0.LastWidth}，校准后的材质={stateInfo_ls0.CurCode}，门幅={stateInfo_ls0.CurWidth}");
                                }
                                break;
                            case 1:
                                sb.AppendLine($"MS1横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("MS1", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "MS1").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"MS1有实际材质，不需要校准");
                                    continue;
                                }
                                if (stateInfo_ms1.CurCode != curPapers[i] || stateInfo_ms1.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ms1, curPaper, curInfo.WO_Wave, "MS1");
                                    stateInfo_ms1.LastCode = stateInfo_ms1.CurCode;
                                    stateInfo_ms1.LastWidth = stateInfo_ms1.CurWidth;
                                    stateInfo_ms1.CurCode = curPapers[i];
                                    stateInfo_ms1.CurWidth = curWidth;
                                    stateInfo_ms1.NextBachCode = curPaper;
                                    stateInfo_ms1.CodeALl = curPaper;

                                    stateInfo_ls1.LastFlute = stateInfo_ls1.CurFlute;
                                    stateInfo_ls1.CurFlute = stateInfo_ms1.CurFlute;

                                    //立即发送MS1换材消息给各业务执行类
                                    flags.Add("MS1");
                                    sb.AppendLine($"MS1材质校准:当前正在用的材质={stateInfo_ms1.LastCode}，门幅={stateInfo_ms1.LastWidth}，校准后的材质={stateInfo_ms1.CurCode}，门幅={stateInfo_ms1.CurWidth}");
                                }
                                break;
                            case 2:
                                sb.AppendLine($"LS1横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("LS1", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "LS1").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"LS1有实际材质，不需要校准");
                                    continue;
                                }
                                if (stateInfo_ls1.CurCode != curPapers[i] || stateInfo_ls1.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ls1, curPaper, curInfo.WO_Wave, "LS1");
                                    stateInfo_ls1.LastCode = stateInfo_ls1.CurCode;
                                    stateInfo_ls1.LastWidth = stateInfo_ls1.CurWidth;
                                    stateInfo_ls1.CurCode = curPapers[i];
                                    stateInfo_ls1.CurWidth = curWidth;
                                    stateInfo_ls1.NextBachCode = curPaper;
                                    stateInfo_ls1.CodeALl = curPaper;

                                    stateInfo_ms1.LastFlute = stateInfo_ms1.CurFlute;
                                    stateInfo_ms1.CurFlute = stateInfo_ls1.CurFlute;

                                    //立即发送LS1换材消息给各业务执行类
                                    flags.Add("LS1");
                                    sb.AppendLine($"LS1材质校准:当前正在用的材质={stateInfo_ls1.LastCode}，门幅={stateInfo_ls1.LastWidth}，校准后的材质={stateInfo_ls1.CurCode}，门幅={stateInfo_ls1.CurWidth}");
                                }

                                break;
                            case 3:
                                sb.AppendLine($"MS2横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("MS2", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "MS2").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"MS2有实际材质，不需要校准");
                                    continue;
                                }

                                if (stateInfo_ms2.CurCode != curPapers[i] || stateInfo_ms2.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ms2, curPaper, curInfo.WO_Wave, "MS2");
                                    stateInfo_ms2.LastCode = stateInfo_ms2.CurCode;
                                    stateInfo_ms2.LastWidth = stateInfo_ms2.CurWidth;
                                    stateInfo_ms2.CurCode = curPapers[i];
                                    stateInfo_ms2.CurWidth = curWidth;
                                    stateInfo_ms2.NextBachCode = curPaper;
                                    stateInfo_ms2.CodeALl = curPaper;

                                    stateInfo_ls2.LastFlute = stateInfo_ls2.CurFlute;
                                    stateInfo_ls2.CurFlute = stateInfo_ms2.CurFlute;

                                    //立即发送MS2换材消息给各业务执行类
                                    flags.Add("MS2");
                                    sb.AppendLine($"MS2材质校准:当前正在用的材质={stateInfo_ms2.LastCode}，门幅={stateInfo_ms2.LastWidth}，校准后的材质={stateInfo_ms2.CurCode}，门幅={stateInfo_ms2.CurWidth}");
                                }
                                break;
                            case 4:
                                sb.AppendLine($"LS2横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("LS2", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "LS2").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"LS2有实际材质，不需要校准");
                                    continue;
                                }
                                if (stateInfo_ls2.CurCode != curPapers[i] || stateInfo_ls2.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ls2, curPaper, curInfo.WO_Wave, "LS2");
                                    stateInfo_ls2.LastCode = stateInfo_ls2.CurCode;
                                    stateInfo_ls2.LastWidth = stateInfo_ls2.CurWidth;
                                    stateInfo_ls2.CurCode = curPapers[i];
                                    stateInfo_ls2.CurWidth = curWidth;
                                    stateInfo_ls2.NextBachCode = curPaper;
                                    stateInfo_ls2.CodeALl = curPaper;

                                    stateInfo_ms2.LastFlute = stateInfo_ms2.CurFlute;
                                    stateInfo_ms2.CurFlute = stateInfo_ls2.CurFlute;

                                    //立即发送LS2换材消息给各业务执行类
                                    flags.Add("LS2");
                                    sb.AppendLine($"LS2材质校准:当前正在用的材质={stateInfo_ls2.LastCode}，门幅={stateInfo_ls2.LastWidth}，校准后的材质={stateInfo_ls2.CurCode}，门幅={stateInfo_ls2.CurWidth}");
                                }
                                break;
                            case 5:
                                sb.AppendLine($"MS3横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("MS3", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "MS3").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"MS3有实际材质，不需要校准");
                                    continue;
                                }
                                if (stateInfo_ms3.CurCode != curPapers[i] || stateInfo_ms3.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ms3, curPaper, curInfo.WO_Wave, "MS3");
                                    stateInfo_ms3.LastCode = stateInfo_ms3.CurCode;
                                    stateInfo_ms3.LastWidth = stateInfo_ms3.CurWidth;
                                    stateInfo_ms3.CurCode = curPapers[i];
                                    stateInfo_ms3.CurWidth = curWidth;
                                    stateInfo_ms3.NextBachCode = curPaper;
                                    stateInfo_ms3.CodeALl = curPaper;

                                    stateInfo_ls3.LastFlute = stateInfo_ls3.CurFlute;
                                    stateInfo_ls3.CurFlute = stateInfo_ms3.CurFlute;

                                    //立即发送MS3换材消息给各业务执行类
                                    flags.Add("MS3");
                                    sb.AppendLine($"MS3材质校准:当前正在用的材质={stateInfo_ms3.LastCode}，门幅={stateInfo_ms3.LastWidth}，校准后的材质={stateInfo_ms3.CurCode}，门幅={stateInfo_ms3.CurWidth}");
                                }
                                break;
                            case 6:
                                sb.AppendLine($"LS3横切换材了。上笔材质={lastPapers[i]}，门幅={lastWidth}；本批材质={curPapers[i]}，门幅={curWidth}");
                                corrCodeInfo.CurCode = curPapers[i];
                                corrCodeInfo.CurWidth = curWidth;
                                corrCodeInfo.CurWidthInch = curWidthInch;
                                GlobalControl.corrDict.AddOrUpdate("LS3", corrCodeInfo, (k, v) => corrCodeInfo);
                                hasDbPaper = BLLFactory<ERPPaperForMachineManager>.Instance.AsQueryable().Where(it => it.State == 1 && it.MachineID == "LS3").Any();
                                if (hasDbPaper)
                                {
                                    sb.AppendLine($"LS3有实际材质，不需要校准");
                                    continue;
                                }
                                if (stateInfo_ls3.CurCode != curPapers[i] || stateInfo_ls3.CurWidth != curWidth)
                                {
                                    GetSPFlute(ref stateInfo_ls3, curPaper, curInfo.WO_Wave, "LS3");
                                    stateInfo_ls3.LastCode = stateInfo_ls3.CurCode;
                                    stateInfo_ls3.LastWidth = stateInfo_ls3.CurWidth;
                                    stateInfo_ls3.CurCode = curPapers[i];
                                    stateInfo_ls3.CurWidth = curWidth;
                                    stateInfo_ls3.NextBachCode = curPaper;
                                    stateInfo_ls3.CodeALl = curPaper;

                                    stateInfo_ms3.LastFlute = stateInfo_ms3.CurFlute;
                                    stateInfo_ms3.CurFlute = stateInfo_ls3.CurFlute;

                                    //立即发送LS3换材消息给各业务执行类
                                    flags.Add("LS3");
                                    sb.AppendLine($"LS3材质校准:当前正在用的材质={stateInfo_ls3.LastCode}，门幅={stateInfo_ls3.LastWidth}，校准后的材质={stateInfo_ls3.CurCode}，门幅={stateInfo_ls3.CurWidth}");
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
                if (flags.Count > 0)
                {
                    sb.AppendLine("立刻给各部位赋值");
                    HandleFirstAll(flags);
                }
                logger.Info(sb.ToString(), module);
                sb = null;
            }
            catch (Exception ex)
            {
                logger.Error($"HQChangeOrder执行异常：{ex}", module);
            }
        }

        /// <summary>
        /// 得到接纸机实际材质进行糊机实材处理函数
        /// </summary>
        /// <param name="spName">接纸机名称</param>
        private async Task GetSPRealPaperToChangeGUPaper(string spName)
        {
            //已处理糊间隙标识
            bool hasHandleGlue = false;
            //已处理包角标识
            bool hasHandleWrap = false;

            var waveSetInfos = BLLFactory<BaseWavesManager>.Instance.GetList();
            while (true)
            {
                try
                {
                    //取到该机台的楞率
                    decimal rate = 1;
                    switch (spName)
                    {
                        case "MS1":
                            if (!string.IsNullOrEmpty(stateInfo_ms1.CurFlute))
                            {
                                var ms1RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms1.CurFlute);
                                if (ms1RateInfo != null)
                                {
                                    rate = ms1RateInfo.SBW_WRate;
                                }
                            }
                            break;
                        case "MS2":

                            if (!string.IsNullOrEmpty(stateInfo_ms2.CurFlute))
                            {
                                var ms2RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms2.CurFlute);
                                if (ms2RateInfo != null)
                                {
                                    rate = ms2RateInfo.SBW_WRate;
                                }
                            }
                            break;
                        case "MS3":
                            if (!string.IsNullOrEmpty(stateInfo_ms3.CurFlute))
                            {
                                var ms3RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms3.CurFlute);
                                if (ms3RateInfo != null)
                                {
                                    rate = ms3RateInfo.SBW_WRate;
                                }
                            }
                            break;
                        default:
                            break;
                    }


                    var spInfo = _temp_SPs.FirstOrDefault(it => it.Name == spName);
                    if (spInfo == null)
                        return;



                    //都处理过一次之后，直接终止本线程
                    if (hasHandleGlue && hasHandleWrap)
                    {
                        logger.Info($"{spName}已经处理过包角和糊间隙了,糊机实材属性清空，后续糊机理论换材就不会有影响", module);
                        //糊机实材属性清空，后续糊机理论换材就不会有影响
                        spInfo.Code = "";
                        spInfo.Brand = "";
                        return;
                    }




                    //必须保证数据字典中有这些设置项
                    var dictDatas = BLLFactory<DictDataInfoManager>.Instance.Context.Queryable<DictDataInfo>()
                        .LeftJoin<DictTypeInfo>((data, type) => data.PD_TypeID == type.PD_ID).Where((data, type) => type.PD_Code == "DistanceToGU").ToList();

                    if (dictDatas == null && dictDatas.Count == 0)
                    {
                        return;
                    }

                    var isUseInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "IsUseDistanceToGU");
                    if (isUseInfo == null || isUseInfo.PD_Value.ToLower() == "false")
                    {
                        logger.Info($"{spName}处理拿到实材后的糊机换材任务终止：当前数据字典设置成不启用", module);
                        //糊机实材属性清空，后续糊机理论换材就不会有影响
                        spInfo.Code = "";
                        spInfo.Brand = "";
                        return;
                    }
                    var setDitanceInfo = dictDatas.FirstOrDefault(it => it.PD_Property == $"DistanceToGU{spName}");
                    if (setDitanceInfo == null)
                        return;
                    //设定的接纸机到糊机的距离
                    decimal setDictance = setDitanceInfo.PD_Value.ToDecimal();
                    string goMeterStr = $"{spName}_Unrolled_m";
                    var goMeterInfo = comm.PointVars.Find(it => it.VarCode == goMeterStr);
                    if (goMeterInfo == null)
                        continue;
                    decimal goMeter = goMeterInfo.VarValue.ToDecimal();
                    decimal dif = setDictance - goMeter / rate;
                    var setDitanceJudgMeterInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "DistanceToGUJudgMeter");
                    decimal judgMeter = setDitanceJudgMeterInfo.PD_Value.ToDecimal();

                    var setDistanceToGUDifInfo = dictDatas.FirstOrDefault(it => it.PD_Property == "DistanceToGUDif");
                    decimal toGUDif = setDistanceToGUDifInfo.PD_Value.ToDecimal();

                    if (dif <= judgMeter)
                    {

                        #region 糊机实材变量对应部位赋值
                        switch (spName)
                        {
                            case "LS0":
                                _temp_GU.Code_LS0 = spInfo.Code;
                                _temp_GU.Brand_LS0 = spInfo.Brand;
                                break;
                            case "LS1":
                                _temp_GU.Code_LS1 = spInfo.Code;
                                _temp_GU.Brand_LS1 = spInfo.Brand;
                                break;
                            case "LS2":
                                _temp_GU.Code_LS2 = spInfo.Code;
                                _temp_GU.Brand_LS2 = spInfo.Brand;
                                break;
                            case "LS3":
                                _temp_GU.Code_LS3 = spInfo.Code;
                                _temp_GU.Brand_LS3 = spInfo.Brand;
                                break;
                            case "MS1":
                                _temp_GU.Code_MS1 = spInfo.Code;
                                _temp_GU.Brand_MS1 = spInfo.Brand;
                                break;
                            case "MS2":
                                _temp_GU.Code_MS2 = spInfo.Code;
                                _temp_GU.Brand_MS2 = spInfo.Brand;
                                break;
                            case "MS3":
                                _temp_GU.Code_MS3 = spInfo.Code;
                                _temp_GU.Brand_MS3 = spInfo.Brand;
                                break;
                            default:
                                break;
                        }
                        #endregion

                        #region 判断是否有其他接纸机到糊机的距离和当前接纸机到糊机的距离差在xx范围内
                        var otherSPs = _temp_SPs.FindAll(it => it.Name != spName && it.Code != "");
                        foreach (var sp in otherSPs)
                        {
                            decimal rateOther = 1;
                            switch (sp.Name)
                            {
                                case "MS1":
                                    if (!string.IsNullOrEmpty(stateInfo_ms1.CurFlute))
                                    {
                                        var ms1RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms1.CurFlute);
                                        if (ms1RateInfo != null)
                                        {
                                            rateOther = ms1RateInfo.SBW_WRate;
                                        }
                                    }
                                    break;
                                case "MS2":
                                    if (!string.IsNullOrEmpty(stateInfo_ms2.CurFlute))
                                    {
                                        var ms2RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms2.CurFlute);
                                        if (ms2RateInfo != null)
                                        {
                                            rateOther = ms2RateInfo.SBW_WRate;
                                        }
                                    }
                                    break;
                                case "MS3":
                                    if (!string.IsNullOrEmpty(stateInfo_ms3.CurFlute))
                                    {
                                        var ms3RateInfo = waveSetInfos.FirstOrDefault(it => it.SBW_Wave == stateInfo_ms3.CurFlute);
                                        if (ms3RateInfo != null)
                                        {
                                            rateOther = ms3RateInfo.SBW_WRate;
                                        }
                                    }
                                    break;
                                default:
                                    break;
                            }

                            string goMeterStrSP = $"{sp.Name}_Unrolled_m";
                            var goMeterInfoSP = comm.PointVars.Find(it => it.VarCode == goMeterStrSP);
                            if (goMeterInfoSP == null)
                                continue;
                            decimal goMeterSP = goMeterInfoSP.VarValue.ToDecimal();
                            var setDitanceInfoSP = dictDatas.FirstOrDefault(it => it.PD_Property == $"DistanceToGU{sp.Name}");
                            if (setDitanceInfoSP == null)
                                continue;
                            //设定的接纸机到糊机的距离
                            decimal setDictanceSP = setDitanceInfoSP.PD_Value.ToDecimal();
                            decimal difSP = setDictanceSP - goMeterSP / rateOther;
                            if (Math.Abs(difSP - dif) < toGUDif && dif < difSP)
                            {
                                logger.Info($"{spName}:其他接纸机{sp.Name}也快要到糊机了，本次实材处理任务终止", module);
                                spInfo.Code = "";
                                spInfo.Brand = "";
                                return;
                            }
                        }
                        #endregion

                        //延迟设置表集合
                        List<AutoDelayInfo> setDelayInfos = BLLFactory<AutoDelayInfoManager>.Instance.GetList(it => it.F_Position == "DF");
                        AutoDelayInfo autoDelayInfo = new AutoDelayInfo();
                        DifWeight(spInfo, out int difweight, out string currentCode, out GuRealInfo curGuRealInfo);
                        if (difweight < 0)
                        {
                            //高换低  type =2
                            autoDelayInfo = setDelayInfos.FirstOrDefault(it => it.F_Type == 2);
                        }
                        else if (difweight >= 0)
                        {
                            //低换高 type =1
                            autoDelayInfo = setDelayInfos.FirstOrDefault(it => it.F_Type == 1);
                        }

                        decimal glueDelay = (decimal)(autoDelayInfo.F_Glue ?? 0);
                        decimal wrapDelay = (decimal)(autoDelayInfo.F_Wrap ?? 0);
                        decimal tensionDelay = (decimal)(autoDelayInfo.F_Tension ?? 0);

                        logger.Info($"接纸机{spName}-设定距离 {setDictance} -已走米数 {goMeter} /楞率 {rate} 小于 {judgMeter},进入糊机实材处理逻辑！糊间隙是否处理={hasHandleGlue}，包角是否处理={hasHandleWrap}", module);

                        //接纸机已走米数-设定值>延迟设置的时候，表示可以发送赋值命令
                        if ((dif + glueDelay) <= 0 && hasHandleGlue == false)
                        {
                            //构造糊机糊间隙立即赋值消息命令
                            PubChangeNowInfo publishInfo = new PubChangeNowInfo();
                            publishInfo.OffSetValue = 0;
                            publishInfo.Part = IPSHandlePart.GlueGu;
                            publishInfo.Width = stateInfo_gu.CurWidth;
                            publishInfo.Flute = stateInfo_gu.CurFlute;
                            publishInfo.Code = currentCode;
                            publishInfo.LastCode = stateInfo_gu.LastCode;
                            publishInfo.LastWidth = stateInfo_gu.LastWidth;
                            publishInfo.LastFlute = stateInfo_gu.LastFlute;
                            publishInfo.BrandMS1 = curGuRealInfo.Brand_MS1;
                            publishInfo.BrandLS1 = curGuRealInfo.Brand_LS1;
                            publishInfo.BrandMS2 = curGuRealInfo.Brand_MS2;
                            publishInfo.BrandLS2 = curGuRealInfo.Brand_LS2;
                            publishInfo.BrandMS3 = curGuRealInfo.Brand_MS3;
                            publishInfo.BrandLS3 = curGuRealInfo.Brand_LS3;
                            PubChangeNow(publishInfo);
                            hasHandleGlue = true;
                            logger.Info($"糊机实材={currentCode},楞型={stateInfo_gu.CurFlute},门幅={stateInfo_gu.CurWidth};发送命令给糊机胶水赋值", module);
                        }

                        if ((dif + wrapDelay) <= 0 && hasHandleWrap == false)
                        {
                            //构造糊机包角立即赋值消息命令
                            PubChangeNowInfo publishInfo = new PubChangeNowInfo();
                            publishInfo.OffSetValue = 0;
                            publishInfo.Part = IPSHandlePart.WrapGu;
                            publishInfo.Width = stateInfo_gu.CurWidth;
                            publishInfo.Flute = stateInfo_gu.CurFlute;
                            publishInfo.Code = currentCode;
                            publishInfo.LastCode = stateInfo_gu.LastCode;
                            publishInfo.LastWidth = stateInfo_gu.LastWidth;
                            publishInfo.LastFlute = stateInfo_gu.LastFlute;
                            publishInfo.BrandMS1 = curGuRealInfo.Brand_MS1;
                            publishInfo.BrandLS1 = curGuRealInfo.Brand_LS1;
                            publishInfo.BrandMS2 = curGuRealInfo.Brand_MS2;
                            publishInfo.BrandLS2 = curGuRealInfo.Brand_LS2;
                            publishInfo.BrandMS3 = curGuRealInfo.Brand_MS3;
                            publishInfo.BrandLS3 = curGuRealInfo.Brand_LS3;
                            PubChangeNow(publishInfo);

                            PubChangeNowInfo publishInfoEx = new PubChangeNowInfo();
                            publishInfoEx.OffSetValue = 0;
                            publishInfoEx.Part = IPSHandlePart.WrapGu_Add2;
                            publishInfoEx.Width = stateInfo_gu.CurWidth;
                            publishInfoEx.Flute = stateInfo_gu.CurFlute;
                            publishInfoEx.Code = currentCode;
                            publishInfoEx.LastCode = stateInfo_gu.LastCode;
                            publishInfoEx.LastWidth = stateInfo_gu.LastWidth;
                            publishInfoEx.LastFlute = stateInfo_gu.LastFlute;
                            publishInfoEx.BrandMS1 = curGuRealInfo.Brand_MS1;
                            publishInfoEx.BrandLS1 = curGuRealInfo.Brand_LS1;
                            publishInfoEx.BrandMS2 = curGuRealInfo.Brand_MS2;
                            publishInfoEx.BrandLS2 = curGuRealInfo.Brand_LS2;
                            publishInfoEx.BrandMS3 = curGuRealInfo.Brand_MS3;
                            publishInfoEx.BrandLS3 = curGuRealInfo.Brand_LS3;
                            PubChangeNow(publishInfoEx);

                            hasHandleWrap = true;
                            logger.Info($"糊机实材={currentCode},楞型={stateInfo_gu.CurFlute},门幅={stateInfo_gu.CurWidth};发送命令给糊机包角赋值", module);

                        }
                    }
                }
                catch (Exception ex)
                {
                    logger.Error($"{spName}拿到实材，处理糊机实材异常：{ex}", module);
                }
                finally
                {
                    await Task.Delay(1000);
                }
            }
        }

        /// <summary>
        /// 上一次糊机的克重-本次换实材的糊机克重
        /// </summary>
        /// <param name="info">实材接纸机信息</param>
        /// <returns></returns>
        private void DifWeight(SPRealInfo info, out int dif, out string curGUCode, out GuRealInfo cur)
        {
            cur = new GuRealInfo();
            List<string> papers = new List<string>();
            if (stateInfo_gu.CurCode.Contains("."))
            {
                papers = stateInfo_gu.CurCode.Split('.').ToList();
            }
            else
            {
                papers = stateInfo_gu.CurCode.ToCharArray().Select(c => c.ToString()).ToList();
            }
            GuRealInfo last = new GuRealInfo();
            for (int i = 0; i < papers.Count; i++)
            {
                switch (i)
                {
                    case 0:
                        if (_temp_GU.Brand_LS0 != "")
                        {
                            last.Code_LS0 = _temp_GU.Code_LS0;
                            last.Brand_LS0 = _temp_GU.Brand_LS0;

                            cur.Code_LS0 = _temp_GU.Code_LS0;
                            cur.Brand_LS0 = _temp_GU.Brand_LS0;
                        }
                        else
                        {
                            last.Code_LS0 = papers[i];
                            last.Brand_LS0 = "";

                            cur.Code_LS0 = papers[i];
                            cur.Brand_LS0 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_LS0 = papers[i];
                            last.Brand_LS0 = "";

                            cur.Code_LS0 = papers[i];
                            cur.Brand_LS0 = "";
                        }
                        break;
                    case 1:
                        if (_temp_GU.Brand_MS1 != "")
                        {
                            last.Code_MS1 = _temp_GU.Code_MS1;
                            last.Brand_MS1 = _temp_GU.Brand_MS1;

                            cur.Code_MS1 = _temp_GU.Code_MS1;
                            cur.Brand_MS1 = _temp_GU.Brand_MS1;
                        }
                        else
                        {
                            last.Code_MS1 = papers[i];
                            last.Brand_MS1 = "";

                            cur.Code_MS1 = papers[i];
                            cur.Brand_MS1 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_MS1 = papers[i];
                            last.Brand_MS1 = "";

                            cur.Code_MS1 = papers[i];
                            cur.Brand_MS1 = "";
                        }
                        break;
                    case 2:
                        if (_temp_GU.Brand_LS1 != "")
                        {
                            last.Code_LS1 = _temp_GU.Code_LS1;
                            last.Brand_LS1 = _temp_GU.Brand_LS1;

                            cur.Code_LS1 = _temp_GU.Code_LS1;
                            cur.Brand_LS1 = _temp_GU.Brand_LS1;
                        }
                        else
                        {
                            last.Code_LS1 = papers[i];
                            last.Brand_LS1 = "";

                            cur.Code_LS1 = papers[i];
                            cur.Brand_LS1 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_LS1 = papers[i];
                            last.Brand_LS1 = "";

                            cur.Code_LS1 = papers[i];
                            cur.Brand_LS1 = "";
                        }
                        break;
                    case 3:
                        if (_temp_GU.Brand_MS2 != "")
                        {
                            last.Code_MS2 = _temp_GU.Code_MS2;
                            last.Brand_MS2 = _temp_GU.Brand_MS2;

                            cur.Code_MS2 = _temp_GU.Code_MS2;
                            cur.Brand_MS2 = _temp_GU.Brand_MS2;
                        }
                        else
                        {
                            last.Code_MS2 = papers[i];
                            last.Brand_MS2 = "";

                            cur.Code_MS2 = papers[i];
                            cur.Brand_MS2 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_MS2 = papers[i];
                            last.Brand_MS2 = "";

                            cur.Code_MS2 = papers[i];
                            cur.Brand_MS2 = "";
                        }
                        break;
                    case 4:
                        if (_temp_GU.Brand_LS2 != "")
                        {
                            last.Code_LS2 = _temp_GU.Code_LS2;
                            last.Brand_LS2 = _temp_GU.Brand_LS2;

                            cur.Code_LS2 = _temp_GU.Code_LS2;
                            cur.Brand_LS2 = _temp_GU.Brand_LS2;
                        }
                        else
                        {
                            last.Code_LS2 = papers[i];
                            last.Brand_LS2 = "";

                            cur.Code_LS2 = papers[i];
                            cur.Brand_LS2 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_LS2 = papers[i];
                            last.Brand_LS2 = "";

                            cur.Code_LS2 = papers[i];
                            cur.Brand_LS2 = "";
                        }
                        break;
                    case 5:
                        if (_temp_GU.Brand_MS3 != "")
                        {
                            last.Code_MS3 = _temp_GU.Code_MS3;
                            last.Brand_MS3 = _temp_GU.Brand_MS3;

                            cur.Code_MS3 = _temp_GU.Code_MS3;
                            cur.Brand_MS3 = _temp_GU.Brand_MS3;
                        }
                        else
                        {
                            last.Code_MS3 = papers[i];
                            last.Brand_MS3 = "";

                            cur.Code_MS3 = papers[i];
                            cur.Brand_MS3 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_MS3 = papers[i];
                            last.Brand_MS3 = "";

                            cur.Code_MS3 = papers[i];
                            cur.Brand_MS3 = "";
                        }
                        break;
                    case 6:
                        if (_temp_GU.Brand_LS3 != "")
                        {
                            last.Code_LS3 = _temp_GU.Code_LS3;
                            last.Brand_LS3 = _temp_GU.Brand_LS3;

                            cur.Code_LS3 = _temp_GU.Code_LS3;
                            cur.Brand_LS3 = _temp_GU.Brand_LS3;
                        }
                        else
                        {
                            last.Code_LS3 = papers[i];
                            last.Brand_LS3 = "";

                            cur.Code_LS3 = papers[i];
                            cur.Brand_LS3 = "";
                        }

                        if (papers[i] == "-")
                        {
                            last.Code_LS3 = papers[i];
                            last.Brand_LS3 = "";

                            cur.Code_LS3 = papers[i];
                            cur.Brand_LS3 = "";
                        }
                        break;
                    default:
                        break;
                }
            }
            switch (info.Name)
            {
                case "LS0":
                    if (cur.Code_LS0 != "-")
                    {
                        cur.Code_LS0 = info.Code;
                        cur.Brand_LS0 = info.Brand;
                    }
                    break;
                case "LS1":
                    if (cur.Code_LS1 != "-")
                    {
                        cur.Code_LS1 = info.Code;
                        cur.Brand_LS1 = info.Brand;
                    }
                    break;
                case "LS2":
                    if (cur.Code_LS2 != "-")
                    {
                        cur.Code_LS2 = info.Code;
                        cur.Brand_LS2 = info.Brand;
                    }
                    break;
                case "LS3":
                    if (cur.Code_LS3 != "-")
                    {
                        cur.Code_LS3 = info.Code;
                        cur.Brand_LS3 = info.Brand;
                    }
                    break;
                case "MS1":
                    if (cur.Code_MS1 != "-")
                    {
                        cur.Code_MS1 = info.Code;
                        cur.Brand_MS1 = info.Brand;
                    }
                    break;
                case "MS2":
                    if (cur.Code_MS2 != "-")
                    {
                        cur.Code_MS2 = info.Code;
                        cur.Brand_MS2 = info.Brand;
                    }
                    break;
                case "MS3":
                    if (cur.Code_MS3 != "-")
                    {
                        cur.Code_MS3 = info.Code;
                        cur.Brand_MS3 = info.Brand;
                    }
                    break;
                default:
                    break;
            }

            string lastCode = "";
            if (last.Code_LS0 != "")
            {
                lastCode = last.Code_LS0;
            }
            if (last.Code_MS1 != "")
            {
                lastCode += $".{last.Code_MS1}";
            }
            if (last.Code_LS1 != "")
            {
                lastCode += $".{last.Code_LS1}";
            }
            if (last.Code_MS2 != "")
            {
                lastCode += $".{last.Code_MS2}";
            }
            if (last.Code_LS2 != "")
            {
                lastCode += $".{last.Code_LS2}";
            }
            if (last.Code_MS3 != "")
            {
                lastCode += $".{last.Code_MS3}";
            }
            if (last.Code_LS3 != "")
            {
                lastCode += $".{last.Code_LS3}";
            }

            string curCode = "";
            if (cur.Code_LS0 != "")
            {
                curCode = cur.Code_LS0;
            }
            if (cur.Code_MS1 != "")
            {
                curCode += $".{cur.Code_MS1}";
            }
            if (cur.Code_LS1 != "")
            {
                curCode += $".{cur.Code_LS1}";
            }
            if (cur.Code_MS2 != "")
            {
                curCode += $".{cur.Code_MS2}";
            }
            if (cur.Code_LS2 != "")
            {
                curCode += $".{cur.Code_LS2}";
            }
            if (cur.Code_MS3 != "")
            {
                curCode += $".{cur.Code_MS3}";
            }
            if (cur.Code_LS3 != "")
            {
                curCode += $".{cur.Code_LS3}";
            }
            int x1 = BLLFactory<PaperCodeInfoManager>.Instance.GetSumWeight(lastCode);
            int x2 = BLLFactory<PaperCodeInfoManager>.Instance.GetSumWeight(curCode);

            dif = x2 - x1;
            curGUCode = curCode;
        }

        #endregion <方法>

        #region <事件>
        /// <summary>
        /// 换材发布消息事件
        /// </summary>
        public event EventHandler OnPublish;
        /// <summary>
        /// 换材消息发布函数
        /// </summary>
        /// <param name="msg"></param>
        public void Publish(List<PublishInfo> msg)
        {
            // 触发事件，通知所有订阅者
            if (OnPublish != null)
                OnPublish(msg, EventArgs.Empty);
        }


        /// <summary>
        /// 立刻赋值消息事件
        /// </summary>
        public event EventHandler OnPubChangeNow;

        /// <summary>
        /// 立刻赋值消息发布函数
        /// </summary>
        /// <param name="msg"></param>
        public void PubChangeNow(PubChangeNowInfo msg)
        {
            // 触发事件，通知所有订阅者
            if (OnPubChangeNow != null)
                OnPubChangeNow(msg, EventArgs.Empty);
        }


        /// <summary>
        /// 接纸机获取实际材质事件
        /// </summary>
        public event EventHandler OnGetRealPaper;
        /// <summary>
        /// 接纸机获取实际材质消息发布函数
        /// </summary>
        public void PubGetRealPaper(SPPaperInfo msg)
        {
            // 触发事件，通知所有订阅者
            if (OnGetRealPaper != null)
                OnGetRealPaper(msg, EventArgs.Empty);
        }


        /// <summary>
        /// 换材消息
        /// </summary>
        public event EventHandler OnPubChangePaper;

        /// <summary>
        /// 换材发布订阅
        /// </summary>
        /// <param name="msg"></param>
        public void PubChangePaper(PartPaperCode msg)
        {
            // 触发事件，通知所有订阅者
            if (OnPubChangePaper != null)
                OnPubChangePaper(msg, EventArgs.Empty);
        }


        /// <summary>
        /// 同材换卷or残卷换卷事件
        /// </summary>
        public event EventHandler OnChangeRollRemain;
        /// <summary>
        /// 同材换卷or残卷换卷事件消息发布函数
        /// </summary>
        public void ChangeRollRemain(ChangeRollRemainEventModel msg)
        {
            // 触发事件，通知所有订阅者
            if (OnChangeRollRemain != null)
                OnChangeRollRemain(msg, EventArgs.Empty);
        }
        #endregion <事件>
    }

    /// <summary>
    /// 各设备的历史以及当前状态值数据模型
    /// </summary>
    public class DriveStateInfo
    {
        /// <summary>
        /// 下批材质 只有LS0需要使用
        /// </summary>
        public string NextBachCode { get; set; } = "";
        /// <summary>
        /// 上一次使用的材质编码
        /// </summary>
        public string LastCode { get; set; } = "-";
        /// <summary>
        /// 上一次使用的楞型（SF为单瓦楞型 DF为订单楞型）
        /// </summary>
        public string LastFlute { get; set; } = "";
        /// <summary>
        /// 上一次使用的门幅
        /// </summary>
        public int LastWidth { get; set; } = 0;

        /// <summary>
        /// 对应的全材质，这个属性适用于面纸
        /// </summary>
        public string CodeALl { get; set; } = "";

        /// <summary>
        /// 本次使用的材质编码
        /// </summary>
        public string CurCode { get; set; } = "-";
        /// <summary>
        /// 本次使用的楞型（芯纸为单瓦楞型 里纸=芯纸 面纸=订单楞型 DF为订单楞型）
        /// </summary>
        public string CurFlute { get; set; } = "";
        /// <summary>
        /// 本次使用的门幅
        /// </summary>
        public int CurWidth { get; set; } = 0;

        /// <summary>
        /// 换卷标识
        /// </summary>
        public bool IsChangeRoll { get; set; } = false;

        /// <summary>
        /// 换材准备中标识
        /// </summary>
        public bool VChangePaper { get; set; } = false;

        /// <summary>
        /// 换材标识
        /// </summary>
        public bool IsChangePaper { get; set; } = false;

        /// <summary>
        /// 本部位上一次PLC拿到的换卷信号值
        /// </summary>
        public string LastPlcChangeRoll_Part { get; set; } = "";

        /// <summary>
        /// 接纸机同材剩余判断换材准备区间1
        /// </summary>
        public int SPRange1 { get; set; } = 0;
        /// <summary>
        /// 接纸机同材剩余判断换材准备区间2
        /// </summary>
        public int SPRange2 { get; set; } = 0;
        /// <summary>
        /// 接纸机同材剩余判断换材准备区间3
        /// </summary>
        public int SPRange3 { get; set; } = 0;

        /// <summary>
        /// 糊机同材剩余判断换材准备区间1
        /// </summary>
        public int GuRange1 { get; set; } = 0;
        /// <summary>
        /// 糊机同材剩余判断换材准备区间2
        /// </summary>
        public int GuRange2 { get; set; } = 0;
        /// <summary>
        /// 糊机同材剩余判断换材准备区间3
        /// </summary>
        public int GuRange3 { get; set; } = 0;

        /// <summary>
        /// 面纸品牌
        /// </summary>
        public string BrandLS0 { get; set; } = "";
        /// <summary>
        /// 1芯品牌
        /// </summary>
        public string BrandMS1 { get; set; } = "";
        /// <summary>
        /// 1里品牌
        /// </summary>
        public string BrandLS1 { get; set; } = "";
        /// <summary>
        /// 2芯品牌
        /// </summary>
        public string BrandMS2 { get; set; } = "";
        /// <summary>
        /// 2里品牌
        /// </summary>
        public string BrandLS2 { get; set; } = "";
        /// <summary>
        /// 3芯品牌
        /// </summary>
        public string BrandMS3 { get; set; } = "";
        /// <summary>
        /// 3里品牌
        /// </summary>
        public string BrandLS3 { get; set; } = "";

        //以下为后加的属性，主要解决接纸机带材很多+实际材质给的不正确的情况

        /// <summary>
        /// 下批理论材质
        /// </summary>
        public string NextBatchTheoryCode { get; set; } = "";

        /// <summary>
        /// 下批理论全材质
        /// </summary>
        public string NextBatchTheoryCodeAll { get; set; } = "";
        /// <summary>
        /// 下批理论门幅
        /// </summary>
        public int NextBatchTheoryWidth { get; set; }
        /// <summary>
        /// 下批理论楞型
        /// </summary>
        public string NextBatchTheoryFlute { get; set; } = "";
    }


    /// <summary>
    /// 糊机实际材质对象模型
    /// </summary>
    public class GuRealInfo
    {
        /// <summary>
        /// 面纸材质
        /// </summary>
        public string Code_LS0 { get; set; } = "";

        /// <summary>
        /// MS1材质
        /// </summary>
        public string Code_MS1 { get; set; } = "";

        /// <summary>
        /// LS1材质
        /// </summary>
        public string Code_LS1 { get; set; } = "";

        /// <summary>
        /// MS2材质
        /// </summary>
        public string Code_MS2 { get; set; } = "";

        /// <summary>
        /// LS2材质
        /// </summary>
        public string Code_LS2 { get; set; } = "";

        /// <summary>
        /// MS3材质
        /// </summary>
        public string Code_MS3 { get; set; } = "";

        /// <summary>
        /// LS3材质
        /// </summary>
        public string Code_LS3 { get; set; } = "";

        /// <summary>
        /// 面纸品牌
        /// </summary>
        public string Brand_LS0 { get; set; } = "";

        /// <summary>
        /// MS1品牌
        /// </summary>
        public string Brand_MS1 { get; set; } = "";

        /// <summary>
        /// LS1品牌
        /// </summary>
        public string Brand_LS1 { get; set; } = "";

        /// <summary>
        /// MS2品牌
        /// </summary>
        public string Brand_MS2 { get; set; } = "";

        /// <summary>
        /// LS2品牌
        /// </summary>
        public string Brand_LS2 { get; set; } = "";

        /// <summary>
        /// MS3品牌
        /// </summary>
        public string Brand_MS3 { get; set; } = "";

        /// <summary>
        /// LS3品牌
        /// </summary>
        public string Brand_LS3 { get; set; } = "";
    }


    /// <summary>
    /// 接纸机实际材质记录对象模型
    /// </summary>
    public class SPRealInfo
    {
        /// <summary>
        /// 接纸机名称
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// 接纸机实际材质
        /// </summary>
        public string Code { get; set; } = "";

        /// <summary>
        /// 接纸机实际材质品牌
        /// </summary>
        public string Brand { get; set; } = "";
    }
}