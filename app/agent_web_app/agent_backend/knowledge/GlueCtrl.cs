#region << 版 本 注 释 >>
/*----------------------------------------------------------------
 * 版权所有 (c) 2024  NJRN 保留所有权利。
 * CLR版本：4.0.30319.42000
 * 机器名称：BCCNSHGNB226
 * 公司名称：
 * 命名空间：BTS.Server.Start.IPSBizs.NewCtrl
 * 唯一标识：f9437121-30f4-48c0-bc6c-c324a2d579de
 * 文件名：GlueCtrl
 * 当前用户域：BHS
 * 
 * 创建者：QZhou
 * 创建时间：2024/4/11 8:53:28
 * 版本：V1.0.0
 * 描述：糊间隙业务类
 * 订阅者
 *
 * ----------------------------------------------------------------
 * 修改人：
 * 时间：
 * 修改说明：
 *
 * 版本：V1.0.1
 *----------------------------------------------------------------*/
#endregion << 版 本 注 释 >>

using BTS.Commons;
using BTS.Dtos;
using BTS.Entites;
using BTS.Logs;
using BTS.Server.Core;
using BTS.Services;
using BTS.Services.Services.IPSNew;
using System.Text;


namespace BTS.Server
{
    /// <summary>
    /// 糊间隙业务类
    /// </summary>
    public class GlueCtrl
    {
        #region <常量>
        private const string module = "糊间隙控制模块";
        private object _lockGu = new object();
        #endregion <常量>

        #region <变量>
        private Log logger = LogHelper.GetLogger(typeof(GlueCtrl));
        private IPSMainCtrl mainCtrl;
        //private TcpService tcpService;
        private DriverLink comm;
        private CancellationTokenSource cts_gu = new CancellationTokenSource();
        private CancellationTokenSource cts_sf1 = new CancellationTokenSource();
        private CancellationTokenSource cts_sf2 = new CancellationTokenSource();
        private CancellationTokenSource cts_sf3 = new CancellationTokenSource();

        //存储各部位糊间隙和车速的配方
        private CancellationTokenSource cts_glue = new CancellationTokenSource();
        //private List<GluePositionSpeedValue> tempGu1 = new List<GluePositionSpeedValue>();
        //private List<GluePositionSpeedValue> tempGu2 = new List<GluePositionSpeedValue>();
        //private List<GluePositionSpeedValue> tempGu3 = new List<GluePositionSpeedValue>();
        //private List<GluePositionSpeedValue> tempSf1 = new List<GluePositionSpeedValue>();
        //private List<GluePositionSpeedValue> tempSf2 = new List<GluePositionSpeedValue>();
        //private List<GluePositionSpeedValue> tempSf3 = new List<GluePositionSpeedValue>();
        #endregion <变量>

        #region <属性>
        #endregion <属性>

        #region <构造方法和析构方法>
        public GlueCtrl(IPSMainCtrl _mainCtrl, DriverLink _comm)
        {
            comm = _comm;
            mainCtrl = _mainCtrl;
            mainCtrl.OnPublish += HandlePubMsg;
            mainCtrl.OnPubChangeNow += HandleChangeNow;
            CalGlueRealTime();
        }


        #endregion <构造方法和析构方法>

        #region <方法>
        /// <summary>
        /// 订阅主控类发送的赋值消息，具体处理函数
        /// </summary>
        /// <param name="sender">消息</param>
        /// <param name="e"></param>
        private void HandlePubMsg(object sender, EventArgs e)
        {
            if (sender == null)
                return;
            //拿到发送过来的换材消息命令
            List<PublishInfo> msg = sender as List<PublishInfo>;
            foreach (PublishInfo info in msg)
            {
                switch (info.Part)
                {
                    case IPSHandlePart.GlueGu:
                        //从消息中找到关于糊机糊间隙赋值的命令
                        HandleGuGlueMsg(info);
                        break;
                    case IPSHandlePart.GlueSF1:
                        //从消息中找到关于单面机SF1糊间隙赋值的命令
                        HandleSF1GlueMsg(info);
                        break;
                    case IPSHandlePart.GlueSF2:
                        //从消息中找到关于单面机SF2糊间隙赋值的命令
                        HandleSF2GlueMsg(info);
                        break;
                    case IPSHandlePart.GlueSF3:
                        //从消息中找到关于单面机SF3糊间隙赋值的命令
                        HandleSF3GlueMsg(info);
                        break;
                    default:
                        break;
                }
            }
        }

        /// <summary>
        /// 处理糊机糊间隙换材消息
        /// </summary>
        private void HandleGuGlueMsg(PublishInfo info)
        {
            logger.Info($"进入HandleGuGlueMsg ，本次为正常换材，材质={info.Code},楞型={info.Flute}", module);
            //处理之前先终止之前已经开启的线程任务，再开启新的线程任务
            cts_gu.Cancel();
            cts_gu = new CancellationTokenSource();
            CancellationToken token = cts_gu.Token;
            Task.Run(() => { SetGlueGu(info, token); }, token);
        }
        /// <summary>
        /// 糊机糊间隙具体赋值函数
        /// </summary>
        /// <param name="info"></param>
        private void SetGlueGu(PublishInfo info, CancellationToken token, bool isChangeNow = false, bool isFirst = false)
        {
            try
            {
                if (token.IsCancellationRequested)
                {
                    logger.Info("糊机糊间隙赋值任务取消,因为该期间内又收到一个新的糊机糊间隙赋值任务", module);
                    return;
                }
                logger.Info($"进入 SetGlueGu 准备点位赋值，材质={info.Code},楞型={info.Flute},品牌LS0={info.BrandLS0},品牌MS1={info.BrandMS1},品牌LS1={info.BrandLS1},品牌MS2={info.BrandMS2},品牌LS2={info.BrandLS2},品牌MS3={info.BrandMS3},品牌LS3={info.BrandLS3}", module);
                List<string> paperOldList = new List<string>();
                List<string> paperList = new List<string>();
                string paper = "";
                //拿到需要赋值的 材质 门幅 楞型，查找QDM系数,
                //这里拿到的楞型是订单全楞型，材质是订单材质
                //单材质编码的项目时，只要把 -替换成空查询即可
                //多位材质编码的项目是，进行材质转换，P1.3.8.-.-  转成P1.3.8
                if (info.Code.Contains("."))
                {
                    paperOldList = info.Code.Split('.').ToList();
                    paperList = info.Code.Split('.').Where(it => it != "-").ToList();
                    paper = string.Join(".", paperList);
                }
                else
                {
                    paperOldList = info.Code.ToCharArray().Select(it => it.ToString()).ToList();
                    paper = info.Code.Replace("-", "");
                    foreach (var c in paper.ToCharArray())
                    {
                        paperList.Add(c.ToString());
                    }
                }
                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
                //根据当前用户选择的是哪几个糊机糊间隙部位在使用，得到具体的糊间隙配方值，发送给机器，
                //根据延迟设置那边的设置情况，按照高换低还是低换高，来确定是延迟赋值还是提前赋值
                decimal lastWeight = BLLFactory<PaperCodeInfoManager>.Instance.GetSumWeight(info.LastCode);
                decimal curWeight = BLLFactory<PaperCodeInfoManager>.Instance.GetSumWeight(info.Code);

                string pCodeFloor1 = "";//1层材质
                string pCodeFloor2 = "";//2层材质
                string pCodeFloor3 = "";//3层材质

                string brandpCodeFloor1 = "";//1层材质品牌
                string brandpCodeFloor2 = "";//2层材质品牌
                string brandpCodeFloor3 = "";//3层材质品牌

                switch (paperList.Count)
                {
                    case 3:
                        pCodeFloor1 = paperList[0] + "/" + paperList[1];

                        if (paperOldList.Count == 3)
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                        }
                        else if (paperOldList.Count == 5 && paperOldList[1] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                        }
                        else if (paperOldList.Count == 5 && paperOldList[1] == "-" && paperOldList[3] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] != "-" && paperOldList[3] == "-" && paperOldList[5] == "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] == "-" && paperOldList[3] != "-" && paperOldList[5] == "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] == "-" && paperOldList[3] == "-" && paperOldList[5] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS3;
                        }
                        break;
                    case 4:
                        pCodeFloor1 = paperList[1] + "/" + paperList[2];

                        if (paperOldList.Count == 5)
                        {
                            brandpCodeFloor1 = info.BrandLS1 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] != "-" && paperOldList[3] != "-" && paperOldList[5] == "-")
                        {
                            brandpCodeFloor1 = info.BrandLS1 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] != "-" && paperOldList[3] == "-" && paperOldList[5] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS1 + "/" + info.BrandMS3;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] == "-" && paperOldList[3] != "-" && paperOldList[5] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS2 + "/" + info.BrandMS3;
                        }

                        break;
                    case 5:
                        pCodeFloor1 = paperList[0] + "/" + paperList[1];
                        pCodeFloor2 = paperList[2] + "/" + paperList[3];

                        if (paperOldList.Count == 5)
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                            brandpCodeFloor2 = info.BrandLS1 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] != "-" && paperOldList[3] != "-" && paperOldList[5] == "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                            brandpCodeFloor2 = info.BrandLS1 + "/" + info.BrandMS2;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] != "-" && paperOldList[3] == "-" && paperOldList[5] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                            brandpCodeFloor2 = info.BrandLS1 + "/" + info.BrandMS3;
                        }
                        else if (paperOldList.Count == 7 && paperOldList[1] == "-" && paperOldList[3] != "-" && paperOldList[5] != "-")
                        {
                            brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS2;
                            brandpCodeFloor2 = info.BrandLS2 + "/" + info.BrandMS3;
                        }
                        break;
                    case 6:
                        pCodeFloor1 = paperList[1] + "/" + paperList[2];
                        pCodeFloor2 = paperList[2] + "/" + paperList[3];

                        brandpCodeFloor1 = info.BrandLS1 + "/" + info.BrandMS2;
                        brandpCodeFloor2 = info.BrandLS2 + "/" + info.BrandMS3;
                        break;
                    case 7:
                        pCodeFloor1 = paperList[0] + "/" + paperList[1];
                        pCodeFloor2 = paperList[2] + "/" + paperList[3];
                        pCodeFloor3 = paperList[4] + "/" + paperList[5];

                        brandpCodeFloor1 = info.BrandLS0 + "/" + info.BrandMS1;
                        brandpCodeFloor2 = info.BrandLS1 + "/" + info.BrandMS2;
                        brandpCodeFloor3 = info.BrandLS2 + "/" + info.BrandMS3;
                        break;
                    default:
                        break;
                }

                //设备部位 糊机1层的车速系数
                var speedCoefGu1 = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.Gu1).OrderBy(it => it.Speed).ToList();
                //设备部位 糊机2层的车速系数
                var speedCoefGu2 = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.Gu2).OrderBy(it => it.Speed).ToList();
                //设备部位 糊机3层的车速系数
                var speedCoefGu3 = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.Gu3).OrderBy(it => it.Speed).ToList();
                //基础设置
                var glueSetInfos = BLLFactory<GlueGuSetInfoManager>.Instance.GetList(it => it.Flute == info.Flute);


                //把用户当前启用的糊机糊间隙部位放入顺序列表中
                List<GluePositionEnum> driverList = new List<GluePositionEnum>();
                if (formSetInfo.F_Glue_GU_1st_Form_IsOn)
                    driverList.Add(GluePositionEnum.Gu1);
                if (formSetInfo.F_Glue_GU_2nd_Form_IsOn)
                    driverList.Add(GluePositionEnum.Gu2);
                if (formSetInfo.F_Glue_GU_3rd_Form_IsOn)
                    driverList.Add(GluePositionEnum.Gu3);

                //原纸资料全部记录
                var allPapers = BLLFactory<PaperCodeInfoManager>.Instance.GetList();
                //供应商材质档案
                var brandPapers = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((b, p) => b.F_PaperCodeID == p.SPC_ID).Select((b, p) => new { Code = p.SPC_Code, Brand = b.F_Brand, WrapWgt = b.F_WrapWeight, ID = b.F_ID }).ToList();

                var brandGlueInfos = BLLFactory<GlueGuBrandSetInfoManager>.Instance.GetList();

                List<PaperPositionCodeDriverInfo> mapList = new List<PaperPositionCodeDriverInfo>();
                bool isMapErr = false;//用户界面勾选的使用部位和材质匹配不上标识
                int index = 0;
                if (!string.IsNullOrEmpty(pCodeFloor1))
                {
                    if (index >= driverList.Count())
                    {
                        StringBuilder sb1 = new StringBuilder();
                        sb1.AppendLine("糊机糊间隙设备部位和材质匹配失败！");
                        sb1.AppendLine("用户勾选的糊机糊间隙设备部位:");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_1st_Form_IsOn}");
                        sb1.AppendLine($"中层={formSetInfo.F_Glue_GU_2nd_Form_IsOn}");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_3rd_Form_IsOn}");
                        sb1.AppendLine($"当前糊间隙材质情况：1层={pCodeFloor1}；2层={pCodeFloor2}；3层={pCodeFloor3}");
                        logger.Warn(sb1.ToString(), module);
                        sb1 = null;
                        isMapErr = true;
                    }
                    else
                    {
                        PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                        mapInfo.PaperPosition = PaperPositionEnum.Floor1;
                        mapInfo.PaperCode = pCodeFloor1;
                        mapInfo.Driver = driverList[index];

                        decimal brandOffset0 = 0;
                        decimal brandOffset1 = 0;
                        //计算出克重
                        int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor1.Split('/')[0]).SPC_GlueWeight ?? 0;
                        if (brandPapers.Exists(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]))
                        {
                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset0 += (brandOffsetInfo?.Offset1 ?? 0);
                            //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]).GuleWgt;
                        }

                        int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor1.Split('/')[1]).SPC_GlueWeight ?? 0;

                        if (brandPapers.Exists(it => it.Code == pCodeFloor1.Split('/')[1] && it.Brand == brandpCodeFloor1.Split('/')[1]))
                        {
                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[1] && it.Brand == brandpCodeFloor1.Split('/')[1]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset1 += (brandOffsetInfo?.Offset1 ?? 0);
                            //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[1]).GuleWgt;
                        }

                        mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);
                        int wgt = msWgt + lsWgt;
                        mapInfo.Weight = wgt;

                        mapList.Add(mapInfo);
                        index++;
                    }

                }
                if (!string.IsNullOrEmpty(pCodeFloor2))
                {
                    if (index >= driverList.Count())
                    {
                        StringBuilder sb1 = new StringBuilder();
                        sb1.AppendLine("糊机糊间隙设备部位和材质匹配失败！");
                        sb1.AppendLine("用户勾选的糊机糊间隙设备部位:");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_1st_Form_IsOn}");
                        sb1.AppendLine($"中层={formSetInfo.F_Glue_GU_2nd_Form_IsOn}");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_3rd_Form_IsOn}");
                        sb1.AppendLine($"当前糊间隙材质情况：1层={pCodeFloor1}；2层={pCodeFloor2}；3层={pCodeFloor3}");
                        logger.Warn(sb1.ToString(), module);
                        sb1 = null;
                        isMapErr = true;
                    }
                    else
                    {
                        PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                        mapInfo.PaperPosition = PaperPositionEnum.Floor2;
                        mapInfo.PaperCode = pCodeFloor2;
                        mapInfo.Driver = driverList[index];

                        decimal brandOffset0 = 0;
                        decimal brandOffset1 = 0;
                        //计算出克重
                        int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor2.Split('/')[0]).SPC_GlueWeight ?? 0;

                        if (brandPapers.Exists(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]))
                        {
                            //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]).GuleWgt;

                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset0 += (brandOffsetInfo?.Offset2 ?? 0);
                        }


                        int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor2.Split('/')[1]).SPC_GlueWeight ?? 0;

                        if (brandPapers.Exists(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]))
                        {
                            //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]).GuleWgt;
                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset1 += (brandOffsetInfo?.Offset2 ?? 0);
                        }
                        mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);
                        int wgt = msWgt + lsWgt;
                        mapInfo.Weight = wgt;
                        mapList.Add(mapInfo);
                        index++;
                    }

                }
                if (!string.IsNullOrEmpty(pCodeFloor3))
                {
                    if (index >= driverList.Count())
                    {
                        StringBuilder sb1 = new StringBuilder();
                        sb1.AppendLine("糊机糊间隙设备部位和材质匹配失败！");
                        sb1.AppendLine("用户勾选的糊机糊间隙设备部位:");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_1st_Form_IsOn}");
                        sb1.AppendLine($"中层={formSetInfo.F_Glue_GU_2nd_Form_IsOn}");
                        sb1.AppendLine($"下层={formSetInfo.F_Glue_GU_3rd_Form_IsOn}");
                        sb1.AppendLine($"当前糊间隙材质情况：1层={pCodeFloor1}；2层={pCodeFloor2}；3层={pCodeFloor3}");
                        logger.Warn(sb1.ToString(), module);
                        sb1 = null;
                        isMapErr = true;
                    }
                    else
                    {
                        PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                        mapInfo.PaperPosition = PaperPositionEnum.Floor3;
                        mapInfo.PaperCode = pCodeFloor3;
                        mapInfo.Driver = driverList[index];
                        decimal brandOffset0 = 0;
                        decimal brandOffset1 = 0;
                        //计算出克重
                        int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor3.Split('/')[0]).SPC_GlueWeight ?? 0;

                        if (brandPapers.Exists(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]))
                        {
                            //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]).GuleWgt;

                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset0 += (brandOffsetInfo?.Offset3 ?? 0);
                        }

                        int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor3.Split('/')[1]).SPC_GlueWeight ?? 0;

                        if (brandPapers.Exists(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]))
                        {
                            //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]).GuleWgt;

                            var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]);
                            var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                            brandOffset1 += (brandOffsetInfo?.Offset3 ?? 0);
                        }
                        mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);

                        int wgt = msWgt + lsWgt;
                        mapInfo.Weight = wgt;
                        mapList.Add(mapInfo);
                        index++;
                    }

                }

                if (isMapErr)
                {
                    logger.Info("用户勾选的糊机糊间隙使用部位和材质匹配对应不上，使用默认情况处理", module);
                    //如果用户勾选的使用部位和材质解析之后匹配不上， 则使用默认方式进行处理
                    index = 0;
                    driverList.Clear();
                    switch (paperOldList.Count)
                    {
                        case 3:
                            if (paperOldList[1] != "-")
                            {
                                driverList.Add(GluePositionEnum.Gu1);
                            }
                            break;
                        case 5:
                            if (paperOldList[0] == "-")
                            {
                                //4层板默认用GU2
                                if (paperOldList[1] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu2);
                                }
                            }
                            else
                            {
                                if (paperOldList[1] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu1);
                                }
                                if (paperOldList[3] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu2);
                                }
                            }

                            break;
                        case 7:
                            int cnt = paperOldList.Where(it => it != "-").Count();
                            if (cnt == 4)
                            {
                                if (paperOldList[1] != "-" && paperOldList[3] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu2);
                                }
                                else if (paperOldList[1] != "-" && paperOldList[5] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu3);
                                }
                                else if (paperOldList[3] != "-" && paperOldList[5] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu3);
                                }
                            }
                            else if (cnt == 6)
                            {
                                driverList.Add(GluePositionEnum.Gu2);
                                driverList.Add(GluePositionEnum.Gu3);
                            }
                            else
                            {
                                if (paperOldList[1] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu1);
                                }
                                if (paperOldList[3] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu2);
                                }
                                if (paperOldList[5] != "-")
                                {
                                    driverList.Add(GluePositionEnum.Gu3);
                                }
                            }

                            break;
                        default:
                            break;
                    }
                    mapList.Clear();
                    if (!string.IsNullOrEmpty(pCodeFloor1))
                    {
                        if (index < driverList.Count())
                        {
                            PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                            mapInfo.PaperPosition = PaperPositionEnum.Floor1;
                            mapInfo.PaperCode = pCodeFloor1;
                            mapInfo.Driver = driverList[index];
                            decimal brandOffset0 = 0;
                            decimal brandOffset1 = 0;
                            //计算出克重
                            int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor1.Split('/')[0]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]))
                            {
                                //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]).GuleWgt;
                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[0] && it.Brand == brandpCodeFloor1.Split('/')[0]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset0 += (brandOffsetInfo?.Offset1 ?? 0);

                            }

                            int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor1.Split('/')[1]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor1.Split('/')[1] && it.Brand == brandpCodeFloor1.Split('/')[1]))
                            {
                                //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[1] && it.Brand == brandpCodeFloor1.Split('/')[1]).GuleWgt;

                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor1.Split('/')[1] && it.Brand == brandpCodeFloor1.Split('/')[1]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset1 += (brandOffsetInfo?.Offset1 ?? 0);
                            }

                            mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);
                            int wgt = msWgt + lsWgt;
                            mapInfo.Weight = wgt;

                            mapList.Add(mapInfo);
                            index++;
                        }
                    }
                    if (!string.IsNullOrEmpty(pCodeFloor2))
                    {
                        if (index < driverList.Count())
                        {
                            PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                            mapInfo.PaperPosition = PaperPositionEnum.Floor2;
                            mapInfo.PaperCode = pCodeFloor2;
                            mapInfo.Driver = driverList[index];
                            decimal brandOffset0 = 0;
                            decimal brandOffset1 = 0;
                            //计算出克重
                            int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor2.Split('/')[0]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]))
                            {
                                //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]).GuleWgt;

                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[0] && it.Brand == brandpCodeFloor2.Split('/')[0]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset0 += (brandOffsetInfo?.Offset2 ?? 0);
                            }

                            int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor2.Split('/')[1]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]))
                            {
                                //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]).GuleWgt;

                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor2.Split('/')[1] && it.Brand == brandpCodeFloor2.Split('/')[1]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset1 += (brandOffsetInfo?.Offset2 ?? 0);
                            }
                            mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);
                            int wgt = msWgt + lsWgt;
                            mapInfo.Weight = wgt;
                            mapList.Add(mapInfo);
                            index++;
                        }

                    }
                    if (!string.IsNullOrEmpty(pCodeFloor3))
                    {
                        if (index < driverList.Count())
                        {
                            PaperPositionCodeDriverInfo mapInfo = new PaperPositionCodeDriverInfo();
                            mapInfo.PaperPosition = PaperPositionEnum.Floor3;
                            mapInfo.PaperCode = pCodeFloor3;
                            mapInfo.Driver = driverList[index];
                            decimal brandOffset0 = 0;
                            decimal brandOffset1 = 0;
                            //计算出克重
                            int msWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor3.Split('/')[0]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]))
                            {
                                //msWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]).GuleWgt;

                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[0] && it.Brand == brandpCodeFloor3.Split('/')[0]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset0 += (brandOffsetInfo?.Offset3 ?? 0);
                            }

                            int lsWgt = allPapers.FirstOrDefault(it => it.SPC_Code == pCodeFloor3.Split('/')[1]).SPC_GlueWeight ?? 0;

                            if (brandPapers.Exists(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]))
                            {
                                //lsWgt = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]).GuleWgt;

                                var brand = brandPapers.FirstOrDefault(it => it.Code == pCodeFloor3.Split('/')[1] && it.Brand == brandpCodeFloor3.Split('/')[1]);
                                var brandOffsetInfo = brandGlueInfos.Where(it => it.BrandID == brand.ID && it.Flute == info.Flute).FirstOrDefault();
                                brandOffset1 += (brandOffsetInfo?.Offset3 ?? 0);
                            }
                            mapInfo.BrandOffset = Math.Max(brandOffset0, brandOffset1);
                            int wgt = msWgt + lsWgt;
                            mapInfo.Weight = wgt;
                            mapList.Add(mapInfo);
                            index++;
                        }

                    }
                }
                decimal paperFloor1QdmCoef = 1; //纸张一层的qdm系数
                decimal paperFloor2QdmCoef = 1; //纸张二层的qdm系数
                decimal paperFloor3QdmCoef = 1; //纸张三层的qdm系数
                decimal paperFloor1FormQdmCoef = 1; //纸张一层的界面系数
                decimal paperFloor2FormQdmCoef = 1; //纸张二层的界面系数
                decimal paperFloor3FormQdmCoef = 1; //纸张三层的界面系数
                QdmCoefDFInfo dfInfo = new QdmCoefDFInfo();
                if (isChangeNow && !isFirst)
                {
                    QdmCoefDFInfo dfQdmInfo = QdmCtrl.GetQdmDFCoef(paper, info.Flute);
                    foreach (var minfo in mapList)
                    {
                        if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            dfInfo.Glue1 = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                            paperFloor1QdmCoef = dfQdmInfo.Glue1;
                            paperFloor1FormQdmCoef = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            dfInfo.Glue1 = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                            paperFloor1QdmCoef = dfQdmInfo.Glue1;
                            paperFloor1FormQdmCoef = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            dfInfo.Glue1 = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                            paperFloor1QdmCoef = dfQdmInfo.Glue1;
                            paperFloor1FormQdmCoef = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            dfInfo.Glue2 = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                            paperFloor2QdmCoef = dfQdmInfo.Glue2;
                            paperFloor2FormQdmCoef = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            dfInfo.Glue2 = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                            paperFloor2QdmCoef = dfQdmInfo.Glue2;
                            paperFloor2FormQdmCoef = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            dfInfo.Glue2 = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                            paperFloor2QdmCoef = dfQdmInfo.Glue2;
                            paperFloor2FormQdmCoef = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            dfInfo.Glue3 = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                            paperFloor3QdmCoef = dfQdmInfo.Glue3;
                            paperFloor3FormQdmCoef = formSetInfo.F_Glue_GU_1st_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            dfInfo.Glue3 = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                            paperFloor3QdmCoef = dfQdmInfo.Glue3;
                            paperFloor3FormQdmCoef = formSetInfo.F_Glue_GU_2nd_FormQdm_Factor;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            dfInfo.Glue3 = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                            paperFloor3QdmCoef = dfQdmInfo.Glue3;
                            paperFloor3FormQdmCoef = formSetInfo.F_Glue_GU_3rd_FormQdm_Factor;
                        }
                    }
                }
                else
                {
                    //正常换材，取QDM系数，然后更新formset表
                    dfInfo = QdmCtrl.GetQdmDFCoef(paper, info.Flute);
                    decimal coef1 = 1;
                    decimal coef2 = 1;
                    decimal coef3 = 1;
                    foreach (var minfo in mapList)
                    {
                        if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            coef1 = dfInfo.Glue1;
                            paperFloor1QdmCoef = dfInfo.Glue1;
                            paperFloor1FormQdmCoef = dfInfo.Glue1;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            coef2 = dfInfo.Glue1;
                            paperFloor1QdmCoef = dfInfo.Glue1;
                            paperFloor1FormQdmCoef = dfInfo.Glue1;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor1 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            coef3 = dfInfo.Glue1;
                            paperFloor1QdmCoef = dfInfo.Glue1;
                            paperFloor1FormQdmCoef = dfInfo.Glue1;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            coef1 = dfInfo.Glue2;
                            paperFloor2QdmCoef = dfInfo.Glue2;
                            paperFloor2FormQdmCoef = dfInfo.Glue2;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            coef2 = dfInfo.Glue2;
                            paperFloor2QdmCoef = dfInfo.Glue2;
                            paperFloor2FormQdmCoef = dfInfo.Glue2;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor2 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            coef3 = dfInfo.Glue2;
                            paperFloor2QdmCoef = dfInfo.Glue2;
                            paperFloor2FormQdmCoef = dfInfo.Glue2;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu1)
                        {
                            coef1 = dfInfo.Glue3;
                            paperFloor3QdmCoef = dfInfo.Glue3;
                            paperFloor3FormQdmCoef = dfInfo.Glue3;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu2)
                        {
                            coef2 = dfInfo.Glue3;
                            paperFloor3QdmCoef = dfInfo.Glue3;
                            paperFloor3FormQdmCoef = dfInfo.Glue3;
                        }
                        else if (minfo.PaperPosition == PaperPositionEnum.Floor3 && minfo.Driver == GluePositionEnum.Gu3)
                        {
                            coef3 = dfInfo.Glue3;
                            paperFloor3QdmCoef = dfInfo.Glue3;
                            paperFloor3FormQdmCoef = dfInfo.Glue3;
                        }
                    }
                    BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                    {
                        F_Glue_GU_1st_FormQdm_Factor = coef1,
                        F_Glue_GU_2nd_FormQdm_Factor = coef2,
                        F_Glue_GU_3rd_FormQdm_Factor = coef3,
                    }).Where(it => 1 == 1).ExecuteCommand();
                    //发送M108命令，让客户端界面重新取一下QDM系数
                    SendM108();
                }

                //抵换高还是高换低
                AutoDelayInfo autoDelayInfo = new AutoDelayInfo();
                if (lastWeight < curWeight)
                {
                    //获取延迟设置
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 1 && it.F_Position == "DF");
                }
                else
                {
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 2 && it.F_Position == "DF");
                }
                //拿到设定的糊机到横切的距离
                int preHQMeter = Convert.ToInt32(BLLFactory<DictDataInfoManager>.Instance.GetDictItems(DictTypesEnum.PreHQMeter.ToString())[0]);
                preHQMeter = -1 * preHQMeter;
                decimal length = preHQMeter;
                while (length < autoDelayInfo?.F_Glue && isChangeNow == false)
                {
                    try
                    {
                        if (token.IsCancellationRequested)
                        {
                            logger.Info("糊机糊间隙赋值任务取消,因为该期间内又收到一个新的糊机糊间隙赋值任务", module);
                            return;
                        }
                        //获取当前车速
                        var speedInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.DF_MachineSpeed.ToString());
                        if (speedInfo != null)
                        {
                            length += (speedInfo.VarValue.ToDecimal() / 60m * 0.1m);
                        }
                    }
                    catch (Exception)
                    {
                        throw;
                    }
                    finally
                    {
                        Thread.Sleep(100);
                    }
                }


                decimal gu1OffSet = 0, gu2OffSet = 0, gu3OffSet = 0;
                if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueGU1, out ActPostionOffSet actPostionOffSet1))
                {
                    gu1OffSet = actPostionOffSet1.OffSetValue;
                }
                if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueGU2, out ActPostionOffSet actPostionOffSet2))
                {
                    gu2OffSet = actPostionOffSet2.OffSetValue;
                }
                if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueGU3, out ActPostionOffSet actPostionOffSet3))
                {
                    gu3OffSet = actPostionOffSet3.OffSetValue;
                }

                //计算糊间隙配方
                Dictionary<string, decimal> dict = new Dictionary<string, decimal>();
                StringBuilder sb = new StringBuilder();
                foreach (var map in mapList)
                {
                    decimal realqdmCoef = 1;
                    decimal realformqdmCoef = 1;
     
                    if (map.Driver == GluePositionEnum.Gu1)
                    {
                        var glueSetInfo = glueSetInfos.FirstOrDefault(it => it.Position == map.PaperPosition);
                        int i = 0;//车速段标记
                        foreach (var scInfo in speedCoefGu1)
                        {
                            //计算8段车速对应的糊间隙值
                            //首先计算基准值，限制一个最大值
                            //得到的值再乘以车速系数，qdm系数，界面系数，得到的值，比较最小值
                            Dictionary<string, object> dic = new Dictionary<string, object>
                            {
                                { "MinGlueGap", glueSetInfo.MinGlue },
                                { "MaxGlueGap", glueSetInfo.MaxGlue },
                                { "MinGms", glueSetInfo.MinWeight },
                                { "MaxGms", glueSetInfo.MaxWeight },
                                { "CurrentGms", map.Weight },
                                { "FluteCoef", glueSetInfo.Coef },
                                { "SpeedCoef", 1 },
                                { "QDMCoef", 1 },
                                { "AdjustCoef", 1 }
                            };
                            decimal setValue = IPSCalMethod.CalGlueGap(dic);

                            //不再按照基础表限制最大值
                            //if (setValue > glueSetInfo.MaxGlue)
                            //    setValue = glueSetInfo.MaxGlue;

                            if (setValue > 60)
                                setValue = 60;

                            //再乘以各种系数
                            decimal qdmCoef = 1;
                            decimal formCoef = 1;



                            decimal speedCoef = scInfo.Coef;
                            //乘以车速系数之后的值才是基准值
                            decimal baseValue = Math.Round(setValue * speedCoef, 2);

                            if (map.PaperPosition == PaperPositionEnum.Floor1)
                            {
                                qdmCoef = dfInfo.Glue1;
                                realqdmCoef = paperFloor1QdmCoef;
                                realformqdmCoef = paperFloor1FormQdmCoef;
                            }
                            else if (map.PaperPosition == PaperPositionEnum.Floor2)
                            {
                                qdmCoef = dfInfo.Glue2;
                                realqdmCoef = paperFloor2QdmCoef;
                                realformqdmCoef = paperFloor2FormQdmCoef;
                            }
                            else if (map.PaperPosition == PaperPositionEnum.Floor3)
                            {
                                qdmCoef = dfInfo.Glue3;
                                realqdmCoef = paperFloor3QdmCoef;
                                realformqdmCoef = paperFloor3FormQdmCoef;
                            }


                            if (map.Driver == GluePositionEnum.Gu1)
                                formCoef = formSetInfo.F_Glue_GU_1st_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu2)
                                formCoef = formSetInfo.F_Glue_GU_2nd_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu3)
                                formCoef = formSetInfo.F_Glue_GU_3rd_Form_Factor;
                            setValue = setValue * qdmCoef * formCoef * speedCoef;
                            setValue = Math.Round(setValue, 2);
                            decimal UnlimitValue = setValue + map.BrandOffset;

                            setValue += gu1OffSet;
                            setValue += map.BrandOffset;

                            sb.AppendLine($"GU1糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={map.Weight};车速系数={speedCoef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={qdmCoef};界面系数={formCoef};计算结果：{setValue}");

                            if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                            {
                                sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                                setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                            }


                            switch (i)
                            {
                                case 0:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed0_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value0_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 1:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed1_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value1_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 2:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed2_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value2_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 3:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed3_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value3_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 4:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed4_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value4_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 5:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed5_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value5_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 6:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed6_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value6_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 7:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed7_L1.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value7_L1.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                default:
                                    break;
                            }
                            i++;

                            GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                            {
                                Speed = scInfo.Speed,
                                Value = setValue,
                                QdmCoef = qdmCoef,
                                FormCoef = formCoef,
                                MinValue = scInfo.MinValue ?? 0,
                                MaxValue = 999,
                                BaseValue = baseValue,
                                UnrestrictedSetValue = UnlimitValue,
                                BrandOffSetValue =map.BrandOffset,
                                OffSetValue = gu1OffSet,
                                RealQdmCoef = realqdmCoef,
                                FormQdmCoef = realformqdmCoef,
                            };

                            GlobalControl.tempGu1.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                        }
                        string msg = sb.ToString();
                        if (!string.IsNullOrEmpty(msg))
                        {
                            logger.Info(msg, module);
                        }
                        sb.Clear();

                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueGU1, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_GU_1st_Form_Factor, RealQdmCoef = realqdmCoef, FormQdmCoef = realformqdmCoef }, string.Join(".", paperOldList), info.Flute);

                    }
                    else if (map.Driver == GluePositionEnum.Gu2)
                    {
                        var glueSetInfo = glueSetInfos.FirstOrDefault(it => it.Position == map.PaperPosition);
                        int i = 0;//车速段标记
                        foreach (var scInfo in speedCoefGu2)
                        {
                            //计算8段车速对应的糊间隙值
                            Dictionary<string, object> dic = new Dictionary<string, object>
                            {
                                { "MinGlueGap", glueSetInfo.MinGlue },
                                { "MaxGlueGap", glueSetInfo.MaxGlue },
                                { "MinGms", glueSetInfo.MinWeight },
                                { "MaxGms", glueSetInfo.MaxWeight },
                                { "CurrentGms", map.Weight },
                                { "FluteCoef", glueSetInfo.Coef },
                                { "SpeedCoef", 1 },
                                { "QDMCoef", 1 },
                                { "AdjustCoef", 1 }
                            };

                            decimal setValue = IPSCalMethod.CalGlueGap(dic);

                            if (setValue > 60)
                                setValue = 60;

                            //再乘以各种系数
                            decimal qdmCoef = 1;
                            decimal formCoef = 1;
                            decimal speedCoef = scInfo.Coef;
                            decimal baseValue = Math.Round(setValue * speedCoef, 2);

                            if (map.PaperPosition == PaperPositionEnum.Floor1)
                            {
                                qdmCoef = dfInfo.Glue1;
                                realqdmCoef = paperFloor1QdmCoef;
                                realformqdmCoef = paperFloor1FormQdmCoef;
                            }
                            else if (map.PaperPosition == PaperPositionEnum.Floor2)
                            {
                                qdmCoef = dfInfo.Glue2;
                                realqdmCoef = paperFloor2QdmCoef;
                                realformqdmCoef = paperFloor2FormQdmCoef;
                            }
                            else if (map.PaperPosition == PaperPositionEnum.Floor3)
                            {
                                qdmCoef = dfInfo.Glue3;
                                realqdmCoef = paperFloor3QdmCoef;
                                realformqdmCoef = paperFloor3FormQdmCoef;
                            }


                            if (map.Driver == GluePositionEnum.Gu1)
                                formCoef = formSetInfo.F_Glue_GU_1st_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu2)
                                formCoef = formSetInfo.F_Glue_GU_2nd_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu3)
                                formCoef = formSetInfo.F_Glue_GU_3rd_Form_Factor;
                            setValue = setValue * qdmCoef * formCoef * speedCoef;
                            setValue = Math.Round(setValue, 2);
                            decimal UnlimitValue = setValue + map.BrandOffset;

                            setValue += gu2OffSet;
                            setValue += map.BrandOffset;

                            sb.AppendLine($"GU2糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={map.Weight};车速系数={speedCoef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={qdmCoef};界面系数={formCoef};计算结果：{setValue}");

                            if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                            {
                                sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                                setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                            }

                            switch (i)
                            {
                                case 0:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed0_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value0_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 1:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed1_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value1_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 2:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed2_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value2_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 3:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed3_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value3_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 4:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed4_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value4_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 5:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed5_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value5_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 6:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed6_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value6_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 7:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed7_L2.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value7_L2.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                default:
                                    break;
                            }
                            i++;

                            GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                            {
                                Speed = scInfo.Speed,
                                Value = setValue,
                                QdmCoef = qdmCoef,
                                FormCoef = formCoef,
                                MinValue = scInfo.MinValue ?? 0,
                                MaxValue = 999,
                                BaseValue = baseValue,
                                UnrestrictedSetValue = UnlimitValue,
                                BrandOffSetValue = map.BrandOffset,
                                OffSetValue = gu2OffSet,
                                RealQdmCoef = realqdmCoef,
                                FormQdmCoef = realformqdmCoef,
                            };

                            GlobalControl.tempGu2.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                        }
                        string msg = sb.ToString();
                        if (!string.IsNullOrEmpty(msg))
                        {
                            logger.Info(msg, module);
                        }
                        sb.Clear();

                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueGU2, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_GU_2nd_Form_Factor, RealQdmCoef = realqdmCoef, FormQdmCoef = realformqdmCoef }, string.Join(".", paperOldList), info.Flute);

                    }
                    else if (map.Driver == GluePositionEnum.Gu3)
                    {
                        var glueSetInfo = glueSetInfos.FirstOrDefault(it => it.Position == map.PaperPosition);
                        int i = 0;//车速段标记
                        foreach (var scInfo in speedCoefGu3)
                        {
                            //计算8段车速对应的糊间隙值
                            Dictionary<string, object> dic = new Dictionary<string, object>
                            {
                                { "MinGlueGap", glueSetInfo.MinGlue },
                                { "MaxGlueGap", glueSetInfo.MaxGlue },
                                { "MinGms", glueSetInfo.MinWeight },
                                { "MaxGms", glueSetInfo.MaxWeight },
                                { "CurrentGms", map.Weight },
                                { "FluteCoef", glueSetInfo.Coef },
                                { "SpeedCoef", 1 },
                                { "QDMCoef", 1 },
                                { "AdjustCoef", 1 }
                            };

                            decimal setValue = IPSCalMethod.CalGlueGap(dic);//得到基础值

                            if (setValue > 60)
                                setValue = 60;

                            //再乘以各种系数
                            decimal qdmCoef = 1;
                            decimal formCoef = 1;
                            decimal speedCoef = scInfo.Coef;

                            decimal baseValue = Math.Round(setValue * speedCoef, 2);

                            if (map.PaperPosition == PaperPositionEnum.Floor1)
                            {
                                qdmCoef = dfInfo.Glue1;
                                realqdmCoef = paperFloor1QdmCoef;
                                realformqdmCoef = paperFloor1FormQdmCoef;
                            }
                            else if (map.PaperPosition == PaperPositionEnum.Floor2)
                            {
                                qdmCoef = dfInfo.Glue2;
                                realqdmCoef = paperFloor2QdmCoef;
                                realformqdmCoef = paperFloor2FormQdmCoef;
                            }

                            else if (map.PaperPosition == PaperPositionEnum.Floor3)
                            {
                                qdmCoef = dfInfo.Glue3;
                                realqdmCoef = paperFloor3QdmCoef;
                                realformqdmCoef = paperFloor3FormQdmCoef;
                            }


                            if (map.Driver == GluePositionEnum.Gu1)
                                formCoef = formSetInfo.F_Glue_GU_1st_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu2)
                                formCoef = formSetInfo.F_Glue_GU_2nd_Form_Factor;
                            else if (map.Driver == GluePositionEnum.Gu3)
                                formCoef = formSetInfo.F_Glue_GU_3rd_Form_Factor;

                            setValue = setValue * qdmCoef * formCoef * speedCoef;
                            setValue = Math.Round(setValue, 2);
                            decimal UnlimitValue = setValue + map.BrandOffset;

                            setValue += gu3OffSet;
                            setValue += map.BrandOffset;

                            sb.AppendLine($"GU3糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={map.Weight};车速系数={speedCoef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={qdmCoef};界面系数={formCoef};计算结果：{setValue}");

                            if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                            {
                                sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                                setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                            }


                            switch (i)
                            {
                                case 0:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed0_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value0_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 1:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed1_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value1_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 2:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed2_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value2_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 3:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed3_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value3_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 4:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed4_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value4_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 5:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed5_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value5_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 6:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed6_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value6_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                case 7:
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Speed7_L3.ToString(), scInfo.Speed);
                                    dict.Add(PointVarEnum.DF_GUGap_Curve_Value7_L3.ToString(), Math.Round(setValue / 100m, 2));
                                    break;
                                default:
                                    break;
                            }
                            i++;



                            GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                            {
                                Speed = scInfo.Speed,
                                Value = setValue,
                                QdmCoef = qdmCoef,
                                FormCoef = formCoef,
                                MinValue = scInfo.MinValue ?? 0,
                                MaxValue = 999,
                                BaseValue = baseValue,
                                UnrestrictedSetValue = UnlimitValue,
                                OffSetValue = gu3OffSet,
                                BrandOffSetValue = map.BrandOffset,
                                RealQdmCoef = realqdmCoef,
                                FormQdmCoef = realformqdmCoef,
                            };


                            GlobalControl.tempGu3.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                        }
                        string msg = sb.ToString();
                        if (!string.IsNullOrEmpty(msg))
                        {
                            logger.Info(msg, module);
                        }
                        sb.Clear();

                        GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueGU3, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_GU_3rd_Form_Factor, RealQdmCoef = realqdmCoef, FormQdmCoef = realformqdmCoef }, string.Join(".", paperOldList), info.Flute);

                    }
                }



                sb = null;

                if (token.IsCancellationRequested)
                {
                    logger.Info("即将往设备写值，但是因为该期间内又收到一个新的糊机糊间隙赋值任务，糊机糊间隙赋值任务取消", module);
                    return;
                }
                lock (_lockGu)
                {
                    foreach (string key in dict.Keys)
                    {
                        if (token.IsCancellationRequested)
                        {
                            logger.Info("即将往设备写值，但是因为该期间内又收到一个新的糊机糊间隙赋值任务，糊机糊间隙赋值任务取消", module);
                            return;
                        }
                        if (key.Contains("L1"))
                        {
                            if (!formSetInfo.F_Glue_GU_1st_Form_IsOpen)
                            {
                                continue;
                            }
                            else
                            {
                                comm.WriteVar(PointVarEnum.DF_GUGap_Base_Value_L1.ToString(), 0.6);
                                comm.WriteVar(PointVarEnum.DF_GUGap_Offset_L1.ToString(), 0);
                            }
                        }
                        if (key.Contains("L2"))
                        {
                            if (!formSetInfo.F_Glue_GU_2nd_Form_IsOpen)
                            {
                                continue;
                            }
                            else
                            {
                                comm.WriteVar(PointVarEnum.DF_GUGap_Base_Value_L2.ToString(), 0.6);
                                comm.WriteVar(PointVarEnum.DF_GUGap_Offset_L2.ToString(), 0);
                            }
                        }
                        if (key.Contains("L3"))
                        {
                            if (!formSetInfo.F_Glue_GU_3rd_Form_IsOpen)
                            {
                                continue;
                            }
                            else
                            {
                                comm.WriteVar(PointVarEnum.DF_GUGap_Base_Value_L3.ToString(), 0.6);
                                comm.WriteVar(PointVarEnum.DF_GUGap_Offset_L3.ToString(), 0);
                            }
                        }
                        comm.WriteVar(key, dict[key]);
                    }

                    logger.Info($"SetGlueGu--糊机糊间隙设备写值完成,材质={info.Code},楞型={info.Flute}", module);
                }

            }
            catch (OperationCanceledException)
            {
                logger.Warn($"SetGlueGu--糊机糊间隙任务取消,材质={info.Code},楞型={info.Flute}", module);
            }
            catch (Exception ex)
            {
                logger.Error($"SetGlueGu--糊机糊间隙赋值异常：{ex}", module);
            }
        }

        /// <summary>
        /// 处理单面机1糊间隙换材消息
        /// </summary>
        /// <param name="info"></param>
        private void HandleSF1GlueMsg(PublishInfo info)
        {
            logger.Info($"进入HandleSF1GlueMsg，本次为正常换材，材质={info.Code},楞型={info.Flute}", module);
            cts_sf1.Cancel();
            cts_sf1 = new CancellationTokenSource();
            CancellationToken token = cts_sf1.Token;
            Task.Run(() => { SetGlueSF1(info, token); }, token);
        }

        /// <summary>
        /// 给SF1糊间隙赋值
        /// </summary>
        /// <param name="info"></param>
        private void SetGlueSF1(PublishInfo info, CancellationToken token, bool isChangeNow = false, bool isFirst = false)
        {
            try
            {
                //拿到当前的MS材质和LS材质，上一次的MS材质和上一次的LS材质，当前的单瓦楞型
                string flute = info.Flute;
                string codeMS = info.Code.Split('/')[0];
                string codeLS = info.Code.Split('/')[1];
                string lastCodeMS = info.LastCode.Split('/')[0];
                string lastCodeLS = info.LastCode.Split('/')[1];
                logger.Info($"进入SetGlueSF1准备点位赋值，芯纸材质={codeMS}，里纸材质={codeLS},楞型={flute}", module);
                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
                QdmCoefSFInfo sfInfo = new QdmCoefSFInfo();

                decimal realQdmCoef = 1;
                decimal formQdmCoef = 1;
                if (isChangeNow && !isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = formSetInfo.F_Glue_SF1_FormQdm_Factor;

                    //如果是立刻更新，则说明是客户端修改系数，直接取formset表中的系数即可
                    sfInfo.Glue = formSetInfo.F_Glue_SF1_FormQdm_Factor;
                }
                else if (isChangeNow && isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = qdmCoefSFInfo.Glue;
                    sfInfo.Glue = qdmCoefSFInfo.Glue;
                    BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                    {
                        F_Glue_SF1_FormQdm_Factor = sfInfo.Glue
                    }).Where(it => 1 == 1).ExecuteCommand();
                    //发送M108命令，让客户端界面重新取一下QDM系数
                    SendM108();

                }
                else
                {
                    if (codeMS == lastCodeMS && codeLS == lastCodeLS)
                    {
                        sfInfo.Glue = formSetInfo.F_Glue_SF1_FormQdm_Factor;
                    }
                    else
                    {
                        //正常换材，取QDM系数，然后更新formset表
                        sfInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                        realQdmCoef = sfInfo.Glue;
                        formQdmCoef = sfInfo.Glue;

                        BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                        {
                            F_Glue_SF1_FormQdm_Factor = sfInfo.Glue
                        }).Where(it => 1 == 1).ExecuteCommand();
                        //发送M108命令，让客户端界面重新取一下QDM系数
                        SendM108();
                    }
                }

                //上次使用的材质的克重
                var lastCodeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeMS);
                var lastCodeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeLS);
                //本次使用的材质的克重
                var codeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeMS);
                var codeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeLS);
                //根据延迟设置那边的设置情况，按照高换低还是低换高，来确定是延迟赋值还是提前赋值
                int lastWeight = (lastCodeMsInfo?.SPC_Weight ?? 0) + (lastCodeLsInfo?.SPC_Weight ?? 0);
                int curWeight = (codeMsInfo?.SPC_Weight ?? 0) + (codeLsInfo?.SPC_Weight ?? 0);
                //抵换高还是高换低
                AutoDelayInfo autoDelayInfo = new AutoDelayInfo();
                if (lastWeight < curWeight)
                {
                    //获取延迟设置
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 1 && it.F_Position == "MS1");
                }
                else
                {
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 2 && it.F_Position == "MS1");
                }
                decimal meter = 0;//定义初始已走米数用于延迟赋值
                while (meter <= autoDelayInfo?.F_Glue && isChangeNow == false)
                {
                    try
                    {
                        if (token.IsCancellationRequested)
                        {
                            logger.Info("SetGlueSF1--产生了一个新的任务，该任务终止", module);
                            return;
                        }
                        var speedInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF1_MachineSpeed.ToString());
                        if (speedInfo != null)
                        {
                            meter += speedInfo.VarValue.ToDecimal() / 60m * 0.1m;
                        }
                    }
                    catch (Exception)
                    {
                        throw;
                    }
                    finally
                    {
                        Thread.Sleep(100);
                    }
                }
                //开始进行赋值动作,计算单面机糊间隙
                //车速系数
                var speedCoefInfos = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.SF1).OrderBy(it => it.Speed).ToList();
                //基础设置
                var glueSetInfo = BLLFactory<GlueSFSetInfoManager>.Instance.GetFirst(it => it.FluteSF == flute);
                //拿到当前设置的SF糊间隙qdm系数
                decimal glueQdmCoef = sfInfo.Glue;

                decimal glueFormCoef = formSetInfo.F_Glue_SF1_Form_Factor;
                Dictionary<string, decimal> dict = new Dictionary<string, decimal>();
                int index = 0;

                curWeight = (codeMsInfo?.SPC_GlueWeight ?? 0) + (codeLsInfo?.SPC_GlueWeight ?? 0);//用糊间隙克重算当前克重

                #region 计算供应商品牌对应的克重
                string brandMS = info.BrandMS1;
                string brandLS = info.BrandLS1;
                decimal brandOffsetMS = 0;
                decimal brandOffsetLS = 0;
                if (!string.IsNullOrEmpty(brandMS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandMS && paper.SPC_Code == codeMS).First();

                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetMS += (glueSFBrand?.Offset ?? 0);
                    }

                }

                if (!string.IsNullOrEmpty(brandLS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandLS && paper.SPC_Code == codeLS).First();
                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetLS += (glueSFBrand?.Offset ?? 0);
                    }
                }
                #endregion

                StringBuilder sb = new StringBuilder();

                foreach (var scInfo in speedCoefInfos)
                {
                    //计算8段车速对应的糊间隙值
                    Dictionary<string, object> dic = new Dictionary<string, object>
                    {
                        { "MinGlueGap", glueSetInfo.MinGlue },
                        { "MaxGlueGap", glueSetInfo.MaxGlue },
                        { "MinGms", glueSetInfo.MinWeight },
                        { "MaxGms", glueSetInfo.MaxWeight },
                        { "CurrentGms", curWeight },
                        { "FluteCoef", glueSetInfo.Coef },
                        { "QDMCoef", 1 },
                        { "AdjustCoef", 1 },
                        { "SpeedCoef", 1 }
                    };

                    decimal setValue = IPSCalMethod.CalGlueGap(dic);//基础值

                    decimal baseValue = Math.Round(setValue * scInfo.Coef, 2);
                    setValue = setValue * scInfo.Coef * glueQdmCoef * glueFormCoef;
                    setValue = Math.Round(setValue, 2);

                    decimal offSet = 0;
                    if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueSF1, out ActPostionOffSet actPostionOffSet))
                    {
                        offSet = actPostionOffSet.OffSetValue;
                    }
                    decimal brandOffset = Math.Max(brandOffsetMS, brandOffsetLS);
                    //offSet += brandOffset;//加上品牌补偿值
                    decimal unrestrictedSetValue = setValue + brandOffset;

                    setValue += offSet;
                    setValue += brandOffset;

                    sb.AppendLine($"SF1糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={curWeight};车速系数={scInfo.Coef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={glueQdmCoef};界面系数={glueFormCoef};弯翘偏移量={offSet};计算结果：{setValue}");

                    if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                    {
                        sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                        setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                    }


                    switch (index)
                    {
                        case 0:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed0.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value0.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 1:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed1.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value1.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 2:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed2.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value2.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 3:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed3.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value3.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 4:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed4.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value4.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 5:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed5.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value5.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 6:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed6.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value6.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 7:
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Speed7.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF1_GUGap_Curve_Value7.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        default:
                            break;
                    }
                    index++;
                    GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                    {
                        Speed = scInfo.Speed,
                        Value = setValue,
                        QdmCoef = glueQdmCoef,
                        FormCoef = glueFormCoef,
                        MinValue = scInfo.MinValue ?? 10,
                        MaxValue = 999,
                        BaseValue = baseValue,
                        OffSetValue = offSet,
                        BrandOffSetValue= brandOffset,
                        UnrestrictedSetValue = unrestrictedSetValue,
                        FormQdmCoef = formQdmCoef,
                        RealQdmCoef = realQdmCoef,
                    };

                    GlobalControl.tempSf1.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                }

                string msg = sb.ToString();
                if (!string.IsNullOrEmpty(msg))
                {
                    logger.Info(msg, module);
                }
                sb.Clear();
                sb = null;

                GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueSF1, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_SF1_Form_Factor, RealQdmCoef = realQdmCoef, FormQdmCoef = formQdmCoef }, codeMS + "." + codeLS, info.Flute);


                //机器写值之前再次判断是否要终止线程
                if (token.IsCancellationRequested)
                {
                    logger.Info("SetGlueSF1--产生了一个新的任务，该任务终止", module);
                    return;
                }
                if (!formSetInfo.F_Glue_SF1_Form_IsOpen)
                {
                    logger.Info("SetGlueSF1--SF1糊间隙没有启用，往设备写值动作终止", module);
                    return;
                }

                foreach (var key in dict.Keys)
                {
                    if (token.IsCancellationRequested)
                    {
                        logger.Info("SF1糊间隙 即将往设备写值，但是因为该期间内又收到一个新的产生了一个新的任务，该任务终止", module);
                        return;
                    }
                    comm.WriteVar(key, dict[key]);
                }
                //写停机糊间隙
                comm.WriteVar(PointVarEnum.SF1_GUGap_Base_Value.ToString(), 0.6);
                comm.WriteVar(PointVarEnum.SF1_GUGap_Offset.ToString(), 0);
                logger.Info($"SetGlueSF1--SF1糊间隙往设备写值动作完成,材质={info.Code},楞型={info.Flute}", module);
            }
            catch (OperationCanceledException)
            {
                logger.Warn($"SetGlueSF1--产生了一个新的任务，该任务终止.材质={info.Code},楞型={info.Flute}", module);
            }
            catch (Exception ex)
            {
                logger.Error($"SetGlueSF1--执行异常：{ex}", module);
            }
        }

        /// <summary>
        /// 处理单面机2糊间隙换材消息
        /// </summary>
        /// <param name="info"></param>
        private void HandleSF2GlueMsg(PublishInfo info)
        {
            logger.Info($"进入 HandleSF2GlueMsg，本次为正常换材，材质={info.Code},楞型={info.Flute}", module);
            cts_sf2.Cancel();
            cts_sf2 = new CancellationTokenSource();
            CancellationToken token = cts_sf2.Token;
            Task.Run(() => { SetGlueSF2(info, token); }, token);
        }

        /// <summary>
        /// 给SF2糊间隙赋值
        /// </summary>
        /// <param name="info"></param>
        private void SetGlueSF2(PublishInfo info, CancellationToken token, bool isChangeNow = false, bool isFirst = false)
        {
            try
            {
                //拿到当前的MS材质和LS材质，上一次的MS材质和上一次的LS材质，当前的单瓦楞型
                string flute = info.Flute;
                string codeMS = info.Code.Split('/')[0];
                string codeLS = info.Code.Split('/')[1];
                string lastCodeMS = info.LastCode.Split('/')[0];
                string lastCodeLS = info.LastCode.Split('/')[1];
                logger.Info($"进入SetGlueSF2准备点位赋值，芯纸材质={codeMS}，里纸材质={codeLS},楞型={flute}", module);

                decimal realQdmCoef = 1;
                decimal formQdmCoef = 1;
                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
                QdmCoefSFInfo sfInfo = new QdmCoefSFInfo();
                if (isChangeNow && !isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = formSetInfo.F_Glue_SF2_FormQdm_Factor;
                    //如果是立刻更新，则说明是客户端修改系数，直接取formset表中的系数即可
                    sfInfo.Glue = formSetInfo.F_Glue_SF2_FormQdm_Factor;
                }
                else if (isChangeNow && isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = qdmCoefSFInfo.Glue;

                    sfInfo.Glue = qdmCoefSFInfo.Glue;
                    BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                    {
                        F_Glue_SF2_FormQdm_Factor = sfInfo.Glue
                    }).Where(it => 1 == 1).ExecuteCommand();
                    //发送M108命令，让客户端界面重新取一下QDM系数
                    SendM108();

                }
                else
                {
                    if (codeMS == lastCodeMS && codeLS == lastCodeLS)
                    {
                        sfInfo.Glue = formSetInfo.F_Glue_SF2_FormQdm_Factor;
                    }
                    else
                    {
                        //正常换材，取QDM系数，然后更新formset表
                        sfInfo = sfInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                        realQdmCoef = sfInfo.Glue;
                        formQdmCoef = sfInfo.Glue;
                        BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                        {
                            F_Glue_SF2_FormQdm_Factor = sfInfo.Glue
                        }).Where(it => 1 == 1).ExecuteCommand();
                        //发送M108命令，让客户端界面重新取一下QDM系数
                        SendM108();
                    }
                }

                //上次使用的材质的克重
                var lastCodeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeMS);
                var lastCodeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeLS);
                //本次使用的材质的克重
                var codeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeMS);
                var codeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeLS);
                //根据延迟设置那边的设置情况，按照高换低还是低换高，来确定是延迟赋值还是提前赋值
                int lastWeight = (lastCodeMsInfo?.SPC_Weight ?? 0) + (lastCodeLsInfo?.SPC_Weight ?? 0);
                int curWeight = (codeMsInfo?.SPC_Weight ?? 0) + (codeLsInfo?.SPC_Weight ?? 0);
                //抵换高还是高换低
                AutoDelayInfo autoDelayInfo = new AutoDelayInfo();
                if (lastWeight < curWeight)
                {
                    //获取延迟设置
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 1 && it.F_Position == "MS2");
                }
                else
                {
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 2 && it.F_Position == "MS2");
                }
                decimal meter = 0;//定义初始已走米数用于延迟赋值
                while (meter <= autoDelayInfo?.F_Glue && isChangeNow == false)
                {
                    try
                    {
                        if (token.IsCancellationRequested)
                        {
                            logger.Info("SetGlueSF2--产生了一个新的任务，该任务终止", module);
                            return;
                        }
                        var speedInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF2_MachineSpeed.ToString());
                        if (speedInfo != null)
                        {
                            meter += speedInfo.VarValue.ToDecimal() / 60m * 0.1m;
                        }
                    }
                    catch (Exception)
                    {
                        throw;
                    }
                    finally
                    {
                        Thread.Sleep(100);
                    }
                }
                //开始进行赋值动作,计算单面机糊间隙
                //车速系数
                var speedCoefInfos = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.SF2).OrderBy(it => it.Speed).ToList();
                //基础设置
                var glueSetInfo = BLLFactory<GlueSFSetInfoManager>.Instance.GetFirst(it => it.FluteSF == flute);
                //拿到当前设置的SF糊间隙qdm系数
                decimal glueQdmCoef = sfInfo.Glue;

                decimal glueFormCoef = formSetInfo.F_Glue_SF2_Form_Factor;
                Dictionary<string, decimal> dict = new Dictionary<string, decimal>();
                int index = 0;
                curWeight = (codeMsInfo?.SPC_GlueWeight ?? 0) + (codeLsInfo?.SPC_GlueWeight ?? 0);

                #region 计算供应商品牌对应的克重
                string brandMS = info.BrandMS2;
                string brandLS = info.BrandLS2;
                decimal brandOffsetMS = 0;
                decimal brandOffsetLS = 0;
                if (!string.IsNullOrEmpty(brandMS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandMS && paper.SPC_Code == codeMS).First();
                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetMS += (glueSFBrand?.Offset ?? 0);
                    }

                }


                if (!string.IsNullOrEmpty(brandLS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandLS && paper.SPC_Code == codeLS).First();
                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetLS += (glueSFBrand?.Offset ?? 0);
                    }
                }
                #endregion

                StringBuilder sb = new StringBuilder();

                foreach (var scInfo in speedCoefInfos)
                {
                    //计算8段车速对应的糊间隙值
                    Dictionary<string, object> dic = new Dictionary<string, object>
                    {
                        { "MinGlueGap", glueSetInfo.MinGlue },
                        { "MaxGlueGap", glueSetInfo.MaxGlue },
                        { "MinGms", glueSetInfo.MinWeight },
                        { "MaxGms", glueSetInfo.MaxWeight },
                        { "CurrentGms", curWeight },
                        { "FluteCoef", glueSetInfo.Coef },
                        { "QDMCoef", 1 },
                        { "AdjustCoef", 1 },
                        { "SpeedCoef", 1 }
                    };

                    decimal setValue = IPSCalMethod.CalGlueGap(dic);//基础值

                    if (setValue > 60)
                        setValue = 60;

                    decimal baseValue = Math.Round(setValue * scInfo.Coef, 2);


                    setValue = setValue * glueQdmCoef * glueFormCoef * scInfo.Coef;
                    setValue = Math.Round(setValue, 2);

                    decimal offSet = 0;
                    if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueSF2, out ActPostionOffSet actPostionOffSet))
                    {
                        offSet = actPostionOffSet.OffSetValue;
                    }
                    decimal brandOffset = Math.Max(brandOffsetMS, brandOffsetLS);
                    decimal unrestrictedSetValue = setValue + brandOffset;

                    setValue += offSet;
                    setValue += brandOffset;

                    sb.AppendLine($"SF2糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={curWeight};车速系数={scInfo.Coef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={glueQdmCoef};界面系数={glueFormCoef};弯翘偏移量={offSet};计算结果：{setValue}");

                    if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                    {
                        sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                        setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                    }

                    switch (index)
                    {
                        case 0:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed0.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value0.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 1:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed1.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value1.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 2:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed2.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value2.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 3:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed3.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value3.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 4:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed4.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value4.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 5:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed5.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value5.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 6:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed6.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value6.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 7:
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Speed7.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF2_GUGap_Curve_Value7.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        default:
                            break;
                    }
                    index++;
                    GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                    {
                        Speed = scInfo.Speed,
                        Value = setValue,
                        QdmCoef = glueQdmCoef,
                        FormCoef = glueFormCoef,
                        MinValue = scInfo.MinValue ?? 10,
                        MaxValue = 999,
                        BaseValue = baseValue,
                        UnrestrictedSetValue = unrestrictedSetValue,
                        OffSetValue = offSet,
                        BrandOffSetValue = brandOffset,
                        FormQdmCoef = formQdmCoef,
                        RealQdmCoef = realQdmCoef,
                    };

                    GlobalControl.tempSf2.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                }

                string msg = sb.ToString();
                if (!string.IsNullOrEmpty(msg))
                    logger.Info(msg, module);
                sb.Clear();
                sb = null;

                GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueSF2, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_SF2_Form_Factor, RealQdmCoef = realQdmCoef, FormQdmCoef = formQdmCoef }, codeMS + "." + codeLS, info.Flute);


                //机器写值之前再次判断是否要终止线程
                if (token.IsCancellationRequested)
                {
                    logger.Info("SetGlueSF2--产生了一个新的任务，该任务终止", module);
                    return;
                }
                if (!formSetInfo.F_Glue_SF2_Form_IsOpen)
                {
                    logger.Info("SetGlueSF2--SF2糊间隙没有启用，往设备写值动作终止", module);
                    return;
                }

                foreach (var key in dict.Keys)
                {
                    if (token.IsCancellationRequested)
                    {
                        logger.Info("SF2糊间隙 即将往设备写值，但是因为该期间内又收到一个新的产生了一个新的任务，该任务终止", module);
                        return;
                    }
                    comm.WriteVar(key, dict[key]);
                }
                //写停机糊间隙
                comm.WriteVar(PointVarEnum.SF2_GUGap_Base_Value.ToString(), 0.6);
                comm.WriteVar(PointVarEnum.SF2_GUGap_Offset.ToString(), 0);
                logger.Info($"SetGlueSF2--SF2糊间隙往设备写值动作完成,材质={info.Code},楞型={info.Flute}", module);
            }
            catch (OperationCanceledException)
            {
                logger.Warn($"SetGlueSF2--产生了一个新的任务，该任务终止.材质={info.Code},楞型={info.Flute}", module);
            }
            catch (Exception ex)
            {
                logger.Error($"SetGlueSF2--执行异常：{ex}", module);
            }
        }

        /// <summary>
        /// 处理单面机3糊间隙换材消息
        /// </summary>
        /// <param name="info"></param>
        private void HandleSF3GlueMsg(PublishInfo info)
        {
            logger.Info($"进入HandleSF3GlueMsg，本次为正常换材，材质={info.Code},楞型={info.Flute}", module);
            cts_sf3.Cancel();
            cts_sf3 = new CancellationTokenSource();
            CancellationToken token = cts_sf3.Token;
            Task.Run(() => { SetGlueSF3(info, token); }, token);
        }

        /// <summary>
        /// 给SF3糊间隙赋值
        /// </summary>
        /// <param name="info"></param>
        private void SetGlueSF3(PublishInfo info, CancellationToken token, bool isChangeNow = false, bool isFirst = false)
        {
            try
            {
                //拿到当前的MS材质和LS材质，上一次的MS材质和上一次的LS材质，当前的单瓦楞型
                string flute = info.Flute;
                string codeMS = info.Code.Split('/')[0];
                string codeLS = info.Code.Split('/')[1];
                string lastCodeMS = info.LastCode.Split('/')[0];
                string lastCodeLS = info.LastCode.Split('/')[1];
                logger.Info($"进入SetGlueSF3准备点位赋值，芯纸材质={codeMS}，里纸材质={codeLS},楞型={flute}", module);
                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
                QdmCoefSFInfo sfInfo = new QdmCoefSFInfo();
                decimal realQdmCoef = 1;
                decimal formQdmCoef = 1;
                if (isChangeNow && !isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = formSetInfo.F_Glue_SF3_FormQdm_Factor;

                    //如果是立刻更新，则说明是客户端修改系数，直接取formset表中的系数即可
                    sfInfo.Glue = formSetInfo.F_Glue_SF3_FormQdm_Factor;
                }
                else if (isChangeNow && isFirst)
                {
                    QdmCoefSFInfo qdmCoefSFInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                    realQdmCoef = qdmCoefSFInfo.Glue;
                    formQdmCoef = qdmCoefSFInfo.Glue;
                    sfInfo.Glue = qdmCoefSFInfo.Glue;
                    BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                    {
                        F_Glue_SF3_FormQdm_Factor = sfInfo.Glue
                    }).Where(it => 1 == 1).ExecuteCommand();
                    //发送M108命令，让客户端界面重新取一下QDM系数
                    SendM108();
                }
                else
                {
                    if (codeMS == lastCodeMS && codeLS == lastCodeLS)
                    {
                        sfInfo.Glue = formSetInfo.F_Glue_SF3_FormQdm_Factor;
                    }
                    else
                    {
                        //正常换材，取QDM系数，然后更新formset表
                        sfInfo = QdmCtrl.GetQdmCoefSFInfo(codeMS, codeLS, flute);
                        realQdmCoef = sfInfo.Glue;
                        formQdmCoef = sfInfo.Glue;
                        BLLFactory<FormSetQdmFactorInfoManager>.Instance.AsUpdateable().SetColumns(it => new FormSetFactorInfo
                        {
                            F_Glue_SF3_FormQdm_Factor = sfInfo.Glue
                        }).Where(it => 1 == 1).ExecuteCommand();
                        //发送M108命令，让客户端界面重新取一下QDM系数
                        SendM108();
                    }
                }

                //上次使用的材质的克重
                var lastCodeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeMS);
                var lastCodeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == lastCodeLS);
                //本次使用的材质的克重
                var codeMsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeMS);
                var codeLsInfo = BLLFactory<PaperCodeInfoManager>.Instance.GetFirst(it => it.SPC_Code == codeLS);
                //根据延迟设置那边的设置情况，按照高换低还是低换高，来确定是延迟赋值还是提前赋值
                int lastWeight = (lastCodeMsInfo?.SPC_Weight ?? 0) + (lastCodeLsInfo?.SPC_Weight ?? 0);
                int curWeight = (codeMsInfo?.SPC_Weight ?? 0) + (codeLsInfo?.SPC_Weight ?? 0);
                //抵换高还是高换低
                AutoDelayInfo autoDelayInfo = new AutoDelayInfo();
                if (lastWeight < curWeight)
                {
                    //获取延迟设置
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 1 && it.F_Position == "MS3");
                }
                else
                {
                    autoDelayInfo = BLLFactory<AutoDelayInfoManager>.Instance.GetFirst(it => it.F_Type == 2 && it.F_Position == "MS3");
                }
                decimal meter = 0;//定义初始已走米数用于延迟赋值
                while (meter <= autoDelayInfo?.F_Glue && isChangeNow == false)
                {
                    try
                    {
                        if (token.IsCancellationRequested)
                        {
                            logger.Info("SetGlueSF3--产生了一个新的任务，该任务终止", module);
                            return;
                        }
                        var speedInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF3_MachineSpeed.ToString());
                        if (speedInfo != null)
                        {
                            meter += speedInfo.VarValue.ToDecimal() / 60m * 0.1m;
                        }
                    }
                    catch (Exception)
                    {
                        throw;
                    }
                    finally
                    {
                        Thread.Sleep(100);
                    }
                }
                //开始进行赋值动作,计算单面机糊间隙
                //车速系数
                var speedCoefInfos = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().Where(it => it.Position == GluePositionEnum.SF3).OrderBy(it => it.Speed).ToList();
                //基础设置
                var glueSetInfo = BLLFactory<GlueSFSetInfoManager>.Instance.GetFirst(it => it.FluteSF == flute);
                //拿到当前设置的SF糊间隙qdm系数
                decimal glueQdmCoef = sfInfo.Glue;

                decimal glueFormCoef = formSetInfo.F_Glue_SF3_Form_Factor;
                Dictionary<string, decimal> dict = new Dictionary<string, decimal>();
                int index = 0;
                curWeight = (codeMsInfo?.SPC_GlueWeight ?? 0) + (codeLsInfo?.SPC_GlueWeight ?? 0);

                #region 计算供应商品牌对应的克重
                string brandMS = info.BrandMS3;
                string brandLS = info.BrandLS3;
                decimal brandOffsetMS = 0;
                decimal brandOffsetLS = 0;
                if (!string.IsNullOrEmpty(brandMS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandMS && paper.SPC_Code == codeMS).First();
                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetMS += (glueSFBrand?.Offset ?? 0);
                    }

                }

                if (!string.IsNullOrEmpty(brandLS))
                {
                    var pinfo = BLLFactory<PaperCodeBrandInfoManager>.Instance.Context.Queryable<PaperCodeBrandInfo>().LeftJoin<PaperCodeInfo>((brand, paper) => brand.F_PaperCodeID == paper.SPC_ID).Where((brand, paper) => brand.F_Brand == brandLS && paper.SPC_Code == codeLS).First();
                    if (pinfo != null)
                    {
                        var glueSFBrand = BLLFactory<GlueSFBrandSetInfoManager>.Instance.AsQueryable().Where(a => a.BrandID == pinfo.F_ID && a.FluteSF == flute).First();

                        brandOffsetLS += (glueSFBrand?.Offset ?? 0);
                    }
                }

                #endregion
                StringBuilder sb = new StringBuilder();
                foreach (var scInfo in speedCoefInfos)
                {
                    //计算8段车速对应的糊间隙值
                    Dictionary<string, object> dic = new Dictionary<string, object>
                    {
                        { "MinGlueGap", glueSetInfo.MinGlue },
                        { "MaxGlueGap", glueSetInfo.MaxGlue },
                        { "MinGms", glueSetInfo.MinWeight },
                        { "MaxGms", glueSetInfo.MaxWeight },
                        { "CurrentGms", curWeight },
                        { "FluteCoef", glueSetInfo.Coef },
                        { "QDMCoef", 1 },
                        { "AdjustCoef", 1 },
                        { "SpeedCoef", 1 }
                    };

                    decimal setValue = IPSCalMethod.CalGlueGap(dic);//基础值

                    //if (setValue > glueSetInfo.MaxGlue)
                    //    setValue = glueSetInfo.MaxGlue;
                    if (setValue > 60)
                        setValue = 60;

                    decimal baseValue = Math.Round(setValue * scInfo.Coef, 2);

                    setValue = setValue * glueQdmCoef * glueFormCoef * scInfo.Coef;
                    setValue = Math.Round(setValue, 2);

                    decimal offSet = 0;
                    if (GlobalControl.execWarpSetDatail.warpPositionValue.TryGetValue(IpsDriverPositionEnum.GlueSF3, out ActPostionOffSet actPostionOffSet))
                    {
                        offSet = actPostionOffSet.OffSetValue;
                    }
                    decimal brandOffset = Math.Max(brandOffsetMS, brandOffsetLS);
                    decimal unrestrictedSetValue = setValue + brandOffset;

                    setValue += offSet;
                    setValue += brandOffset;

                    sb.AppendLine($"SF3糊间隙计算结果：车速值={scInfo.Speed};最小糊间隙={glueSetInfo.MinGlue};最大糊间隙={glueSetInfo.MaxGlue};最小克重={glueSetInfo.MinWeight};最大克重={glueSetInfo.MaxWeight};当前胶水克重={curWeight};车速系数={scInfo.Coef};车速限制的最小值={scInfo.MinValue ?? 0};QDM系数={glueQdmCoef};界面系数={glueFormCoef};计算结果：{setValue}");

                    if (setValue < Convert.ToDecimal(scInfo.MinValue ?? 0))
                    {
                        sb.AppendLine($"计算值={setValue} < 最小值={Convert.ToDecimal(scInfo.MinValue ?? 0)}，因此设定值={Convert.ToDecimal(scInfo.MinValue ?? 0)}");
                        setValue = Convert.ToDecimal(scInfo.MinValue ?? 0);
                    }
                    switch (index)
                    {
                        case 0:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed0.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value0.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 1:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed1.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value1.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 2:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed2.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value2.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 3:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed3.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value3.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 4:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed4.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value4.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 5:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed5.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value5.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 6:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed6.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value6.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        case 7:
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Speed7.ToString(), scInfo.Speed);
                            dict.Add(PointVarEnum.SF3_GUGap_Curve_Value7.ToString(), Math.Round(setValue / 100m, 2));
                            break;
                        default:
                            break;
                    }
                    index++;
                    GluePositionSpeedValue gluePositionSpeedValue = new GluePositionSpeedValue
                    {
                        Speed = scInfo.Speed,
                        Value = setValue,
                        QdmCoef = glueQdmCoef,
                        FormCoef = glueFormCoef,
                        MinValue = scInfo.MinValue ?? 10,
                        MaxValue = 999,
                        BaseValue = baseValue,
                        UnrestrictedSetValue = unrestrictedSetValue,
                        OffSetValue = offSet,
                        FormQdmCoef = formQdmCoef,
                        RealQdmCoef = realQdmCoef,
                    };
                    GlobalControl.tempSf3.AddOrUpdate(scInfo.Speed, gluePositionSpeedValue, (key, oldValue) => gluePositionSpeedValue);
                }
                string msg = sb.ToString();
                if (!string.IsNullOrEmpty(msg))
                    logger.Info(msg, module);
                sb.Clear();
                sb = null;


                GlobalControl.SetChangeRecord(IpsDriverPositionEnum.GlueSF3, new IpsValueInfo() { FormCoef = formSetInfo.F_Glue_SF3_Form_Factor, RealQdmCoef = realQdmCoef, FormQdmCoef = formQdmCoef }, codeMS + "." + codeLS, info.Flute);


                //机器写值之前再次判断是否要终止线程
                if (token.IsCancellationRequested)
                {
                    logger.Info("SetGlueSF3--产生了一个新的任务，该任务终止", module);
                    return;
                }
                //如果界面没有勾选启用，则不往设备发送
                if (!formSetInfo.F_Glue_SF3_Form_IsOpen)
                {
                    logger.Info("SetGlueSF3--SF3糊间隙没有启用，往设备写值动作终止", module);
                    return;
                }


                foreach (var key in dict.Keys)
                {
                    if (token.IsCancellationRequested)
                    {
                        logger.Info("SF3糊间隙 即将往设备写值，但是因为该期间内又收到一个新的产生了一个新的任务，该任务终止", module);
                        return;
                    }
                    comm.WriteVar(key, dict[key]);
                }
                //写停机糊间隙
                comm.WriteVar(PointVarEnum.SF3_GUGap_Base_Value.ToString(), 0.6);
                comm.WriteVar(PointVarEnum.SF3_GUGap_Offset.ToString(), 0);
                logger.Info($"SetGlueSF3--SF3糊间隙往设备写值动作完成,材质={info.Code},楞型={info.Flute}", module);
            }
            catch (Exception ex)
            {
                logger.Error($"SetGlueSF3--执行异常：{ex}", module);
            }
        }

        /// <summary>
        /// 实时反馈当前设置的糊间隙任务
        /// </summary>
        private void CalGlueRealTime()
        {
            cts_glue = new CancellationTokenSource();
            Task.Factory.StartNew(async () =>
            {

                while (true)
                {
                    try
                    {
                        if (cts_glue.IsCancellationRequested)
                        {
                            return;
                        }

                        //获取当前各糊间隙部位对应的实时车速，车速大于列表中的值的时候，取对应的糊间隙设定值
                        var speedGuInfo = comm.PointVars.Find(it => it.VarCode == PointVarEnum.DF_MachineSpeed.ToString());
                        int speedGu = speedGuInfo == null ? 0 : speedGuInfo.VarValue.ToInt32();

                        #region 校正多余的车速段

                        List<GlueSpeedCoefInfo> glueSpeedCoefInfos = BLLFactory<GlueSpeedCoefInfoManager>.Instance.AsQueryable().ToList();
                        //设备部位 糊机1层的车速系数
                        var speedCoefGu1 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.Gu1).Select(a => a.Speed).ToList();
                        //设备部位 糊机2层的车速系数
                        var speedCoefGu2 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.Gu2).Select(a => a.Speed).ToList();
                        //设备部位 糊机3层的车速系数
                        var speedCoefGu3 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.Gu3).Select(a => a.Speed).ToList();
                        //设备部位 SF1的车速系数
                        var speedCoefSF1 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.SF1).Select(a => a.Speed).ToList();
                        //设备部位 糊机2层的车速系数
                        var speedCoefSF2 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.SF2).Select(a => a.Speed).ToList();
                        //设备部位 糊机3层的车速系数
                        var speedCoefSF3 = glueSpeedCoefInfos.Where(it => it.Position == GluePositionEnum.SF3).Select(a => a.Speed).ToList();

                        List<int> speedGu1s = GlobalControl.tempGu1.Keys.ToList();
                        foreach (var item in speedGu1s)
                        {
                            if (!speedCoefGu1.Contains(item))
                            {
                                GlobalControl.tempGu1.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }


                        List<int> speedGu2s = GlobalControl.tempGu2.Keys.ToList();
                        foreach (var item in speedGu2s)
                        {
                            if (!speedCoefGu2.Contains(item))
                            {
                                GlobalControl.tempGu2.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }

                        List<int> speedGu3s = GlobalControl.tempGu3.Keys.ToList();
                        foreach (var item in speedGu3s)
                        {
                            if (!speedCoefGu3.Contains(item))
                            {
                                GlobalControl.tempGu3.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }

                        List<int> speedSf1s = GlobalControl.tempSf1.Keys.ToList();
                        foreach (var item in speedSf1s)
                        {
                            if (!speedCoefSF1.Contains(item))
                            {
                                GlobalControl.tempSf1.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }

                        List<int> speedSf2s = GlobalControl.tempSf2.Keys.ToList();
                        foreach (var item in speedSf2s)
                        {
                            if (!speedCoefSF2.Contains(item))
                            {
                                GlobalControl.tempSf2.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }

                        List<int> speedSf3s = GlobalControl.tempSf3.Keys.ToList();
                        foreach (var item in speedSf3s)
                        {
                            if (!speedCoefSF3.Contains(item))
                            {
                                GlobalControl.tempSf3.TryRemove(item, out GluePositionSpeedValue gluePositionSpeedValue);
                            }
                        }
                        #endregion


                        var gu1 = GlobalControl.tempGu1.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();

                        for (int i = 0; i < gu1.Count; i++)
                        {
                            var item = gu1[i];
                            //找车速
                            if (speedGu >= item.Speed || i == (gu1.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueGU1);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU1,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU1,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }

                        var gu2 = GlobalControl.tempGu2.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();
                        for (int i = 0; i < gu2.Count; i++)
                        {
                            var item = gu2[i];
                            if (speedGu >= item.Speed || i == (gu2.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueGU2);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU2,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU2,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }

                        //var gu3 = GlobalControl.tempGu3.OrderByDescending(it => it.Speed).ToList();
                        var gu3 = GlobalControl.tempGu3.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();
                        for (int i = 0; i < gu3.Count; i++)
                        {
                            var item = gu3[i];
                            if (speedGu >= item.Speed || i == (gu3.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueGU3);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU3,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueGU3,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }

                        var speedSF1Info = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF1_MachineSpeed.ToString());
                        int speedSf1 = speedSF1Info == null ? 0 : speedSF1Info.VarValue.ToInt32();
                        //var sf1 = GlobalControl.tempSf1.OrderByDescending(it => it.Speed).ToList();

                        var sf1 = GlobalControl.tempSf1.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();

                        for (int i = 0; i < sf1.Count; i++)
                        {
                            var item = sf1[i];
                            if (speedSf1 >= item.Speed || i == (sf1.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueSF1);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF1,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF1,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }

                        var speedSF2Info = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF2_MachineSpeed.ToString());
                        int speedSf2 = speedSF2Info == null ? 0 : speedSF2Info.VarValue.ToInt32();
                        var sf2 = GlobalControl.tempSf2.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();

                        for (int i = 0; i < sf2.Count; i++)
                        {
                            var item = sf2[i];
                            if (speedSf2 >= item.Speed || i == (sf2.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueSF2);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF2,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF2,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }

                        var speedSF3Info = comm.PointVars.Find(it => it.VarCode == PointVarEnum.SF3_MachineSpeed.ToString());
                        int speedSf3 = speedSF3Info == null ? 0 : speedSF3Info.VarValue.ToInt32();
                        var sf3 = GlobalControl.tempSf3.Select(a => a.Value).OrderByDescending(a => a.Speed).ToList();
                        for (int i = 0; i < sf3.Count; i++)
                        {
                            var item = sf3[i];
                            if (speedSf3 >= item.Speed || i == (sf3.Count - 1))
                            {
                                var valueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == IpsDriverPositionEnum.GlueSF3);
                                if (valueInfo == null)
                                {
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF3,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                else
                                {
                                    GlobalControl.ipsValueInfos.Remove(valueInfo);
                                    GlobalControl.ipsValueInfos.Add(new IpsValueInfo
                                    {
                                        Position = IpsDriverPositionEnum.GlueSF3,
                                        SetValue = item.Value,
                                        FormCoef = item.FormCoef,
                                        QdmCoef = item.QdmCoef,
                                        MinValue = item.MinValue,
                                        MaxValue = item.MaxValue,
                                        BaseValue = item.BaseValue,
                                        OffSetValue = item.OffSetValue,
                                        BrandOffSetValue = item.BrandOffSetValue,
                                        UnrestrictedSetValue = item.UnrestrictedSetValue,
                                        FormQdmCoef = item.FormQdmCoef,
                                        RealQdmCoef = item.RealQdmCoef,
                                    });
                                }
                                break;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.Error($"CalGlueRealTime 任务异常出错：{ex}", module);
                    }
                    finally
                    {
                        await Task.Delay(300);
                    }


                }
            }, cts_glue.Token);

        }

        /// <summary>
        /// 立刻赋值消息接收到处理函数
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void HandleChangeNow(object sender, EventArgs e)
        {
            if (sender == null)
                return;
            try
            {
                PubChangeNowInfo msg = (PubChangeNowInfo)sender;
                PublishInfo info = new PublishInfo();
                info.Part = msg.Part;
                info.Flute = msg.Flute;
                info.LastFlute = msg.LastFlute;
                info.Code = msg.Code;
                info.Width = msg.Width;
                info.LastWidth = msg.LastWidth;
                info.LastCode = msg.LastCode;
                info.BrandLS0 = msg.BrandLS0;
                info.BrandLS1 = msg.BrandLS1;
                info.BrandLS2 = msg.BrandLS2;
                info.BrandLS3 = msg.BrandLS3;
                info.BrandMS1 = msg.BrandMS1;
                info.BrandMS2 = msg.BrandMS2;
                info.BrandMS3 = msg.BrandMS3;
                switch (msg.Part)
                {
                    case IPSHandlePart.GlueGu:
                    case IPSHandlePart.GlueGu1:
                    case IPSHandlePart.GlueGu2:
                    case IPSHandlePart.GlueGu3:
                        logger.Info($"HandleChangeNow 糊机糊间隙立即换材，材质={info.Code},楞型={info.Flute},偏移量={msg.OffSetValue}", module);
                        cts_gu.Cancel();
                        cts_gu = new CancellationTokenSource();
                        CancellationToken tokenGU = cts_gu.Token;
                        Task.Run(() => { SetGlueGu(info, tokenGU, true, msg.IsFirst); }, tokenGU);
                        break;
                    case IPSHandlePart.GlueSF1:
                        logger.Info($"HandleChangeNow SF1糊间隙立即换材，材质={info.Code},楞型={info.Flute},偏移量={msg.OffSetValue}", module);
                        cts_sf1.Cancel();
                        cts_sf1 = new CancellationTokenSource();
                        CancellationToken tokenSF1 = cts_sf1.Token;
                        Task.Run(() => { SetGlueSF1(info, tokenSF1, true, msg.IsFirst); }, tokenSF1);
                        break;
                    case IPSHandlePart.GlueSF2:
                        logger.Info($"HandleChangeNow SF2糊间隙立即换材，材质={info.Code},楞型={info.Flute},偏移量={msg.OffSetValue}", module);
                        cts_sf2.Cancel();
                        cts_sf2 = new CancellationTokenSource();
                        CancellationToken tokenSF2 = cts_sf2.Token;
                        Task.Run(() => { SetGlueSF2(info, tokenSF2, true, msg.IsFirst); }, tokenSF2);
                        break;
                    case IPSHandlePart.GlueSF3:
                        logger.Info($"HandleChangeNow SF3糊间隙立即换材，材质={info.Code},楞型={info.Flute},偏移量={msg.OffSetValue}", module);
                        cts_sf3.Cancel();
                        cts_sf3 = new CancellationTokenSource();
                        CancellationToken tokenSF3 = cts_sf3.Token;
                        Task.Run(() => { SetGlueSF3(info, tokenSF3, true, msg.IsFirst); }, tokenSF3);
                        break;
                    default:
                        break;
                }
            }
            catch (Exception ex)
            {
                logger.Error($"立刻给糊间隙赋值异常失败：{ex}", "IPSNew-糊间隙");
            }

        }

        /// <summary>
        /// 通知客户端刷新QDM界面系数
        /// </summary>
        private void SendM108()
        {
            GlobalInfos._m108s.AddOrUpdate("M108", "", (k, v) => "");
            //await GlobalInfos.SendMsg("M108");
        }
        #endregion <方法>

        #region <事件>
        #endregion <事件>
    }
}