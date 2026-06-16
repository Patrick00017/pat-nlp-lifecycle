using BTS.Commons;
using BTS.Dtos;
using BTS.Dtos.Enums;
using BTS.Entites;
using BTS.Entites.IPSNew.Warp;
using BTS.Entites.Report;
using BTS.Logs;
using BTS.Server.Core;
using BTS.Services;
using BTS.Services.Services.IPSNew;
using BTS.Services.Services.IPSNew.FormSetExs;
using BTS.Services.Services.Report;
using Dm.util;
using NPOI.SS.UserModel;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TouchSocket.Http;
using TouchSocket.Sockets;

namespace BTS.Server.IPS
{
    /// <summary>
    /// 弯翘控制类
    /// </summary>
    public class WarpCtrl : IWarpCtrl
    {
        #region 常量

        private const string moduleWarp = "弯翘判定模块";

        #endregion

        #region 变量

        private Log logger;

        private DriverLink comm;

        private IPSMainCtrl iPSMainCtrl;

        /// <summary>
        /// 糊机面底
        /// </summary>
        public DriveStateInfo stateInfo_gu_face_bottom = new DriveStateInfo();

        ///// <summary>
        ///// 当前执行的弯翘状态 (自动或者手动)
        ///// </summary>
        //public static string CurWrapExecStatu = "";  // 空是无状态

        ///// <summary>
        ///// 当前检测设备反馈的弯翘状态
        ///// </summary>
        //public static DetectionStatus CurDetectionStatu = new DetectionStatus();  // 空是无状态


        public int execWrapOrderID { get; set; } = 0;//自动执行弯翘时订单的ID
        public decimal execWrapOrderProductMeters { get; set; } = 0;//自动执行弯翘时订单已生产的米数

        int restWrapOrderID = 0;//面底换材时订单的ID
        decimal restWrapOrderProductMeters = 0;//面底换材时订单已生产的米数


        #endregion

        #region 构造函数

        public WarpCtrl(IPSMainCtrl _iPSMainCtrl, DriverLink _comm)
        {
            comm = _comm;
            iPSMainCtrl = _iPSMainCtrl;
            iPSMainCtrl.OnPubChangePaper += HandleChangePaper;
            logger = LogHelper.GetLogger(typeof(WarpCtrl));

            // 检测是否需要自动执行弯翘
            Task.Run(MonitorWarpAutoExec);

            // 检查是否需要取消弯翘调平
            Task.Run(DFFaceBottomChangePaper);
        }

        #endregion

        #region 方法

        private void HandleChangePaper(object sender, EventArgs e)
        {
            PartPaperCode msg = (PartPaperCode)sender;
            if (msg == null) { return; }
            HandBottomPaperChange(msg.Part);
        }

        /// <summary>
        /// 处理底纸接纸机换材后弯翘相关偏移量重置
        /// </summary>
        /// <param name="MachineCode"></param>
        private void HandBottomPaperChange(string MachineCode)
        {
            try
            {

                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);
                var formsetexInfos = BLLFactory<FormSetExInfoManager>.Instance.GetList();
                //if (formSetInfo.F_Warp_Form_IsOpen == false) return;

                OrderInfo curOrder = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();

                if (curOrder == null || curOrder.WO_ID == 0) return;

                try
                {

                    List<string> lstPaperCode = new List<string>();

                    if (curOrder.WO_PaperCode.Contains("."))
                    {
                        lstPaperCode = curOrder.WO_PaperCode.Split('.').ToList();
                    }
                    else
                    {
                        lstPaperCode = curOrder.WO_PaperCode.ToCharArray().Select(a => a.ToString()).ToList();
                    }

                    int lastCodeIndex = lstPaperCode.FindLastIndex(a => a != "-");
                    int postionCount = lastCodeIndex + 1;
                    if ((MachineCode == "LS1" && postionCount == 3)
                         || (MachineCode == "LS2" && postionCount == 4)
                         || (MachineCode == "LS2" && postionCount == 5)
                         || (MachineCode == "LS3" && postionCount == 6)
                         || (MachineCode == "LS3" && postionCount == 7)
                        )
                    {

                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS1, out ActPostionOffSet WrapLS1))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrap, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS1, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS1);

                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS2, out ActPostionOffSet WrapLS2))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrap, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS2, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS2);
                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS3, out ActPostionOffSet WrapLS3))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrap, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS3, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS3);
                        }


                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS1ext, out ActPostionOffSet WrapLS1ext))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrapExt, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS1ext, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS1ext);

                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS2ext, out ActPostionOffSet WrapLS2ext))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrapExt, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS2ext, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS2ext);
                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapLS3ext, out ActPostionOffSet WrapLS3ext))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrapExt, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS3ext, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.WrapLS3ext);
                        }



                        // 温度模式下只移除偏移量 等包角控制类赋值
                        GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapTemperatureLS1, out ActPostionOffSet TWrapLS1);
                        GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapTemperatureLS2, out ActPostionOffSet TWrapLS2);
                        GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.WrapTemperatureLS3, out ActPostionOffSet TWrapLS3);



                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.TensionLS1, out ActPostionOffSet TensionLS1))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomBridgeTension, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.TensionLS1, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.TensionLS1);

                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.TensionLS2, out ActPostionOffSet TensionLS2))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomBridgeTension, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.TensionLS2, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.TensionLS2);
                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.TensionLS3, out ActPostionOffSet TensionLS3))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomBridgeTension, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.TensionLS3, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.TensionLS3);

                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.GlueSF1, out ActPostionOffSet GlueSF1))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFGlue, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.GlueSF1, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.GlueSF1);
                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.GlueSF2, out ActPostionOffSet GlueSF2))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFGlue, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.GlueSF2, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.GlueSF2);
                        }
                        if (GlobalControl.execWarpSetDatail.warpPositionValue.TryRemove(IpsDriverPositionEnum.GlueSF3, out ActPostionOffSet GlueSF3))
                        {
                            ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFGlue, OffSetValue = 0 };
                            GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.GlueSF3, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                            //添加偏移量后发布
                            WarpPublish(IpsDriverPositionEnum.GlueSF3);
                        }
                    }
                }
                catch (Exception ex)
                {
                    logger.Error(ex.ToString(), moduleWarp);
                }
                finally
                {

                }


            }
            catch (Exception ex) { }

        }

        /// <summary>
        /// 检测是否需要自动执行弯翘
        /// </summary>
        /// <returns></returns>
        private async Task MonitorWarpAutoExec()
        {
            List<SpeedInfo> speedInfos = new List<SpeedInfo>();
            while (true)
            {
                try
                {
                    int DFSpeed = (comm.PointVars.Find(it => it.VarCode == PointVarEnum.DF_MachineSpeed.ToString())?.VarValue.ToInt32()) ?? 0;
                    DateTime dtNow = DateTime.Now;
                    speedInfos.Add(new SpeedInfo() { RecordTime = dtNow, Speed = DFSpeed });
                    List<DictDataInfo> datas = BLLFactory<DictDataInfoManager>.Instance.GetDictItemsToModelList(DictTypesEnum.WarpSet.ToString());
                    string WarpSpeed = datas.FirstOrDefault(it => it.PD_Property == "WarpSpeed")?.PD_Value;
                    string WarpGapMeter = datas.FirstOrDefault(it => it.PD_Property == "WarpGapMeter")?.PD_Value;
                    string SamePaperMeter = datas.FirstOrDefault(it => it.PD_Property == "SamePaperMeter")?.PD_Value;
                    string WarpSpeedTime = datas.FirstOrDefault(it => it.PD_Property == "WarpSpeedTime")?.PD_Value;
                    Int32.TryParse(WarpSpeed, out int iWarpSpeed);
                    Int32.TryParse(WarpSpeedTime, out int iWarpSpeedTime);
                    decimal.TryParse(WarpGapMeter, out decimal dWarpGapMeter);
                    decimal.TryParse(SamePaperMeter, out decimal dSamePaperMeter);
                    speedInfos.RemoveAll(a => a.RecordTime < dtNow.AddSeconds((-1) * iWarpSpeedTime));

                    // 换单不执行
                    if (GlobalControl.isDoingChangeOrder) continue;

                    FormSetFactorInfo formSetFactorInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(a => 1 == 1);

                    //自动控制开关没开不自动执行
                    if (formSetFactorInfo.F_Warp_Form_IsOpen == false) continue;

                    // 车速不满足执行的条件不执行
                    if (DFSpeed <= iWarpSpeed) continue;

                    // 判断车速连续多少秒内是否满足要求
                    if (speedInfos.Exists(a => a.Speed <= iWarpSpeed)) continue;

                    //当面底同材剩余米数小于设置值时，不执行自动调平
                    if (GlobalControl.dfFaceBottomSameMaterilaLeftMeter < dSamePaperMeter) continue;

                    string strCurDetectionStatu = GlobalControl.CurDetectionStatu.JudgeWarpStatuCmd;
                    // 设备没有给(或者S弯、无纸板、正常、或者给的度数配置里找不到)当前检测的状态不执行
                    if (string.IsNullOrWhiteSpace(strCurDetectionStatu)) continue;

                    OrderInfo orderInfo = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();

                    if (orderInfo == null) continue;

                    // 产品20250812新的需求四层和六层也执行自动调平
                    // if (orderInfo.WO_Floor == 4 || orderInfo.WO_Floor == 6) continue;// 4层或者6层板不执行自动调平

                    // 当前已执行过调平，需要判断调平的米数间隔
                    if (GlobalControl.CurWrapExecStatu != "" && execWrapOrderID != 0)
                    {
                        decimal dProductMeter = 0;
                        // 还是同一个订单
                        if (orderInfo.WO_ID == execWrapOrderID)
                        {
                            // 计算生产米数
                            dProductMeter = GlobalControl.curOrderProduct - execWrapOrderProductMeters;
                        }
                        else // 非同一笔订单
                        {
                            var result = BLLFactory<ProEventP04PInfoManager>.Instance.Context.Queryable<ProEventP04PInfo>().Where(a => a.SetupID == execWrapOrderID)
                                 .InnerJoin<OrderFinishInfo>((a, b) => a.SetupID == b.WO_ID).Select((a, b) => new { ActualCuts1 = a.ActualCuts1, P04PID = a.ID, CutLen = b.WOF_CutLength }).ToList();
                            decimal dProductLen = result.Sum(a => (a.ActualCuts1 ?? 0) * 0.001m * a.CutLen);
                            dProductMeter = dProductLen - execWrapOrderProductMeters;
                            if (dProductMeter < 0)
                            {
                                dProductMeter = 0;
                            }
                            int P04PID = result.Count > 0 ? result[0].P04PID : 0;
                            if (P04PID > 0)
                            {

                                dProductMeter += BLLFactory<ProEventP04PInfoManager>.Instance.Context.Queryable<ProEventP04PInfo>().Where(a => a.ID > P04PID && a.SetupID != execWrapOrderID && a.SetupID != orderInfo.WO_ID)
                                     .InnerJoin<OrderFinishInfo>((a, b) => a.SetupID == b.WO_ID).Select((a, b) => new { ActualCuts1 = a.ActualCuts1, CutLen = b.WOF_CutLength }).ToList().Sum(a => (a.ActualCuts1 ?? 0) * 0.001m * a.CutLen);

                            }
                            dProductMeter += GlobalControl.curOrderProduct;
                        }

                        // 米数间隔不满足不执行
                        if (dProductMeter < dWarpGapMeter) continue;
                    }

                    // 面底换过材质的情况下 需要判断换材后的米数是否满足执行条件
                    if (restWrapOrderID > 0)
                    {
                        decimal dProductMeter = 0;
                        // 还是同一个订单
                        if (orderInfo.WO_ID == restWrapOrderID)
                        {
                            // 计算生产米数
                            dProductMeter = GlobalControl.curOrderProduct - restWrapOrderProductMeters;
                        }
                        else // 非同一笔订单
                        {
                            var result = BLLFactory<ProEventP04PInfoManager>.Instance.Context.Queryable<ProEventP04PInfo>().Where(a => a.SetupID == restWrapOrderID)
                                 .InnerJoin<OrderFinishInfo>((a, b) => a.SetupID == b.WO_ID).Select((a, b) => new { ActualCuts1 = a.ActualCuts1, P04PID = a.ID, CutLen = b.WOF_CutLength }).ToList();
                            decimal dProductLen = result.Sum(a => (a.ActualCuts1 ?? 0) * 0.001m * a.CutLen);
                            dProductMeter = dProductLen - restWrapOrderProductMeters;
                            if (dProductMeter < 0)
                            {
                                dProductMeter = 0;
                            }
                            int P04PID = result.Count > 0 ? result[0].P04PID : 0;
                            if (P04PID > 0)
                            {

                                dProductMeter += BLLFactory<ProEventP04PInfoManager>.Instance.Context.Queryable<ProEventP04PInfo>().Where(a => a.ID > P04PID && a.SetupID != restWrapOrderID && a.SetupID != orderInfo.WO_ID)
                                     .InnerJoin<OrderFinishInfo>((a, b) => a.SetupID == b.WO_ID).Select((a, b) => new { ActualCuts1 = a.ActualCuts1, CutLen = b.WOF_CutLength }).ToList().Sum(a => (a.ActualCuts1 ?? 0) * 0.001m * a.CutLen);
                            }
                            dProductMeter += GlobalControl.curOrderProduct;
                        }

                        // 米数间隔不满足不执行
                        if (dProductMeter < dWarpGapMeter) continue;
                    }

                    //查询弯翘调整项目
                    List<WarpWaveSetInfo> curvedWarpSetInfos = BLLFactory<WarpWaveSetInfoManager>.Instance.GetList();
                    string exeCmd = strCurDetectionStatu;
                    if (GlobalControl.CurWrapExecStatu != "") // 当前正在执行弯翘调整
                    {
                        int result = 0;
                        result += CalWarpValue(GlobalControl.CurWrapExecStatu);
                        result += CalWarpValue(strCurDetectionStatu);

                        if (result >= 3)
                        {
                            exeCmd = "UP3";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行上弯重", moduleWarp);
                        }
                        else if (result == 2)
                        {
                            exeCmd = "UP2";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行上弯中", moduleWarp);
                        }
                        else if (result == 1)
                        {
                            exeCmd = "UP1";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行上弯轻", moduleWarp);
                        }
                        else if (result == -1)
                        {
                            exeCmd = "DOWN1";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行下弯轻", moduleWarp);
                        }
                        else if (result == -2)
                        {
                            exeCmd = "DOWN2";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行下弯中", moduleWarp);
                        }
                        else if (result <= -3)
                        {
                            exeCmd = "DOWN3";
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行下弯重", moduleWarp);
                        }
                        else if (result == 0) // 需要重置调平
                        {
                            logger.Info($"当前执行的是{GlobalControl.CurWrapExecStatu}，现在设备检测出来是{strCurDetectionStatu},执行重置调平", moduleWarp);
                            RestCurvedWarp(true);
                            execWrapOrderID = (GlobalControl.curOrder?.WO_ID) ?? 0;
                            execWrapOrderProductMeters = GlobalControl.curOrderProduct;
                            try
                            {
                                await GlobalInfos.SendMsg("M104", "REST");
                            }
                            catch
                            {

                            }
                            continue;
                        }
                    }

                    logger.Info($"执行自动弯翘，调用弯翘调平方法ExecCurvedWarp", moduleWarp);
                    // 执行自动弯翘 记录执行此刻的订单
                    ExecCurvedWarp(orderInfo, exeCmd, curvedWarpSetInfos, true);

                    try
                    {
                        await GlobalInfos.SendMsg("M104", exeCmd);
                        //string strMsg = (Encoding.Default.GetString(GlobalControl.bStart) + "M104" + exeCmd);
                        //var tcpOnlineClients = tcpService.GetClients();
                        //foreach (var client in tcpOnlineClients)
                        //{
                        //    if (client.Online)
                        //    {
                        //        try
                        //        {
                        //            client.MainSocket.SendTimeout = 500;
                        //            client.MainSocket.ReceiveTimeout = 500;
                        //            client.Send(Encoding.Default.GetBytes(strMsg));
                        //        }
                        //        catch { }
                        //    }
                        //}
                    }
                    catch
                    {

                    }
                }
                catch (Exception ex)
                {
                    logger.Error(ex.ToString(), moduleWarp);
                }
                finally
                {
                    await Task.Delay(1000);
                }
            }
        }

        /// <summary>
        /// 计算弯翘对应的值
        /// </summary>
        /// <param name="CurWrapExecStatu"></param>
        /// <returns></returns>
        private int CalWarpValue(string CurWrapExecStatu)
        {
            int result = 0;
            if (CurWrapExecStatu == "UP3")
            {
                result += 3;
            }
            else if (CurWrapExecStatu == "UP2")
            {
                result += 2;
            }
            else if (CurWrapExecStatu == "UP1")
            {
                result += 1;
            }
            else if (CurWrapExecStatu == "DOWN1")
            {
                result += (-1);
            }
            else if (CurWrapExecStatu == "DOWN2")
            {
                result += (-2);
            }
            else if (CurWrapExecStatu == "DOWN3")
            {
                result += (-3);
            }

            return result;
        }

        /// <summary>
        /// 根据弯翘结果执行弯翘调平 (先取消之前的调平变量内容再执行新的调平)
        /// </summary>
        public void ExecCurvedWarp(OrderInfo curOrder, string cmd, List<WarpWaveSetInfo> curvedWarpSetInfos, bool isAuto)
        {
            try
            {
                if (string.IsNullOrEmpty(cmd)) return;
                RestCurvedWarp(); // 重置弯翘
                Thread.Sleep(800);
                GlobalControl.CurWrapExecStatu = cmd;
                if (isAuto)
                {
                    execWrapOrderID = (curOrder?.WO_ID) ?? 0;
                    execWrapOrderProductMeters = GlobalControl.curOrderProduct;
                }
                else
                {
                    execWrapOrderID = 0;
                    execWrapOrderProductMeters = 0;
                }
                //判断是几层
                int floor = curOrder.WO_Wave.Substring(0, 1).ToInt16();
                string wave = curOrder.WO_Wave;
                List<string> lstPaperCode = new List<string>();

                if (curOrder.WO_PaperCode.Contains("."))
                {
                    lstPaperCode = curOrder.WO_PaperCode.Split('.').ToList();
                }
                else
                {
                    lstPaperCode = curOrder.WO_PaperCode.ToCharArray().Select(it => it.ToString()).ToList();
                }
                List<PositonValue> positonValues = new List<PositonValue>();
                curvedWarpSetInfos = curvedWarpSetInfos.Where(a => a.Wave == wave && a.F_Position != WarpPositionEnum.WarpStand).ToList();
                if (curvedWarpSetInfos.Count == 0) return;


                switch (cmd)
                {
                    case "UP1":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Up1 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Up1 }).ToList();
                        break;
                    case "UP2":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Up2 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Up2 }).ToList();
                        break;
                    case "UP3":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Up3 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Up3 }).ToList();
                        break;
                    case "DOWN1":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Down1 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Down1 }).ToList();
                        break;
                    case "DOWN2":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Down2 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Down2 }).ToList();
                        break;
                    case "DOWN3":
                        positonValues = curvedWarpSetInfos.Where(a => a.F_Down3 != null).Select(a => new PositonValue() { F_Position = a.F_Position, F_Value = (decimal)a.F_Down3 }).ToList();
                        break;

                    default:
                        break;

                }
                if (positonValues.Count == 0) return;

                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);

                string bottomMachine = "LS1"; //底纸接纸机机台
                IpsDriverPositionEnum actWrapPart = IpsDriverPositionEnum.WrapGU1;


                int lastCharIndex = lstPaperCode.FindLastIndex(a => a != "-");
                int lastCharCount = lastCharIndex + 1;

                if (lastCharCount == 3)   // 底纸接纸机是LS1
                {
                    bottomMachine = "LS1";
                    actWrapPart = IpsDriverPositionEnum.WrapGU1;
                }
                else if (lastCharCount == 5)// 底纸接纸机是LS2
                {
                    bottomMachine = "LS2";
                    actWrapPart = IpsDriverPositionEnum.WrapGU2;
                }
                else if (lastCharCount == 7)// 底纸接纸机是LS3
                {
                    bottomMachine = "LS3";
                    actWrapPart = IpsDriverPositionEnum.WrapGU3;
                }

                //把用户当前启用的糊机包角部位放入顺序列表中
                List<PointVarEnum> wrapDrivers = new List<PointVarEnum>();
                if (formSetInfo.F_Wrap_GU_1st_Form_IsOn)
                    wrapDrivers.Add(PointVarEnum.DF_PHWrap_Nominal_1);
                if (formSetInfo.F_Wrap_GU_2nd_Form_IsOn)
                    wrapDrivers.Add(PointVarEnum.DF_PHWrap_Nominal_2);
                if (formSetInfo.F_Wrap_GU_3rd_Form_IsOn)
                    wrapDrivers.Add(PointVarEnum.DF_PHWrap_Nominal_3);

                // 处理底纸的胶水
                switch (lstPaperCode.Where(a => a != "-").Count())
                {
                    case 3:
                        if (wrapDrivers.Count == 1)
                        {
                            if (wrapDrivers[0] == PointVarEnum.DF_PHWrap_Nominal_1)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU1;
                            }
                            else if (wrapDrivers[0] == PointVarEnum.DF_PHWrap_Nominal_2)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU2;
                            }
                            else if (wrapDrivers[0] == PointVarEnum.DF_PHWrap_Nominal_3)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU3;
                            }
                        }
                        break;
                    case 4:
                    case 5:
                        if (wrapDrivers.Count == 2)
                        {
                            if (wrapDrivers[1] == PointVarEnum.DF_PHWrap_Nominal_1)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU1;
                            }
                            else if (wrapDrivers[1] == PointVarEnum.DF_PHWrap_Nominal_2)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU2;
                            }
                            else if (wrapDrivers[1] == PointVarEnum.DF_PHWrap_Nominal_3)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU3;
                            }
                        }
                        break;
                    case 6:
                    case 7:
                        if (wrapDrivers.Count == 3)
                        {
                            if (wrapDrivers[2] == PointVarEnum.DF_PHWrap_Nominal_1)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU1;
                            }
                            else if (wrapDrivers[2] == PointVarEnum.DF_PHWrap_Nominal_2)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU2;
                            }
                            else if (wrapDrivers[2] == PointVarEnum.DF_PHWrap_Nominal_3)
                            {
                                actWrapPart = IpsDriverPositionEnum.WrapGU3;
                            }
                        }
                        break;
                    default:
                        break;
                }
                bool isGlueMap = true;
                //把用户当前启用的糊机包角部位放入顺序列表中
                List<GluePositionEnum> glueDrivers = new List<GluePositionEnum>();
                if (formSetInfo.F_Glue_GU_1st_Form_IsOn)
                    glueDrivers.Add(GluePositionEnum.Gu1);
                if (formSetInfo.F_Glue_GU_2nd_Form_IsOn)
                    glueDrivers.Add(GluePositionEnum.Gu2);
                if (formSetInfo.F_Glue_GU_3rd_Form_IsOn)
                    glueDrivers.Add(GluePositionEnum.Gu3);

                GlobalControl.execWarpSetDatail.BottomSP = bottomMachine;

                var formsetexInfos = BLLFactory<FormSetExInfoManager>.Instance.GetList();
                bool isGUUseTemperatureMode = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureGUIsOpen.ToString())?.Value?.ToBoolean() ?? false;
                bool isSF1UseTemperatureMode = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureSF1IsOpen.ToString())?.Value?.ToBoolean() ?? false;
                bool isSF2UseTemperatureMode = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureSF2IsOpen.ToString())?.Value?.ToBoolean() ?? false;
                bool isSF3UseTemperatureMode = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureSF3IsOpen.ToString())?.Value?.ToBoolean() ?? false;
                bool isLS1TemperatureOn = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureLS1IsOn.ToString())?.Value?.ToBoolean() ?? false;
                bool isLS2TemperatureOn = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureLS2IsOn.ToString())?.Value?.ToBoolean() ?? false;
                bool isLS3TemperatureOn = formsetexInfos.FirstOrDefault(it => it.Code == FormSetExEnum.WrapTemperatureLS3IsOn.ToString())?.Value?.ToBoolean() ?? false;

                foreach (var item in positonValues)
                {
                    if (item.F_Position == WarpPositionEnum.FaceGUWrap) // 面纸糊机包角
                    {
                        //IpsValueInfo valueInfo = GetCurSetInfo(IpsDriverPositionEnum.WrapLS0);
                        //if (valueInfo == null) { continue; }
                        //decimal setValue = valueInfo.UnrestrictedSetValue + item.F_Value;
                        //if (setValue > valueInfo.MaxValue)
                        //{
                        //    setValue = valueInfo.MaxValue;
                        //}
                        //if (setValue < valueInfo.MinValue)
                        //{
                        //    setValue = valueInfo.MinValue;
                        //}
                        //valueInfo.OffSetValue = item.F_Value;
                        //valueInfo.SetValue = setValue;
                        //GlobalControl.ipsValueInfos.Remove(valueInfo);
                        //GlobalControl.ipsValueInfos.Add(valueInfo);
                        //if (formSetInfo.F_Wrap_GU_0_Form_IsOpen && !isGUUseTemperatureMode)
                        //    comm.WriteVar(PointVarEnum.DF_PHWrap_Nominal_0.ToString(), setValue);
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.FaceGUWrap, OffSetValue = item.F_Value };
                        //面纸包角
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapLS0, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.WrapLS0);
                    }
                    else if (item.F_Position == WarpPositionEnum.FaceGUGlue) // 面纸糊机胶水
                    {

                        IpsDriverPositionEnum actGluePart = IpsDriverPositionEnum.GlueGU1;

                        // 处理面值胶水 没有面值的上面弯翘调平会通过部位过滤掉 所以不考虑4层、6层的情况
                        switch (lstPaperCode.Where(a => a != "-").Count())
                        {
                            case 3:
                                if (glueDrivers.Count == 1)
                                {
                                    if (glueDrivers[0] == GluePositionEnum.Gu1)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu2)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU2;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu3)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU3;
                                    }
                                }
                                else
                                {
                                    isGlueMap = false;
                                }
                                break;
                            case 5:
                                if (glueDrivers.Count == 2)
                                {
                                    if (glueDrivers[0] == GluePositionEnum.Gu1)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu2)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU2;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu3)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU3;
                                    }
                                }
                                else
                                {
                                    isGlueMap = false;
                                }
                                break;
                            case 7:
                                if (glueDrivers.Count == 3)
                                {
                                    if (glueDrivers[0] == GluePositionEnum.Gu1)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu2)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU3;
                                    }
                                    else if (glueDrivers[0] == GluePositionEnum.Gu3)
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU3;
                                    }
                                }
                                else
                                {
                                    isGlueMap = false;
                                }
                                break;
                        }

                        // 用户的选择与实际的选择不一致 使用默认情况赋值
                        if (isGlueMap == false)
                        {
                            switch (lstPaperCode.Count)
                            {
                                case 3:
                                    if (lstPaperCode[1] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    break;
                                case 5:
                                    if (lstPaperCode[1] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    else if (lstPaperCode[3] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU2;
                                    }
                                    break;
                                case 7:
                                    if (lstPaperCode[1] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU1;
                                    }
                                    else if (lstPaperCode[3] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU2;
                                    }
                                    else if (lstPaperCode[5] != "-")
                                    {
                                        actGluePart = IpsDriverPositionEnum.GlueGU3;
                                    }
                                    break;
                            }
                        }

                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.FaceGUGlue, OffSetValue = item.F_Value };
                        //面纸糊机胶水
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(actGluePart, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(actGluePart);

                    }
                    else if (item.F_Position == WarpPositionEnum.FaceSPTension) // 面纸接纸机张力
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.FaceSPTension, OffSetValue = item.F_Value };
                        //面纸张力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.TensionLS0, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.TensionLS0);

                    }
                    else if (item.F_Position == WarpPositionEnum.BottomGUWrap) // 底纸糊机包角
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomGUWrap, OffSetValue = item.F_Value };
                        //底纸糊机包角
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(actWrapPart, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(actWrapPart);

                    }
                    else if (item.F_Position == WarpPositionEnum.BottomSFWrap) // 底纸坑机包角
                    {

                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.WrapLS1;
                        IpsDriverPositionEnum ipsDriverPositionEnumEx = IpsDriverPositionEnum.WrapLS1ext;

                        if (bottomMachine == "LS1")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapLS1;
                            ipsDriverPositionEnumEx = IpsDriverPositionEnum.WrapLS1ext;
                        }
                        if (bottomMachine == "LS2")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapLS2;
                            ipsDriverPositionEnumEx = IpsDriverPositionEnum.WrapLS2ext;
                        }
                        if (bottomMachine == "LS3")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapLS3;
                            ipsDriverPositionEnumEx = IpsDriverPositionEnum.WrapLS3ext;
                        }

                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrap, OffSetValue = item.F_Value };
                        //底纸坑机包角
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);


                        ActPostionOffSet actPostionOffSetEx = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrapExt, OffSetValue = item.F_Value };
                        //底纸坑机包角附加
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnumEx, actPostionOffSetEx, (key, oldValue) => actPostionOffSetEx);

                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnumEx);
                    }
                    else if (item.F_Position == WarpPositionEnum.BottomSFGlue)//底纸坑机胶水
                    {
                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.GlueSF1;
                        if (bottomMachine == "LS1")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.GlueSF1;
                        }
                        if (bottomMachine == "LS2")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.GlueSF2;
                        }
                        if (bottomMachine == "LS3")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.GlueSF3;

                        }
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFGlue, OffSetValue = item.F_Value };
                        //底纸坑机胶水
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);
                    }
                    else if (item.F_Position == WarpPositionEnum.BottomSPTension)// 底纸接纸机张力 
                    {
                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.TensionLS1;
                        IpsValueInfo valueInfo = new IpsValueInfo();
                        //bool isCanExce = false;
                        if (bottomMachine == "LS1")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.TensionLS1;

                        }
                        if (bottomMachine == "LS2")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.TensionLS2;

                        }
                        if (bottomMachine == "LS3")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.TensionLS3;

                        }
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSPTension, OffSetValue = item.F_Value };
                        //底纸接纸机张力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);
                    }
                    else if (item.F_Position == WarpPositionEnum.BottomBridgeTension)
                    {
                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.BridgeTension1;

                        if (bottomMachine == "LS1")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.BridgeTension1;
                        }
                        else if (bottomMachine == "LS2")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.BridgeTension2;
                        }
                        else if (bottomMachine == "LS3")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.BridgeTension3;
                        }


                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomBridgeTension, OffSetValue = item.F_Value };
                        //底纸接纸机张力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);

                    }
                    else if (item.F_Position == WarpPositionEnum.DFHotLoadGroupQty) // DF压板组数
                    {
                        GlobalControl.dicHotLoadGroupQty.AddOrUpdate("OffSet", Convert.ToInt32(item.F_Value), (key, oldValue) => Convert.ToInt32(item.F_Value));
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DFHotLoadGroupQty, OffSetValue = item.F_Value };
                        //DF压板组数
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.HotLoadGroupQty, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.HotLoadGroupQty);
                    }
                    else if (item.F_Position == WarpPositionEnum.DFHotPlatePress) // DF热板压力 三段
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DFHotPlatePress, OffSetValue = item.F_Value };
                        //DF热板压力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.HotPlatePress, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.HotPlatePress);
                    }
                    else if (item.F_Position == WarpPositionEnum.DFHotPlate2Press) // DF热板压力 二段
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DFHotPlate2Press, OffSetValue = item.F_Value };
                        //DF热板压力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.HotPlatePress2, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.HotPlatePress2);
                    }
                    else if (item.F_Position == WarpPositionEnum.DFHotPlate1Press) // DF热板压力 一段
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DFHotPlate1Press, OffSetValue = item.F_Value };
                        //DF热板压力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.HotPlatePress1, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.HotPlatePress1);
                    }
                    else if (item.F_Position == WarpPositionEnum.DFColdPlatePress) // DF冷板压力
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DFColdPlatePress, OffSetValue = item.F_Value };
                        //DF冷板压力
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.ColdPlatePress, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.ColdPlatePress);
                    }
                    else if (item.F_Position == WarpPositionEnum.DF1Stream) // 一段蒸汽
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DF1Stream, OffSetValue = item.F_Value };
                        //一段蒸汽
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.DFSteamPart1, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //蒸汽不需要Publish 蒸汽控制实时写值

                    }
                    else if (item.F_Position == WarpPositionEnum.DF2Stream) // 二段蒸汽
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DF2Stream, OffSetValue = item.F_Value };
                        //二段蒸汽
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.DFSteamPart2, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //蒸汽不需要Publish 蒸汽控制实时写值
                    }
                    else if (item.F_Position == WarpPositionEnum.DF3Stream) // 三段蒸汽
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DF3Stream, OffSetValue = item.F_Value };
                        //三段蒸汽
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.DFSteamPart3, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //蒸汽不需要Publish 蒸汽控制实时写值
                    }
                    else if (item.F_Position == WarpPositionEnum.DF4Stream) // 三段蒸汽
                    {
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.DF4Stream, OffSetValue = item.F_Value };
                        //4段蒸汽
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.DFSteamPart4, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //蒸汽不需要Publish 蒸汽控制实时写值
                    }
                    else if (item.F_Position == WarpPositionEnum.FaceGUWrapTemperature) // 面纸糊机包角温度模式
                    {

                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.FaceGUWrapTemperature, OffSetValue = item.F_Value };
                        //面纸包角
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(IpsDriverPositionEnum.WrapTemperatureLS0, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(IpsDriverPositionEnum.WrapTemperatureLS0);

                    }
                    else if (item.F_Position == WarpPositionEnum.BottomGUWrapTemperature) // 糊机底纸包角温度模式
                    {
                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureGU1;

                        if (actWrapPart == IpsDriverPositionEnum.WrapGU1)
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureGU1;
                        }
                        else if (actWrapPart == IpsDriverPositionEnum.WrapGU2)
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureGU2;
                        }
                        else if (actWrapPart == IpsDriverPositionEnum.WrapGU3)
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureGU3;
                        }

                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomGUWrapTemperature, OffSetValue = item.F_Value };

                        //底纸糊机包角
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);

                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);

                    }
                    else if (item.F_Position == WarpPositionEnum.BottomSFWrapTemperature) // 底纸坑机包角温度模式
                    {
                        IpsDriverPositionEnum ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureLS1;
                        if (bottomMachine == "LS1")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureLS1;
                        }
                        if (bottomMachine == "LS2")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureLS2;
                        }
                        if (bottomMachine == "LS3")
                        {
                            ipsDriverPositionEnum = IpsDriverPositionEnum.WrapTemperatureLS3;
                        }
                        ActPostionOffSet actPostionOffSet = new ActPostionOffSet() { WarpPosition = WarpPositionEnum.BottomSFWrap, OffSetValue = item.F_Value };
                        //底纸坑机包角 温度模式
                        GlobalControl.execWarpSetDatail.warpPositionValue.AddOrUpdate(ipsDriverPositionEnum, actPostionOffSet, (key, oldValue) => actPostionOffSet);
                        //添加偏移量后发布
                        WarpPublish(ipsDriverPositionEnum);
                    }
                }
            }
            catch (Exception ex)
            {
                logger.Error(ex.ToString(), moduleWarp);
            }
        }

        /// <summary>
        /// 重置调平
        /// </summary>
        public void RestCurvedWarp(bool isAutoExeRest = false)
        {
            logger.Info("进入重置调平方法RestCurvedWarp", moduleWarp);
            try
            {
                if (isAutoExeRest)
                {
                    GlobalControl.CurWrapExecStatu = "取消调平";
                }
                else
                {
                    GlobalControl.CurWrapExecStatu = "";
                }

                //拿到当前运行的界面系数
                var formSetInfo = BLLFactory<FormSetQdmFactorInfoManager>.Instance.GetFirst(it => 1 == 1);

                Dictionary<IpsDriverPositionEnum, ActPostionOffSet> dic = new Dictionary<IpsDriverPositionEnum, ActPostionOffSet>();
                foreach (var item in GlobalControl.execWarpSetDatail.warpPositionValue)
                {
                    dic.Add(item.Key, item.Value);
                }
                GlobalControl.execWarpSetDatail.warpPositionValue.Clear(); // 清空执行的部位
                foreach (var item in dic)
                {
                    // 压板组数特殊处理
                    if (item.Key == IpsDriverPositionEnum.HotLoadGroupQty)
                    {
                        GlobalControl.dicHotLoadGroupQty.AddOrUpdate("OffSet", 0, (key, oldValue) => 0);
                    }

                    WarpPublish(item.Key);
                }
            }
            catch (Exception ex)
            {
                logger.Error(ex.ToString(), moduleWarp);
            }
            finally
            {
            }
        }

        /// <summary>
        /// 获取当前设置的值
        /// </summary>
        /// <param name="ipsDriverPositionEnum"></param>
        /// <returns></returns>
        private IpsValueInfo GetCurSetInfo(IpsDriverPositionEnum ipsDriverPositionEnum)
        {
            IpsValueInfo ipsValueInfo = null;
            ipsValueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == ipsDriverPositionEnum);
            if (ipsValueInfo == null)
            {
                Thread.Sleep(200);
                ipsValueInfo = GlobalControl.ipsValueInfos.Find(it => it.Position == ipsDriverPositionEnum);
            }
            return ipsValueInfo;
        }

        /// <summary>
        /// 弯翘调整
        /// </summary>
        /// <param name="msg">
        /// 调整内容 
        /// UP1----上弯轻
        /// UP2----上弯中
        /// UP3----上弯重
        /// DOWN1---下弯轻
        /// DOWN2---下弯中
        /// DOWN3---下弯重
        /// RESET_UP1----上弯轻复位
        /// RESET_UP2----上弯中复位
        /// RESET_UP3----上弯重复位
        /// RESET_DOWN1---下弯轻复位
        /// RESET_DOWN2---下弯中复位
        /// RESET_DOWN3---下弯重复位
        /// </param>
        public void HandleCurvedWarp(string msg)
        {
            //获取弯翘调整设置
            //按照设置得到需要调整的项目
            //发布调整消息，带入偏移量，此类消息当业务处理类订阅到之后立刻执行，没有延迟处理
            string cmd = msg.ToUpper();
            //查询弯翘调整项目
            List<WarpWaveSetInfo> curvedWarpSetInfos = BLLFactory<WarpWaveSetInfoManager>.Instance.GetList();
            //当前正在生产的订单
            var curOrderInfo = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();

            if (curOrderInfo == null) return;


            if (cmd.Contains("RESET"))
            {
                string spl = cmd.Split('_')[1];

                logger.Info("执行手动重置弯翘,调用重置方法RestCurvedWarp", moduleWarp);
                RestCurvedWarp();
            }
            else
            {
                logger.Info("执行手动弯翘,调用执行方法ExecCurvedWarp", moduleWarp);
                ExecCurvedWarp(curOrderInfo, cmd, curvedWarpSetInfos, false);
            }
        }

        /// <summary>
        /// 糊机面底换材监听任务
        /// </summary>
        private async Task DFFaceBottomChangePaper()
        {

            while (true)
            {

                try
                {
                    decimal dfRemain = 0;
                    dfRemain = GlobalControl.dfFaceBottomSameMaterilaLeftMeter;//面底同材剩余米数-糊机到横切的距离
                    var curOrder = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();//当前正在生产的订单
                    List<string> curCode = new List<string>();
                    if (curOrder != null)
                    {
                        if (curOrder.WO_PaperCode.Contains("."))
                        {
                            curCode = curOrder.WO_PaperCode.Split('.').ToList();
                        }
                        else
                        {
                            curCode = curOrder.WO_PaperCode.ToCharArray().Select(a => a.ToString()).ToList();
                        }
                    }

                    if (dfRemain == 0 || dfRemain < -50 || dfRemain > 100)
                    {
                        continue;
                    }

                    if (dfRemain > 0 && dfRemain <= 30)
                    {
                        stateInfo_gu_face_bottom.GuRange1++;
                    }
                    else if (dfRemain > 30 && dfRemain <= 60)
                    {
                        stateInfo_gu_face_bottom.GuRange2++;
                    }
                    else if (dfRemain > 60 && dfRemain <= 90)
                    {
                        stateInfo_gu_face_bottom.GuRange3++;
                    }

                    if (dfRemain > 0 && dfRemain <= 90)
                    {
                        logger.Info($"DFFaceBottomChangePaper--dfRemain={dfRemain},GuRange1={stateInfo_gu_face_bottom.GuRange1},GuRange2={stateInfo_gu_face_bottom.GuRange2},GuRange3={stateInfo_gu_face_bottom.GuRange3}", moduleWarp);
                    }

                    if (stateInfo_gu_face_bottom.GuRange1 >= 5 || stateInfo_gu_face_bottom.GuRange2 >= 5 || stateInfo_gu_face_bottom.GuRange3 >= 5)
                    {
                        //List<string> PreSetInfo = GlobalControl.GetDictItems(DictTypesEnum.PreHQMeter.ToString());
                        //int preHQMeter = PreSetInfo[0].ToInt32();
                        if (dfRemain <= 0)
                        {
                            restWrapOrderID = (curOrder?.WO_ID) ?? 0;
                            restWrapOrderProductMeters = GlobalControl.curOrderProduct;

                            logger.Info($"DFFaceBottomChangePaper--已进入换材处理:糊机同材剩余-横切到糊机的距离={dfRemain}", moduleWarp);

                            // 恢复弯翘调平偏移量
                            RestCurvedWarp();

                            try
                            {
                                await GlobalInfos.SendMsg("M104", "REST");
                            }
                            catch
                            {

                            }

                            stateInfo_gu_face_bottom.GuRange1 = 0;
                            stateInfo_gu_face_bottom.GuRange2 = 0;
                            stateInfo_gu_face_bottom.GuRange3 = 0;

                            while (true)
                            {
                                try
                                {
                                    var curOrderNew = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();

                                    if (curOrderNew == null) continue;
                                    List<string> nowCode = new List<string>();

                                    if (curOrderNew.WO_PaperCode.Contains("."))
                                    {
                                        nowCode = curOrderNew.WO_PaperCode.Split('.').ToList();
                                    }
                                    else
                                    {
                                        nowCode = curOrderNew.WO_PaperCode.ToCharArray().Select(a => a.ToString()).ToList();
                                    }


                                    string curOrderMS = "";
                                    string paperMS = "";

                                    switch (nowCode.Count)
                                    {
                                        case 3:
                                            curOrderMS = nowCode[1];
                                            break;
                                        case 5:
                                            if (nowCode[1] != "-")
                                            {
                                                curOrderMS = nowCode[1];
                                            }
                                            else
                                            {
                                                curOrderMS = nowCode[3];
                                            }
                                            break;
                                        case 7:
                                            if (nowCode[1] != "-")
                                            {
                                                curOrderMS = nowCode[1];
                                            }
                                            else if (nowCode[3] != "-")
                                            {
                                                curOrderMS = nowCode[3];
                                            }
                                            else
                                            {
                                                curOrderMS = nowCode[5];
                                            }
                                            break;
                                        default:
                                            break;
                                    }

                                    switch (curCode.Count)
                                    {
                                        case 3:
                                            paperMS = curCode[1];
                                            break;
                                        case 5:
                                            if (curCode[1] != "-")
                                            {
                                                paperMS = curCode[1];
                                            }
                                            else
                                            {
                                                paperMS = curCode[3];
                                            }
                                            break;
                                        case 7:
                                            if (curCode[1] != "-")
                                            {
                                                paperMS = curCode[1];
                                            }
                                            else if (curCode[3] != "-")
                                            {
                                                paperMS = curCode[3];
                                            }
                                            else
                                            {
                                                paperMS = curCode[5];
                                            }
                                            break;
                                        default:
                                            break;
                                    }


                                    if (curCode.FirstOrDefault() == nowCode.FirstOrDefault()
                                        && curOrderMS == paperMS
                                        && curCode.FindLast(a => a != "-") == nowCode.FindLast(a => a != "-"))
                                    {
                                        continue;
                                    }
                                    else
                                    {
                                        break;
                                    }
                                }
                                catch (Exception)
                                {

                                }
                                finally
                                {
                                    //Thread.Sleep(1000);
                                    await Task.Delay(1000);
                                }
                            }
                        }

                    }
                }
                catch (Exception ex)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.AppendLine("DFFaceBottomChangePaper--监听糊机面底换材过程中发生异常：");
                    sb.AppendLine(ex.Message);
                    logger.Error(sb.ToString(), moduleWarp);
                }
                finally
                {
                    //Thread.Sleep(500);
                    await Task.Delay(500);
                }
            }
        }


        /// <summary>
        /// 换单检查是否换材质，换材质则构造弯翘数据
        /// </summary>
        /// <returns></returns>
        public async Task HQChangeOrder()
        {

            List<DictDataInfo> dictDataInfos = BLLFactory<DictDataInfoManager>.Instance.GetDictItemsToModelList(DictTypesEnum.WarpSet.ToString());
            var type = dictDataInfos.FirstOrDefault(it => it.PD_Property == "DetectionType")?.PD_Value?.Trim();

            // 有检测设备才去构造数据
            if (type != null && type != "1" && type != "2")
            {
                return;
            }

            //取当前首笔订单
            var curInfo = BLLFactory<OrderInfoManage>.Instance.GetFirstByWorkNo();
            if (curInfo == null)
            {
                return;
            }

            //取刚刚完工的订单
            var lastInfo = await BLLFactory<OrderFinishInfoManage>.Instance.AsQueryable().OrderByDescending(it => it.WOF_ID).FirstAsync();
            if (lastInfo == null)
            {
                return;
            }

            // 当前订单和上一笔完工订单的材质或楞型有变化 需要构造弯翘数据
            if (curInfo.WO_PaperCode == lastInfo.WOF_PaperCode
                && curInfo.WO_Wave == lastInfo.WOF_Wave
                && curInfo.WO_Width == lastInfo.WOF_Width
                )
            {
                return;
            }


            //取完工表前500条数据
            var lastInfos = await BLLFactory<OrderFinishInfoManage>.Instance.AsQueryable().OrderByDescending(it => it.WOF_ID).Take(500).ToListAsync();
            if (lastInfo == null)
            {
                return;
            }

            List<int> ints = new List<int>();
            foreach (var item in lastInfos)
            {
                if (item.WOF_PaperCode == lastInfo.WOF_PaperCode
                    && item.WOF_Wave == lastInfo.WOF_Wave)
                {
                    if (!ints.Contains(item.WO_ID))
                    {
                        ints.Add(item.WO_ID);
                    }
                }
                else
                {
                    break;
                }
            }

            if (ints.Count == 0) return;

            DateTime dtStart = DateTime.Now.AddDays(-1);
            DateTime dtEndTime = DateTime.Now;
            List<WarpHisRunPar> list = await BLLFactory<HisRunParDataManager>.Instance.AsQueryable().Where(a => ints.Contains(a.F_OrderID)).SplitTable(dtStart, dtEndTime).OrderBy(it => it.F_CreateTime)
                .Select(a => new WarpHisRunPar()
                {
                    F_ID = a.F_Id,
                    F_CreateTime = a.F_CreateTime,
                    F_Paper = a.F_Paper,
                    F_OrderID = a.F_OrderID,
                    F_CurRemain = a.F_CurRemain,
                    F_Shift = a.F_Shift,
                    F_Width = a.F_Width,
                    F_Wave = a.F_Wave,
                    F_Warp_DetectState = a.F_Warp_DetectState,
                }).ToListAsync();

            List<long> lastOrderIDs = (from a in list
                                       group a by a.F_OrderID into grpTemp
                                       select grpTemp.OrderByDescending(a => a.F_CreateTime).FirstOrDefault()).Where(a => a != null).Select(a => a.F_ID).ToList();

            list.RemoveAll(a => lastOrderIDs.Contains(a.F_ID));

            int OrderID = 0;
            string ShiftCode = "";
            string? DetectState = null;

            WarpAnalysisInfo warpAnalysisInfo = new WarpAnalysisInfo();
            List<WarpAnalysisInfo> warpAnalysisInfos = new List<WarpAnalysisInfo>();
            List<WarpHisRunPar> tempList = new List<WarpHisRunPar>();
            decimal orderStartReamin = 0;
            decimal orderProductMeter = 0;
            int index = 0;
            foreach (var item in list)
            {
                if (OrderID != item.F_OrderID || ShiftCode != item.F_Shift)
                {
                    if (OrderID > 0)
                    {

                        warpAnalysisInfo.F_ProductMeter = orderProductMeter;

                        warpAnalysisInfo.F_NormalMeter = orderProductMeter
                            - (warpAnalysisInfo.F_Up1Meter ?? 0)
                            - (warpAnalysisInfo.F_Up2Meter ?? 0)
                            - (warpAnalysisInfo.F_Up3Meter ?? 0)
                            - (warpAnalysisInfo.F_Down1Meter ?? 0)
                            - (warpAnalysisInfo.F_Down2Meter ?? 0)
                            - (warpAnalysisInfo.F_Down3Meter ?? 0);

                        if (warpAnalysisInfo.F_NormalMeter < 0)
                        {
                            warpAnalysisInfo.F_NormalMeter = 0;
                        }

                        warpAnalysisInfos.Add(ModelClone.Clone<WarpAnalysisInfo>(warpAnalysisInfo));
                        //BLLFactory<WarpAnalysisInfoManager>.Instance.Insert(warpAnalysisInfo);
                    }

                    orderProductMeter = item.F_CurRemain;
                    orderStartReamin = item.F_CurRemain;
                    warpAnalysisInfo = new WarpAnalysisInfo() { F_Width = item.F_Width, F_Wave = item.F_Wave, F_PaperCode = item.F_Paper, F_ShifCode = item.F_Shift, F_StartTime = item.F_CreateTime };
                    OrderID = item.F_OrderID;
                    ShiftCode = item.F_Shift;
                    tempList.Clear();
                    DetectState = item.F_Warp_DetectState;
                    tempList.Add(item);
                }
                else
                {
                    orderProductMeter = orderStartReamin - item.F_CurRemain;
                    warpAnalysisInfo.F_EndTime = item.F_CreateTime;

                    if (DetectState != item.F_Warp_DetectState)
                    {
                        tempList.Add(item);
                        if (tempList.Count >= 2)
                        {
                            if (DetectState == "上弯轻")
                            {
                                warpAnalysisInfo.F_Up1Meter = (warpAnalysisInfo.F_Up1Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }
                            else if (DetectState == "上弯中")
                            {
                                warpAnalysisInfo.F_Up2Meter = (warpAnalysisInfo.F_Up2Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }
                            else if (DetectState == "上弯重")
                            {
                                warpAnalysisInfo.F_Up3Meter = (warpAnalysisInfo.F_Up3Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }
                            else if (DetectState == "下弯轻")
                            {
                                warpAnalysisInfo.F_Down1Meter = (warpAnalysisInfo.F_Down1Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }
                            else if (DetectState == "下弯中")
                            {
                                warpAnalysisInfo.F_Down2Meter = (warpAnalysisInfo.F_Down2Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }
                            else if (DetectState == "下弯重")
                            {
                                warpAnalysisInfo.F_Down3Meter = (warpAnalysisInfo.F_Down3Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                            }

                            DetectState = item.F_Warp_DetectState;
                            tempList.Clear();
                            tempList.Add(item);


                        }
                        else
                        {
                            tempList.Clear();
                            tempList.Add(item);
                        }
                    }
                    else
                    {
                        tempList.Add(item);
                    }



                }

                // 最后一条一样的
                if (index + 1 == list.Count)
                {

                    if (tempList.Count >= 2)
                    {
                        if (DetectState == "上弯轻")
                        {
                            warpAnalysisInfo.F_Up1Meter = (warpAnalysisInfo.F_Up1Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                        else if (DetectState == "上弯中")
                        {
                            warpAnalysisInfo.F_Up2Meter = (warpAnalysisInfo.F_Up2Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                        else if (DetectState == "上弯重")
                        {
                            warpAnalysisInfo.F_Up3Meter = (warpAnalysisInfo.F_Up3Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                        else if (DetectState == "下弯轻")
                        {
                            warpAnalysisInfo.F_Down1Meter = (warpAnalysisInfo.F_Down1Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                        else if (DetectState == "下弯中")
                        {
                            warpAnalysisInfo.F_Down2Meter = (warpAnalysisInfo.F_Down2Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                        else if (DetectState == "下弯重")
                        {
                            warpAnalysisInfo.F_Down3Meter = (warpAnalysisInfo.F_Down3Meter ?? 0) + tempList.First().F_CurRemain - tempList.Last().F_CurRemain;
                        }
                    }

                    warpAnalysisInfo.F_ProductMeter = orderProductMeter;

                    warpAnalysisInfo.F_NormalMeter = orderProductMeter
                            - (warpAnalysisInfo.F_Up1Meter ?? 0)
                            - (warpAnalysisInfo.F_Up2Meter ?? 0)
                            - (warpAnalysisInfo.F_Up3Meter ?? 0)
                            - (warpAnalysisInfo.F_Down1Meter ?? 0)
                            - (warpAnalysisInfo.F_Down2Meter ?? 0)
                            - (warpAnalysisInfo.F_Down3Meter ?? 0);

                    if (warpAnalysisInfo.F_NormalMeter < 0)
                    {
                        warpAnalysisInfo.F_NormalMeter = 0;
                    }

                    //BLLFactory<WarpAnalysisInfoManager>.Instance.Insert(warpAnalysisInfo);
                    warpAnalysisInfos.Add(ModelClone.Clone<WarpAnalysisInfo>(warpAnalysisInfo));
                }

                index++;
            }

            if (warpAnalysisInfos.Count > 0)
            {
                var results = (from item in warpAnalysisInfos
                               group item by new { ShiftCode = item.F_ShifCode, Width = item.F_Width, PapeCode = item.F_PaperCode, Wave = item.F_Wave } into grpTemp
                               select new WarpAnalysisInfo
                               {
                                   F_ShifCode = grpTemp.Key.ShiftCode,
                                   F_NormalMeter = grpTemp.Sum(a => a.F_NormalMeter),
                                   F_Down1Meter = grpTemp.Sum(a => a.F_Down1Meter),
                                   F_Down2Meter = grpTemp.Sum(a => a.F_Down2Meter),
                                   F_Down3Meter = grpTemp.Sum(a => a.F_Down3Meter),
                                   F_EndTime = grpTemp.Max(a => a.F_EndTime),
                                   F_StartTime = grpTemp.Min(a => a.F_StartTime),
                                   F_PaperCode = grpTemp.Key.PapeCode,
                                   F_ProductMeter = grpTemp.Sum(a => a.F_ProductMeter),
                                   F_Wave = grpTemp.Key.Wave,
                                   F_Width = grpTemp.Key.Width,
                                   F_Up1Meter = grpTemp.Sum(a => a.F_Up1Meter),
                                   F_Up2Meter = grpTemp.Sum(a => a.F_Up2Meter),
                                   F_Up3Meter = grpTemp.Sum(a => a.F_Up3Meter),
                               }).ToList();

                await BLLFactory<WarpAnalysisInfoManager>.Instance.InsertRangeAsync(results);

            }


        }
        #endregion

        #region 事件

        public event EventHandler OnWarpPublish;

        /// <summary>
        /// 弯翘调整立即赋值
        /// </summary>
        /// <param name="msg"></param>
        public void WarpPublish(IpsDriverPositionEnum position)
        {
            // 触发事件，通知所有订阅者
            if (OnWarpPublish != null)
                OnWarpPublish(position, EventArgs.Empty);
        }

        #endregion
    }


    #region 类对象

    public class DetectionStatus
    {
        /// <summary>
        /// 程序判定的状态
        /// </summary>
        public string JudgeWarpStatu { get; set; } = "";

        /// <summary>
        /// 程序判定的状态
        /// </summary>
        public string JudgeWarpStatuCmd { get; set; } = "";


        public decimal Degree { get; set; }

    }


    public class PositonValue
    {
        public WarpPositionEnum F_Position { get; set; }
        public decimal F_Value { get; set; } = 0;
    }



    public class SpeedInfo
    {
        /// <summary>
        /// 记录时间
        /// </summary>
        public DateTime RecordTime { get; set; }

        /// <summary>
        /// 速度
        /// </summary>
        public int Speed { get; set; }
    }


    public class WarpHisRunPar
    {
        public long F_ID { get; set; }

        public DateTime F_CreateTime { get; set; }

        public string F_Paper { get; set; }

        public int F_OrderID { get; set; }

        public decimal F_CurRemain { get; set; }

        public string F_Shift { get; set; }

        public int F_Width { get; set; }

        public string F_Wave { get; set; }

        public string F_Warp_DetectState { get; set; }


    }

    #endregion


}
