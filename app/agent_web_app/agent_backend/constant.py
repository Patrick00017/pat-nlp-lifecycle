
handle_func_to_splicer_part = {
    'HandleChangeRollLS0': 'ls0',
    'HandleChangePaperLS0': 'ls0',
    'HandleChangeRollLS1': 'ls1',
    'HandleChangePaperLS1': 'ls1',
    'HandleChangeRollMS1': 'ms1',
    'HandleChangePaperMS1': 'ms1',
    'HandleChangeRollLS2': 'ls2',
    'HandleChangePaperLS2': 'ls2',
    'HandleChangeRollMS2': 'ms2',
    'HandleChangePaperMS2': 'ms2',

}

module_types = [
    "BTS.Server.Start.CommHelper",
    "BTS.Server.Start.IPSBizs.AlarmBiz",
    "BTS.Server.Start.IPSBizs.ChangePaperBizModel",
    "BTS.Server.Start.IPSBizs.CruiseCtrl",
    "BTS.Server.Start.IPSBizs.HotSprayCtrl",
    "BTS.Server.Start.IPSBizs.IpsChangePaper",
    "BTS.Server.Start.IPSBizs.NewCtrl.BridgeTensionCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.ColdPlatePressCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.CorrugatedRollCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.GlueCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.HotLoadGroupQtyCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.HotPlatePressCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.IPSMainCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.PressRollMPCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.SPTensionCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.VacuumBlowerCtrl",
    "BTS.Server.Start.IPSBizs.NewCtrl.WrapCtrl",
    "BTS.Server.Start.IPSBizs.RidingRollCtrl",
    "BTS.Server.Start.IPSBizs.SteamBiz",
    "BTS.Server.Start.IpsChangeOrder",
    "BTS.Server.Start.IpsSetValue", # old
    "BTS.Server.Start.PmsCCCSend",
    "BTS.Server.Start.PMSDataService",
    "BTS.Service.WebApi.Attributes.ApiResultAttribute",
    "BTS.Service.WebApi.Models.GlobalVars",
    "BTS.Services.DomainServices.ErpOrderHelper",
]