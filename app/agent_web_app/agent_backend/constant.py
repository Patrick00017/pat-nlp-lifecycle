
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

LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

FIXED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_material_change_in_log",
            "description": "Queries the material change log within a specified time range.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "start_time": {
                        "type": "STRING",
                        "description": "Start time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "end_time": {
                        "type": "STRING",
                        "description": "End time in YYYY-MM-DD HH:MM:SS.mmm"
                    }
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_glue_set_func_call_in_log",
            "description": "Queries whether the glue setting function was called for a specific material at a given time.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "time": {
                        "type": "STRING",
                        "description": "The specific time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "desire_material": {
                        "type": "STRING",
                        "description": "The material identifier to check"
                    }
                },
                "required": ["time", "desire_material"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_material_in_log",
            "description": "Tracks the lifecycle and movement of a specific material within a time range.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "start_time": {
                        "type": "STRING",
                        "description": "Start time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "end_time": {
                        "type": "STRING",
                        "description": "End time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "material": {
                        "type": "STRING",
                        "description": "The material identifier to track"
                    }
                },
                "required": ["start_time", "end_time", "material"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pressroll_mp_set_func_call_in_log",
            "description": "Queries MP pressure roller setting function calls for a specific material within a time range.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "start_time": {
                        "type": "STRING",
                        "description": "Start time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "end_time": {
                        "type": "STRING",
                        "description": "End time in YYYY-MM-DD HH:MM:SS.mmm"
                    },
                    "desire_material": {
                        "type": "STRING",
                        "description": "The material identifier to check"
                    }
                },
                "required": ["start_time", "end_time", "desire_material"]
            }
        }
    }
]

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