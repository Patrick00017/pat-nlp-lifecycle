using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BTS.Server.IPS
{
    public interface IWarpCtrl
    {
        public Task HQChangeOrder();

        public void RestCurvedWarp(bool isAutoExeRest);

        public void HandleCurvedWarp(string message);


        public int execWrapOrderID { get; set; }

        public decimal execWrapOrderProductMeters { get; set; }
    }
}
