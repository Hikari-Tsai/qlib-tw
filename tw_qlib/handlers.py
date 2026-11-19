"""台股自訂資料處理器：Alpha158 再加上三大法人與融資融券特徵（欄位由 build_flows.py 產生）。"""
from __future__ import annotations

from qlib.contrib.data.handler import Alpha158

# 股數欄位都已除以 factor，和 $volume 同單位，所以比值可以跨股票比較
FLOW_FEATURES = {
    **{f"FOR{n}": f"Sum($foreign_net, {n})/(Sum($volume, {n})+1)" for n in (5, 20, 60)},
    **{f"TRU{n}": f"Sum($trust_net, {n})/(Sum($volume, {n})+1)" for n in (5, 20, 60)},
    **{f"DEA{n}": f"Sum($dealer_net, {n})/(Sum($volume, {n})+1)" for n in (5, 20)},
    "FORPOS20": "Mean($foreign_net>0, 20)",  # 外資近 20 日買超天數比例
    "TRUPOS20": "Mean($trust_net>0, 20)",
    "MRGDAYS": "$margin_bal/(Mean($volume, 20)+1)",  # 融資餘額 ÷ 20 日均量
    "MRGCHG5": "$margin_bal/(Ref($margin_bal, 5)+1)-1",
    "MRGCHG20": "$margin_bal/(Ref($margin_bal, 20)+1)-1",
    "SHTDAYS": "$short_bal/(Mean($volume, 20)+1)",
    "SHTMRG": "$short_bal/($margin_bal+1)",  # 券資比
}


class Alpha158Flow(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()
        return fields + list(FLOW_FEATURES.values()), names + list(FLOW_FEATURES.keys())
