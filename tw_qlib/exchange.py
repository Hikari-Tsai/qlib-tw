"""台股交易所。

Qlib 的浮點 limit_threshold（例如 0.1）以 `abs($change) >= 0.1` 判斷漲跌停，
但台股漲跌停價會依升降單位向內取整，實際漲停多半只有 +9.9x%，而 2015-06-01 前的限制是 ±7%，
用浮點門檻幾乎都判斷不到。這裡改用資料中的 `$limit_up`／`$limit_down` 旗標（收在漲停／跌停價為 1）。

Qlib 的 YAML 以 safe 模式載入，無法寫 tuple，而 Exchange 只接受 tuple 形式的運算式門檻，
所以這個子類別把 list 轉成 tuple。
"""
from qlib.backtest.exchange import Exchange


class TWExchange(Exchange):
    def __init__(self, *args, limit_threshold=("$limit_up", "$limit_down"), **kwargs):
        if isinstance(limit_threshold, list):
            limit_threshold = tuple(limit_threshold)
        super().__init__(*args, limit_threshold=limit_threshold, **kwargs)
