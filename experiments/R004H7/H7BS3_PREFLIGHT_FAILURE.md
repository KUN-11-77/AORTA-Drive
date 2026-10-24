# R004H7-BS3 preflight failure

原 `2664756` 实现错误地从 `r004h_group_roles.csv` 查找不存在的 `development` role；该
manifest只定义fit、selection与proxy_cal角色。程序在读取任何trajectory artifact之前
因组数不是24而fail closed，未产生`result.json`，也未读取field、teacher或outcome。

R1只修正身份清单接口：改用此前H7BS1已冻结的
`r004_recording_group_split.csv`及其`split=development` 24 groups，并新增真实manifest
单测。ROI、车辆、时域、全部候选支持规则、门槛和禁止项不变。原空输出目录不得复用。
