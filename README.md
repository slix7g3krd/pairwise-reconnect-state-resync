# pairwise-reconnect-state-resync

## 领域

游戏、图形与多媒体

## 主题

断线重连状态重建

## 说明

超窗全量、握手等资源与会话去重。

初始状态下 `gameplay/reconnect_state_resync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
