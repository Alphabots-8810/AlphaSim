# AlphaSim — FRC 2026 REBUILT Shooter Aim

弹道物理 + 优化管线,为 FRC 2026 big-dumper 风格 shooter 求解
**距离 → hood 角度 + flywheel 线速度**。最终交付是一个闭式函数,机器人代码里
输入到 HUB 的距离即可得到 hood 角和飞轮转速,比赛现场只调**一个系数** `kSpeed` 校准落点。

> 出球高度按 **0.50 m** 标定(V3.1:centered-landing + 物理校准 cost,
> `w_tof=1.1` 由 vortex-shedding 散布推导)。

**🔗 在线**(GitHub Pages):报告与 Alpha Sim 模拟器 → **https://alphabots-8810.github.io/AlphaSim/**

## 直接拿来用

最省事:复制 [`handoff/ShooterAim.java`](handoff/ShooterAim.java)(或 [`shooter_aim.py`](handoff/shooter_aim.py))进机器人项目。

```java
ShooterAim.Aim a = ShooterAim.aimFor(distanceMeters);
setHoodAngle(a.hoodAngleDeg());            // 度
setFlywheelSurfaceSpeed(a.flywheelSpeedMps());   // m/s 线速度(转 RPM: v/(π·D)·60)
```

有效距离 1.5–5.0 m(超出自动 clamp)。现场标定只动 `kSpeed`:打短了调大、打远了调小。
完整交付说明见 [`handoff/README.md`](handoff/README.md)。

## 文档与工具

- **研究报告** [`research_report.html`](research_report.html) — 物理模型 → 算法演化(V1→V4r2)→ 公式/系数怎么算出来的(§5)。
- **Alpha Sim** [`shooter_sim.html`](shooter_sim.html) — 浏览器交互式模拟器,拖 slider 看 100 球 burst 命中率。
- **系数表** [`lookup_polynomial.json`](lookup_polynomial.json) — 机器可读的拟合系数(hood / v_ball / flywheel 线速度 / 参考 RPM)。

## 跑源码

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/aim.py -d 3.5                      # 单点求解
python scripts/aim.py -d 0 --sweep                # 全距离 sweep
python scripts/robust_aim.py -d 3.5 --compare     # 鲁棒 vs 启发式
python scripts/generate_coefficients.py --shooter-height 0.50 --json lookup_polynomial.json
pytest -q                                         # 28 tests
```

## 结构

```
src/        physics(trajectory/shooter) · geometry(target) · optimizer(solve/robust) · viz(plot)
scripts/    aim.py · robust_aim.py · hood_sweep_hitrate.py · generate_coefficients.py
config/     game_2026.yaml(HUB/FUEL 参数) · robot.yaml(机器人几何/cost 权重)
handoff/    交付给 shooter 程序的精简包(函数 + 报告 + sim + json + readme)
tests/      28 pytest
```

## 换出球高度

系数在 0.50 m 下烘焙进多项式。真车高度若明显变化,重算后粘回 `ShooterAim.java`:

```bash
python scripts/generate_coefficients.py --shooter-height <新高度>
```

## License

MIT — see [LICENSE](LICENSE). © 2026 Alphabots — FRC Team 8810.

---
FRC Team 8810 · Alphabots
