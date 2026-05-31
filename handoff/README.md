# FRC 2026 REBUILT — Shooter Aim 算法交付包

输入到 HUB 的水平距离 → 输出 **hood 角度** 和 **flywheel 表面线速度**。
现场只调**一个系数** `kSpeed` 校准落点,不必逐参数重调。

算法来自 *Alpha Sim* 弹道优化管线(V3.1:centered-landing + 物理校准 cost,
`w_tof=1.1` 由 vortex-shedding 散布推导)。推导全过程见 `research_report.html`。

---

## 文件

| 文件 | 说明 |
|---|---|
| `ShooterAim.java` | 交付函数(WPILib / Java 17)。直接进机器人代码。 |
| `shooter_aim.py` | 同一套系数的 Python 版(RobotPy 或离线验证)。 |
| `research_report.html` | 完整研究报告:物理模型 → 算法演化 → **公式与系数怎么算出来的**(§5)。 |
| `shooter_sim.html` | Alpha Sim 交互式模拟器,浏览器直接打开,拖 slider 看命中率。 |
| `lookup_polynomial.json` | 同一套系数的机器可读副本(hood / v_ball / flywheel 线速度 / 参考 RPM)。 |

---

## 快速开始

把 `ShooterAim.java` 丢进机器人项目,然后:

```java
double d = distanceToHubMeters();          // 你的测距(视觉/里程计)
ShooterAim.Aim a = ShooterAim.aimFor(d);
setHoodAngle(a.hoodAngleDeg());            // 度,90° = 垂直向下基准
setFlywheelSurfaceSpeed(a.flywheelSpeedMps());   // m/s,线速度
```

Python:

```python
from shooter_aim import aim
hood_deg, flywheel_mps = aim(d)
```

**有效距离 1.5–5.0 m**;超出会自动 clamp 到边界(不外推,因为 hood 是 U 形曲线,过 5 m 会拐回上升)。

---

## 唯一的现场旋钮:`kSpeed`

`flywheel 线速度 = kSpeed × 拟合曲线(distance)`,默认 `kSpeed = 1.0`。

- 球**打短了** → `kSpeed` 调大(如 1.03)
- 球**打远了** → `kSpeed` 调小(如 0.97)

这一个数把**轮–球打滑、球被压缩、电池电压、空气模型误差**全吸收进去。
**hood 角不给旋钮**——它走拟合曲线,精度 ±0.1°,远小于 servo/装配误差。
现场校准 = 打几发 → 微调 `kSpeed` → 落点居中,**只动一个参数**。

---

## 线速度 ↔ RPM

主输出是**表面线速度 m/s**,因为它**与轮径无关**(你们的新机器人轮径可能不同)。
转 RPM 用自己的轮径 `D`(m):

```
rpm = v / (Math.PI * D) * 60
```

报告和 sim 里出现的 RPM 数字,是本项目 0.1016 m(4")轮径下的等价参考值
(例 `d=3 m`:8.70 m/s ↔ 1636 RPM),两套数并不矛盾。

Alpha Sim 显示的 `v_exit`(球出射速度,如 `d=3 m` 7.14 m/s)是函数 flywheel 线速度的 **×(1−0.18)**:
`v_exit = flywheel线速度 × 0.82`,即 7.14 = 8.70 × 0.82。函数输出的是轮缘速度(含打滑),sim 显示的是球速,同一物理量两端而已。

---

## 出球高度(shooter height)

系数是在**出球高度 0.50 m**(8810 实测)下生成的,这个高度**已经烘焙进多项式系数里了**——
代码里的 `SHOOTER_HEIGHT_M = 0.50` 只是出处标注,运行时改它**不起作用**。

真车量出来的高度若明显不同,需要**重算系数**——这一步在完整的 `frc-shooter-sim` 仿真仓库里跑
(本包不含 `src/`,故不附带可运行的生成器,以免误以为能直接跑):

```bash
# 在 frc-shooter-sim 仓库根目录:
python scripts/generate_coefficients.py --shooter-height 0.55
```

它会打印新的 a/b/c,直接粘回 `ShooterAim.java` / `shooter_aim.py` 两个方法即可。
找 8810 要这个仓库,或直接报高度让他们生成。

---

## 报告渲染提示

`research_report.html` 用 MathJax(CDN)渲染公式,**首次打开需联网**;否则公式会显示成
LaTeX 源码(内容仍在,只是不好看)。已移除原先的 `polyfill.io`(那个 CDN 2024 年被投毒,
且 MathJax 3 不需要它)。

---

*算法/报告/模拟器作者署名见报告页脚。问题直接找 8810。*
