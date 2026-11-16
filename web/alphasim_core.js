/* AlphaSim core - the same physics and aim solver as the Python package, in
 * plain JavaScript so the simulator can solve live in the browser.
 *
 * This file is a line-for-line port of src/physics/batch.py, src/scoring.py
 * and src/optimizer/maxprob.py. tests/test_js_parity.py runs both and fails
 * if they disagree, so change them together.
 *
 * No DOM access in here: it loads in a browser (window.AlphaCore) and in Node
 * (module.exports) alike.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AlphaCore = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---- special functions ---------------------------------------------------

  const SQRT_PI = Math.sqrt(Math.PI);

  /** Complementary error function, ~1e-13 relative accuracy including the tails. */
  function erfc(x) {
    if (x < 0) return 2 - erfc(-x);
    if (x < 2.5) {
      // erf(x) = 2/sqrt(pi) * exp(-x^2) * sum 2^n x^(2n+1) / (1*3*...*(2n+1))
      let term = x, sum = x;
      for (let n = 1; n < 200; n++) {
        term *= 2 * x * x / (2 * n + 1);
        sum += term;
        if (term < sum * 1e-17) break;
      }
      return 1 - 2 / SQRT_PI * Math.exp(-x * x) * sum;
    }
    // Continued fraction, evaluated from the tail inward.
    let k = x;
    for (let n = 80; n >= 1; n--) k = x + (n / 2) / k;
    return Math.exp(-x * x) / (SQRT_PI * k);
  }

  /** Standard normal CDF. */
  function ndtr(x) { return 0.5 * erfc(-x / Math.SQRT2); }

  /** Inverse standard normal CDF (Acklam's rational fit + one Halley step). */
  function ndtri(p) {
    if (!(p > 0)) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
               3.754408661907416e+00];
    const tail = (q) =>
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    let x;
    if (p < 0.02425) x = tail(Math.sqrt(-2 * Math.log(p)));
    else if (p > 1 - 0.02425) x = -tail(Math.sqrt(-2 * Math.log(1 - p)));
    else {
      const q = p - 0.5, r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
          (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    // Halley refinement against the accurate CDF. In the far lower tail work
    // with the relative error so the correction keeps its precision.
    const cdf = ndtr(x);
    if (cdf > 0 && Number.isFinite(x)) {
      const e = cdf - p;
      const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
      x = x - u / (1 + x * u / 2);
    }
    return x;
  }

  // ---- ballistics ------------------------------------------------------------

  /** Per-unit-mass aero constants, as Aero.from_ball. */
  function aeroFromBall(massKg, radiusM, cd, cl, rho, gravity) {
    const k = 0.5 * (rho === undefined ? 1.225 : rho) * Math.PI * radiusM * radiusM / massKg;
    return { kd: k * cd, kl: k * (cl || 0), g: gravity === undefined ? 9.81 : gravity };
  }

  function hermite(s, p0, p1, d0, d1, h) {
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * d0 +
           (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * d1;
  }

  function hermiteD(s, p0, p1, d0, d1, h) {
    const s2 = s * s;
    return (6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * h * d0 +
           (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * h * d1;
  }

  /**
   * Fly one ball until it comes down through the target plane.
   * Fixed-step RK4 with a cubic-Hermite crossing, exactly as simulate_batch.
   * If `path` is an array, [t, x, y] samples are appended for drawing.
   */
  function fly(v, thetaDeg, exitHeight, targetHeight, aero, dt, tMax, path) {
    const h = dt === undefined ? 0.005 : dt;
    const nSteps = Math.ceil((tMax === undefined ? 6.0 : tMax) / h);
    const kd = aero.kd, kl = aero.kl, g = aero.g, H = targetHeight;
    const th = thetaDeg * Math.PI / 180;
    let x = 0, y = exitHeight, vx = v * Math.cos(th), vy = v * Math.sin(th), t = 0;
    const out = { hit: false, x: NaN, t: NaN, vx: NaN, vy: NaN, apexX: 0, apexY: exitHeight };
    if (path) path.push([0, x, y]);
    for (let i = 0; i < nSteps; i++) {
      let sp = Math.hypot(vx, vy);
      const k1x = -kd * sp * vx - kl * sp * vy, k1y = -g - kd * sp * vy + kl * sp * vx;
      let ux = vx + 0.5 * h * k1x, uy = vy + 0.5 * h * k1y;
      sp = Math.hypot(ux, uy);
      const k2x = -kd * sp * ux - kl * sp * uy, k2y = -g - kd * sp * uy + kl * sp * ux;
      ux = vx + 0.5 * h * k2x; uy = vy + 0.5 * h * k2y;
      sp = Math.hypot(ux, uy);
      const k3x = -kd * sp * ux - kl * sp * uy, k3y = -g - kd * sp * uy + kl * sp * ux;
      ux = vx + h * k3x; uy = vy + h * k3y;
      sp = Math.hypot(ux, uy);
      const k4x = -kd * sp * ux - kl * sp * uy, k4y = -g - kd * sp * uy + kl * sp * ux;
      const vx1 = vx + h / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x);
      const vy1 = vy + h / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y);
      const x1 = x + h * vx + h * h / 6.0 * (k1x + k2x + k3x);
      const y1 = y + h * vy + h * h / 6.0 * (k1y + k2y + k3y);

      if (vy > 0 && vy1 <= 0) {
        const s = vy / (vy - vy1);
        out.apexY = hermite(s, y, y1, vy, vy1, h);
        out.apexX = hermite(s, x, x1, vx, vx1, h);
      }
      if (y > H && y1 <= H) {
        let s = (y - H) / (y - y1);
        for (let n = 0; n < 4; n++) {
          const f = hermite(s, y, y1, vy, vy1, h) - H;
          let df = hermiteD(s, y, y1, vy, vy1, h);
          if (df === 0) df = -1e-12;
          s = Math.min(1, Math.max(0, s - f / df));
        }
        out.hit = true;
        out.t = t + s * h;
        out.x = hermite(s, x, x1, vx, vx1, h);
        out.vx = hermiteD(s, x, x1, vx, vx1, h) / h;
        out.vy = hermiteD(s, y, y1, vy, vy1, h) / h;
        if (path) path.push([out.t, out.x, H]);
        return out;
      }
      if (vy1 < 0 && y1 < H) {            // falling below the plane: can never score
        if (path) {
          // Keep drawing the miss down to the floor so it reads as a miss.
          let px = x1, py = y1, pvx = vx1, pvy = vy1, pt = t + h;
          path.push([pt, px, py]);
          for (let n = 0; n < 2000 && py > 0; n++) {
            const s2 = Math.hypot(pvx, pvy);
            pvx += h * (-kd * s2 * pvx - kl * s2 * pvy);
            pvy += h * (-g - kd * s2 * pvy + kl * s2 * pvx);
            px += h * pvx; py += h * pvy; pt += h;
            path.push([pt, px, Math.max(py, 0)]);
          }
        }
        return out;
      }
      x = x1; y = y1; vx = vx1; vy = vy1; t += h;
      if (path) path.push([t, x, y]);
    }
    return out;
  }

  function entryAngleDeg(r) { return Math.atan2(-r.vy, Math.abs(r.vx)) * 180 / Math.PI; }

  /** Closed-form launch speed with no air; NaN if the angle is too flat. */
  function noDragSpeed(distance, thetaDeg, exitHeight, targetHeight, gravity) {
    const th = thetaDeg * Math.PI / 180;
    const rise = distance * Math.tan(th) - (targetHeight - exitHeight);
    if (!(rise > 0)) return NaN;
    return Math.sqrt(0.5 * gravity * distance * distance / (Math.cos(th) * Math.cos(th) * rise));
  }

  /** Launch speed that lands the ball `distance` away; NaN if impossible. */
  function speedForDistance(distance, thetaDeg, exitHeight, targetHeight, aero, vMax, dt) {
    vMax = vMax === undefined ? 40.0 : vMax;
    dt = dt === undefined ? 0.005 : dt;
    const xtol = 1e-9;
    const miss = (v) => {
      const r = fly(v, thetaDeg, exitHeight, targetHeight, aero, dt);
      return r.hit ? r.x - distance : -distance;
    };
    const seed = noDragSpeed(distance, thetaDeg, exitHeight, targetHeight, aero.g);
    if (!Number.isFinite(seed) || !(seed < vMax)) return NaN;
    let lo = Math.max(0.6 * seed, 0.3), hi = Math.min(1.25 * seed, vMax);
    let fLo = miss(lo), fHi = miss(hi);
    for (let i = 0; i < 12; i++) {
      const grow = fHi < 0 && hi < vMax, shrink = fLo > 0 && lo > 0.3;
      if (!(grow || shrink)) break;
      if (grow) { hi = Math.min(hi * 1.3, vMax); fHi = miss(hi); }
      if (shrink) { lo = Math.max(lo * 0.7, 0.3); fLo = miss(lo); }
    }
    if (!(fLo <= 0 && fHi >= 0)) return NaN;

    let side = 0, bisect = false;
    for (let i = 0; i < 100; i++) {
      const width = hi - lo, denom = fHi - fLo;
      let mid = (bisect || denom === 0) ? 0.5 * (lo + hi) : (lo * fHi - hi * fLo) / denom;
      mid = Math.min(hi - 1e-3 * width, Math.max(lo + 1e-3 * width, mid));
      const fMid = miss(mid);
      const neg = fMid < 0;
      if (neg && side === -1) fHi *= 0.5;       // Illinois
      if (!neg && side === 1) fLo *= 0.5;
      if (neg) { lo = mid; fLo = fMid; } else { hi = mid; fHi = fMid; }
      side = neg ? -1 : 1;
      bisect = (hi - lo) > 0.5 * width;

      const exact = Math.abs(fMid) < 1e-10;
      const jump = (hi - lo) < 1e-4 && Math.min(Math.abs(fLo), Math.abs(fHi)) > 1e-2;
      if ((hi - lo) < xtol || exact || jump) {
        const vDone = exact ? mid : 0.5 * (lo + hi);
        const fDone = exact ? fMid : (Math.abs(fLo) < Math.abs(fHi) ? fLo : fHi);
        return Math.abs(fDone) < 1e-4 ? vDone : NaN;
      }
    }
    return NaN;
  }

  // ---- opening geometry ------------------------------------------------------

  /** [lo, hi] along-track positions that clear every rim edge (lo > hi: none). */
  function alongTrackLimits(opening, lateral, entryDeg) {
    const aLong = opening.ballRadius / Math.sin(entryDeg * Math.PI / 180);
    let lo = -Infinity, hi = Infinity, blocked = false;
    for (let k = 0; k < 6; k++) {
      const ang = (opening.bearingDeg + 60 * k) * Math.PI / 180;
      const nx = Math.cos(ang), nz = Math.sin(ang);
      const reach = Math.hypot(aLong * nx, opening.ballRadius * nz);
      const room = opening.apothem - nz * lateral - reach;
      if (Math.abs(nx) < 1e-12) { if (room < 0) blocked = true; }
      else if (nx > 0) hi = Math.min(hi, room / nx);
      else lo = Math.max(lo, room / nx);
    }
    return blocked ? [Infinity, -Infinity] : [lo, hi];
  }

  function halfWidth(opening, entryDeg) {
    return opening.apothem - opening.ballRadius / Math.sin(entryDeg * Math.PI / 180);
  }

  function clears(opening, along, lateral, entryDeg) {
    const lim = alongTrackLimits(opening, lateral, entryDeg);
    return along >= lim[0] && along <= lim[1];
  }

  // ---- error budget -> scoring probability -----------------------------------

  const GL_X = [-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564984,
                0.18343464249564984, 0.525532409916329, 0.7966664774136267, 0.9602898564975362];
  const GL_W = [0.05061426814518826, 0.11119051722668723, 0.15685332293894358, 0.1813418916891809,
                0.1813418916891809, 0.15685332293894358, 0.11119051722668723, 0.05061426814518826];
  const GH_X = [-4.512745863399783, -3.20542900285647, -2.07684797867783, -1.0232556637891326, 0.0,
                1.0232556637891326, 2.07684797867783, 3.20542900285647, 4.512745863399783];
  const GH_W = [2.23458440077466e-05, 0.002789141321231769, 0.049916406765217865, 0.24409750289493945,
                0.4063492063492063, 0.24409750289493945, 0.049916406765217865, 0.002789141321231769,
                2.23458440077466e-05];

  /** Landing spread (1 sigma, m) by source. Budget keys match ErrorBudget. */
  function landingSpread(budget, dxDlnv, dxDthetaDeg, tof, distance) {
    const speed = Math.abs(dxDlnv) * budget.speed_sigma;
    const angle = Math.abs(dxDthetaDeg) * budget.angle_sigma_deg;
    const vortex = budget.vortex_coeff * tof * tof;
    const range = budget.range_sigma_m;
    const yaw = distance * (budget.yaw_sigma_deg * Math.PI / 180);
    return {
      speed, angle, vortex, range, yaw,
      along: Math.sqrt(speed * speed + angle * angle + vortex * vortex + range * range),
      lateral: Math.sqrt(yaw * yaw + vortex * vortex),
    };
  }

  function missProbability(opening, budget, spread, entryDeg, alongBias) {
    const bias = alongBias || 0, s = spread.along;
    let total = 0;
    for (let i = 0; i < GL_X.length; i++) {
      for (let j = 0; j < GH_X.length; j++) {
        const z = budget.lateral_half_width_m * GL_X[i] + spread.lateral * GH_X[j];
        const lim = alongTrackLimits(opening, z, entryDeg), lo = lim[0], hi = lim[1];
        let tails;
        if (!(hi > lo)) tails = 1;
        else if (s > 0) tails = Math.min(1, Math.max(0, ndtr((lo - bias) / s) + ndtr((bias - hi) / s)));
        else tails = (lo <= bias && bias <= hi) ? 0 : 1;
        total += tails * GL_W[i] * GH_W[j];
      }
    }
    return total;
  }

  function sigmaMargin(miss) {
    return -ndtri(0.5 * Math.min(1, Math.max(1e-300, miss)));
  }

  // ---- aim solver --------------------------------------------------------------

  /** The centred-landing shot at one (distance, angle), fully graded. */
  function evalCentered(model, budget, distance, thetaDeg, dt) {
    dt = dt === undefined ? 0.01 : dt;
    const h0 = model.exitHeight, ht = model.targetHeight, aero = model.aero;
    const bad = { ok: false, theta: thetaDeg, v: NaN, margin: NaN, miss: NaN };
    const v = speedForDistance(distance, thetaDeg, h0, ht, aero, model.vMax, dt);
    if (!Number.isFinite(v)) return bad;
    const nominal = fly(v, thetaDeg, h0, ht, aero, dt);
    const ev = 2e-3, et = 0.05;
    const xVp = fly(v * (1 + ev), thetaDeg, h0, ht, aero, dt).x;
    const xVm = fly(v * (1 - ev), thetaDeg, h0, ht, aero, dt).x;
    const xTp = fly(v, thetaDeg + et, h0, ht, aero, dt).x;
    const xTm = fly(v, thetaDeg - et, h0, ht, aero, dt).x;
    const dxDlnv = (xVp - xVm) / (2 * ev), dxDtheta = (xTp - xTm) / (2 * et);
    if (!Number.isFinite(dxDlnv) || !Number.isFinite(dxDtheta)) return bad;
    const entry = entryAngleDeg(nominal);
    const spread = landingSpread(budget, dxDlnv, dxDtheta, nominal.t, distance);
    const miss = missProbability(model.opening, budget, spread, entry);
    return {
      ok: true, theta: thetaDeg, v, tof: nominal.t, entry,
      entrySpeed: Math.hypot(nominal.vx, nominal.vy), apex: nominal.apexY, apexX: nominal.apexX,
      dxDlnv, dxDtheta, halfWidth: halfWidth(model.opening, entry), spread, miss,
      pScore: 1 - miss, margin: sigmaMargin(miss),
    };
  }

  function parabolicPeak(x, y, i) {
    if (i <= 0 || i >= x.length - 1 ||
        !Number.isFinite(y[i - 1]) || !Number.isFinite(y[i]) || !Number.isFinite(y[i + 1])) {
      return [x[i], y[i]];
    }
    const x0 = x[i - 1], x1 = x[i], x2 = x[i + 1], y0 = y[i - 1], y1 = y[i], y2 = y[i + 1];
    const denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    const a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    const b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if (a === 0) return [x1, y1];
    const xv = -b / (2 * a);
    if (!(x0 <= xv && xv <= x2)) return [x1, y1];
    const c = y1 - a * x1 * x1 - b * x1;
    return [xv, a * xv * xv + b * xv + c];
  }

  /** Pick the launch angle from one distance's family (see choose_angle). */
  function chooseAngle(theta, speed, margin, concession) {
    const th = [], v = [], m = [];
    for (let i = 0; i < theta.length; i++) {
      if (Number.isFinite(margin[i]) && Number.isFinite(speed[i])) {
        th.push(theta[i]); v.push(speed[i]); m.push(margin[i]);
      }
    }
    if (!th.length) return null;
    let iBest = 0, iSlow = 0;
    for (let i = 1; i < th.length; i++) {
      if (m[i] > m[iBest]) iBest = i;
      if (v[i] < v[iSlow]) iSlow = i;
    }
    const best = parabolicPeak(th, m, iBest), slow = parabolicPeak(th, v, iSlow);
    const out = { best: best[0], bestMargin: best[1], slow: slow[0], slowSpeed: slow[1],
                  lo: best[0], hi: best[0], floor: best[1] };
    if (!(concession > 0) || !Number.isFinite(best[1])) { out.pick = best[0]; return out; }
    const floor = (1 - concession) * best[1];
    const edge = (step) => {
      let j = iBest;
      while (j + step >= 0 && j + step < th.length && m[j + step] >= floor) j += step;
      const k = j + step;
      if (k < 0 || k >= th.length || m[j] === m[k]) return th[j];
      return th[j] + (floor - m[j]) * (th[k] - th[j]) / (m[k] - m[j]);
    };
    out.lo = edge(-1); out.hi = edge(+1); out.floor = floor;
    out.pick = Math.min(out.hi, Math.max(out.lo, slow[0]));
    return out;
  }

  /** Best shot for one distance, plus the whole family for plotting. */
  function solveAim(model, budget, distance, concession, thetaStep) {
    const step = thetaStep === undefined ? 0.25 : thetaStep;
    const family = [];
    for (let i = 0; ; i++) {
      const th = model.thetaMin + i * step;
      if (th > model.thetaMax + 1e-9) break;
      family.push(evalCentered(model, budget, distance, th));
    }
    const choice = chooseAngle(family.map((f) => f.theta), family.map((f) => f.v),
                               family.map((f) => f.margin), concession);
    if (!choice) return { success: false, family, choice: null };
    const shot = evalCentered(model, budget, distance, choice.pick);
    return Object.assign({ success: shot.ok, family, choice }, shot);
  }

  /** Score an arbitrary (angle, speed); it need not land centred. */
  function evaluateShot(model, budget, distance, hoodDeg, vExit) {
    const h0 = model.exitHeight, ht = model.targetHeight, aero = model.aero;
    const r = fly(vExit, hoodDeg, h0, ht, aero);
    if (!r.hit) return { hit: false, offset: NaN, pScore: 0, margin: 0, apex: r.apexY, apexX: r.apexX };
    const dxDlnv = (fly(vExit * (1 + 2e-3), hoodDeg, h0, ht, aero).x -
                    fly(vExit * (1 - 2e-3), hoodDeg, h0, ht, aero).x) / 4e-3;
    const dxDth = (fly(vExit, hoodDeg + 0.05, h0, ht, aero).x -
                   fly(vExit, hoodDeg - 0.05, h0, ht, aero).x) / 0.1;
    const entry = entryAngleDeg(r), offset = r.x - distance;
    const spread = landingSpread(budget, dxDlnv, dxDth, r.t, distance);
    const miss = missProbability(model.opening, budget, spread, entry, offset);
    return {
      hit: true, offset, pScore: 1 - miss, margin: sigmaMargin(miss), miss, tof: r.t, entry,
      apex: r.apexY, apexX: r.apexX, spread, dxDlnv, dxDtheta: dxDth,
      halfWidth: halfWidth(model.opening, entry),
    };
  }

  // ---- Monte Carlo burst (browser-side; uses its own RNG) -------------------------

  function mulberry32(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalSampler(uniform) {
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0;
      while (u === 0) u = uniform();
      const r = Math.sqrt(-2 * Math.log(u)), a = 2 * Math.PI * uniform();
      spare = r * Math.sin(a);
      return r * Math.cos(a);
    };
  }

  /**
   * Fire n balls at one aim point through `world` (which may differ from the
   * model the aim was solved with). Every error source is drawn independently;
   * each ball flies the full nonlinear trajectory and takes the exact rim test.
   */
  function burst(world, budget, distance, hoodDeg, vExit, n, seed, keepPaths) {
    const uni = mulberry32(seed), nrm = normalSampler(uni);
    const balls = [];
    let scored = 0;
    for (let i = 0; i < n; i++) {
      const v = vExit * (1 + budget.speed_sigma * nrm());
      const th = hoodDeg + budget.angle_sigma_deg * nrm();
      const yaw = budget.yaw_sigma_deg * Math.PI / 180 * nrm();
      const path = keepPaths ? [] : null;
      const r = fly(v, th, world.exitHeight, world.targetHeight, world.aero, keepPaths ? 0.01 : 0.005,
                    6.0, path);
      const nRange = nrm(), nAlong = nrm(), nLat = nrm(), lane = 2 * uni() - 1;
      const ball = { v, theta: th, yaw, hit: r.hit, scored: false, tof: r.t, entry: NaN,
                     along: NaN, lateral: NaN, path };
      if (r.hit) {
        const wander = budget.vortex_coeff * r.t * r.t;
        ball.entry = entryAngleDeg(r);
        ball.along = r.x * Math.cos(yaw) - distance + budget.range_sigma_m * nRange + wander * nAlong;
        ball.lateral = r.x * Math.sin(yaw) + budget.lateral_half_width_m * lane + wander * nLat;
        ball.scored = clears(world.opening, ball.along, ball.lateral, ball.entry);
      }
      if (ball.scored) scored++;
      balls.push(ball);
    }
    return { balls, scored, n, pScore: n ? scored / n : NaN };
  }

  return {
    erfc, ndtr, ndtri, aeroFromBall, fly, entryAngleDeg, noDragSpeed, speedForDistance,
    alongTrackLimits, halfWidth, clears, landingSpread, missProbability, sigmaMargin,
    evalCentered, chooseAngle, solveAim, evaluateShot, mulberry32, normalSampler, burst,
  };
});
