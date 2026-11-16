/* AlphaSim simulator UI. All physics lives in AlphaCore (alphasim_core.js);
 * this file only gathers the controls, calls the core, and draws. */
(function () {
  'use strict';
  const C = AlphaCore, M = DEP.model;
  const $ = (id) => document.getElementById(id);
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const quad = (q, d) => q.a * d * d + q.b * d + q.c;
  const D_MIN = DEP.distance_range_m[0], D_MAX = DEP.distance_range_m[1];
  const BURST_N = 100, BURST_SECONDS = 3.0, MC_TOTAL = 20000, MC_CHUNK = 1000, PATH_DT = 0.01;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const freshBudget = () => Object.assign({}, DEP.error_budget);
  const freshWorld = () => ({ cd: M.drag_coefficient, cl: M.lift_coefficient,
                              exitHeight: DEP.shooter_exit_height_m, bearing: 0 });
  const state = {
    d: 3.5, aim: 'best', concession: DEP.margin_concession, hood: 60, speed: 7.5,
    kSpeed: 1, kSlope: 0, seed: 42, slow: false, budget: freshBudget(), world: freshWorld(),
  };
  let R = null;                                    // latest computed result
  const anim = { playing: false, t: 0, last: 0, showAll: true, id: 0 };
  const mc = { token: 0, n: 0, scored: 0, cloud: [] };
  let hoverAngle = null;

  // ---- controls -------------------------------------------------------------

  const B = state.budget, W = state.world;
  const sliders = [
    { id: 'd', get: () => state.d, set: (v) => { state.d = v; }, fmt: (v) => v.toFixed(2) + ' m' },
    { id: 'concession', get: () => state.concession, set: (v) => { state.concession = v; },
      fmt: (v) => v === 0 ? 'none' : (v * 100).toFixed(1) + ' %' },
    { id: 'hood', get: () => state.hood, set: (v) => { state.hood = v; }, fmt: (v) => v.toFixed(1) + '°' },
    { id: 'speed', get: () => state.speed, set: (v) => { state.speed = v; }, fmt: (v) => v.toFixed(2) + ' m/s' },
    { id: 'bSpeed', get: () => state.budget.speed_sigma * 100, set: (v) => { state.budget.speed_sigma = v / 100; },
      fmt: (v) => v.toFixed(1) + ' %' },
    { id: 'bAngle', get: () => state.budget.angle_sigma_deg, set: (v) => { state.budget.angle_sigma_deg = v; },
      fmt: (v) => v.toFixed(2) + '°' },
    { id: 'bYaw', get: () => state.budget.yaw_sigma_deg, set: (v) => { state.budget.yaw_sigma_deg = v; },
      fmt: (v) => v.toFixed(1) + '°' },
    { id: 'bRange', get: () => state.budget.range_sigma_m * 100, set: (v) => { state.budget.range_sigma_m = v / 100; },
      fmt: (v) => v.toFixed(1) + ' cm' },
    { id: 'bVortex', get: () => state.budget.vortex_coeff, set: (v) => { state.budget.vortex_coeff = v; },
      fmt: (v) => v === 0 ? 'off' : (v * 100).toFixed(1) + ' cm after 1 s' },
    { id: 'bLane', get: () => state.budget.lateral_half_width_m * 100,
      set: (v) => { state.budget.lateral_half_width_m = v / 100; }, fmt: (v) => '±' + v.toFixed(0) + ' cm' },
    { id: 'kSpeed', get: () => state.kSpeed, set: (v) => { state.kSpeed = v; }, fmt: (v) => v.toFixed(3) },
    { id: 'kSlope', get: () => state.kSlope, set: (v) => { state.kSlope = v; },
      fmt: (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(3) },
    { id: 'wCd', get: () => state.world.cd, set: (v) => { state.world.cd = v; }, fmt: (v) => v.toFixed(2) },
    { id: 'wCl', get: () => state.world.cl, set: (v) => { state.world.cl = v; }, fmt: (v) => v.toFixed(2) },
    { id: 'wExit', get: () => state.world.exitHeight, set: (v) => { state.world.exitHeight = v; },
      fmt: (v) => v.toFixed(2) + ' m' },
    { id: 'wBearing', get: () => state.world.bearing, set: (v) => { state.world.bearing = v; },
      fmt: (v) => v.toFixed(0) + '°' },
  ];
  void B; void W;

  function syncControls() {
    for (const s of sliders) {
      const el = $(s.id), v = s.get();
      el.value = v;
      $(s.id + 'Out').textContent = s.fmt(v);
      const lo = parseFloat(el.min), hi = parseFloat(el.max);
      el.style.setProperty('--p', (100 * (clamp(v, lo, hi) - lo) / (hi - lo)).toFixed(2) + '%');
    }
    document.querySelector('input[name="aim"][value="' + state.aim + '"]').checked = true;
    $('bestBox').hidden = state.aim !== 'best';
    $('manualBox').hidden = state.aim !== 'manual';
    $('aimHint').textContent = {
      best: 'Solved live for this distance and the scatter below: the angle most likely to score.',
      shipped: 'What the robot actually runs: the three quadratics in ShooterAim.java, generated for a ' +
               DEP.shooter_exit_height_m.toFixed(2) + ' m exit.',
      manual: 'Set any angle and speed and see what it costs.',
    }[state.aim];
  }

  let settleTimer = 0;
  function changed(animate) {
    stopAnim();
    recompute();
    clearTimeout(settleTimer);
    settleTimer = setTimeout(() => { startMC(); if (animate && !reduceMotion) play(); }, animate ? 140 : 260);
  }

  function bind() {
    $('d').min = D_MIN; $('d').max = D_MAX;
    for (const s of sliders) {
      const el = $(s.id);
      el.addEventListener('input', () => { s.set(parseFloat(el.value)); syncControls(); changed(false); });
      el.addEventListener('change', () => { changed(true); });
    }
    for (const radio of document.querySelectorAll('input[name="aim"]')) {
      radio.addEventListener('change', () => {
        if (radio.value === 'manual' && R && R.cmd) {      // start from the shot on screen
          state.hood = clamp(R.cmd.hood, 25, 85);
          state.speed = clamp(R.cmd.v, 4, 13);
        }
        state.aim = radio.value; syncControls(); changed(true);
      });
    }
    $('centre').addEventListener('click', () => {
      if (!R) return;
      const v = C.speedForDistance(state.d, state.hood, R.world.exitHeight, R.world.targetHeight, R.world.aero);
      if (Number.isFinite(v)) { state.speed = clamp(v / R.gain, 4, 13); syncControls(); changed(true); }
      else note('No speed lands a ball on centre at ' + state.hood.toFixed(1) + '° from here.', true);
    });
    $('autocal').addEventListener('click', () => {
      if (!R || !R.cmd) return;
      const v = C.speedForDistance(state.d, R.cmd.hood, R.world.exitHeight, R.world.targetHeight, R.world.aero);
      if (!Number.isFinite(v)) return;
      const dc = clamp(state.d, D_MIN, D_MAX);
      state.kSpeed = clamp(v / R.cmd.v - state.kSlope * (dc - DEP.k_speed_slope_pivot_m), 0.85, 1.15);
      syncControls(); changed(true);
    });
    $('resetCal').addEventListener('click', () => { state.kSpeed = 1; state.kSlope = 0; syncControls(); changed(true); });
    $('resetBudget').addEventListener('click', () => { state.budget = freshBudget(); syncControls(); changed(true); });
    $('resetWorld').addEventListener('click', () => { state.world = freshWorld(); syncControls(); changed(true); });
    $('fire').addEventListener('click', play);
    $('reseed').addEventListener('click', () => { state.seed += 1; changed(true); });
    $('slow').addEventListener('change', (e) => { state.slow = e.target.checked; });
    $('theme').addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('alphasim-theme', next); } catch (e) { /* private mode */ }
      themeChanged();
    });

    const why = $('why');
    const angleAt = (ev) => {
      if (!whyGeom) return null;
      const r = why.getBoundingClientRect();
      const t = (ev.clientX - r.left - whyGeom.x0) / (whyGeom.x1 - whyGeom.x0);
      return (t < 0 || t > 1) ? null : whyGeom.thLo + t * (whyGeom.thHi - whyGeom.thLo);
    };
    why.addEventListener('pointermove', (ev) => { hoverAngle = angleAt(ev); drawWhy(); });
    why.addEventListener('pointerleave', () => { hoverAngle = null; drawWhy(); });
    why.addEventListener('click', (ev) => {
      const th = angleAt(ev);
      if (th === null || !R) return;
      const v = C.speedForDistance(state.d, th, R.model.exitHeight, R.model.targetHeight, R.model.aero);
      if (!Number.isFinite(v)) return;
      state.aim = 'manual'; state.hood = clamp(Math.round(th * 10) / 10, 25, 85);
      const vc = C.speedForDistance(state.d, state.hood, R.model.exitHeight, R.model.targetHeight, R.model.aero);
      state.speed = clamp(Number.isFinite(vc) ? vc : v, 4, 13);
      syncControls(); changed(true);
    });
  }

  // ---- compute ----------------------------------------------------------------

  function recompute() {
    const b = Object.assign({}, state.budget);
    const opening = { apothem: M.apothem_m, ballRadius: M.ball_radius_m, bearingDeg: state.world.bearing };
    const base = { exitHeight: state.world.exitHeight, targetHeight: M.target_height_m, opening,
                   thetaMin: M.theta_min, thetaMax: M.theta_max, vMax: M.v_max };
    const model = Object.assign({ aero: C.aeroFromBall(M.ball_mass_kg, M.ball_radius_m,
      M.drag_coefficient, M.lift_coefficient, M.air_density, M.gravity) }, base);
    const world = Object.assign({ aero: C.aeroFromBall(M.ball_mass_kg, M.ball_radius_m,
      state.world.cd, state.world.cl, M.air_density, M.gravity) }, base);
    const sol = C.solveAim(model, b, state.d, state.concession);
    const dc = clamp(state.d, D_MIN, D_MAX);
    let cmd = null;
    if (state.aim === 'best') { if (sol.success) cmd = { hood: sol.theta, v: sol.v }; }
    else if (state.aim === 'shipped') cmd = { hood: quad(DEP.hood_angle_deg, dc), v: quad(DEP.v_ball_ms, dc) };
    else cmd = { hood: state.hood, v: state.speed };
    const gain = state.kSpeed + state.kSlope * (dc - DEP.k_speed_slope_pivot_m);
    const res = { d: state.d, model, world, sol, cmd, gain, budget: b, maxApex: 0, tTotal: 0 };
    if (cmd) {
      res.vBall = cmd.v * gain;
      res.grade = C.evaluateShot(world, b, state.d, cmd.hood, res.vBall);
      res.nominalPath = [];
      res.nominal = C.fly(res.vBall, cmd.hood, world.exitHeight, world.targetHeight, world.aero,
                          PATH_DT, 6, res.nominalPath);
      res.burst = C.burst(world, b, state.d, cmd.hood, res.vBall, BURST_N, state.seed, true);
      res.burst.balls.forEach((ball, i) => {
        ball.t0 = i * BURST_SECONDS / BURST_N;
        ball.tEnd = ball.t0 + ball.path[ball.path.length - 1][0];
        res.tTotal = Math.max(res.tTotal, ball.tEnd);
        for (const p of ball.path) if (p[2] > res.maxApex) res.maxApex = p[2];
      });
      res.tTotal += 0.35;
    }
    R = res;
    updateReadouts();
    drawAll();
  }

  function startMC() {
    const token = ++mc.token;
    mc.n = 0; mc.scored = 0; mc.cloud = [];
    if (!R || !R.cmd || !R.grade) { updateScoreSub(); return; }
    const snap = R;
    let seed = 9001;
    (function step() {
      if (token !== mc.token) return;
      const r = C.burst(snap.world, snap.budget, snap.d, snap.cmd.hood, snap.vBall, MC_CHUNK, seed++, false);
      mc.n += r.n; mc.scored += r.scored;
      if (mc.cloud.length < 1200) {
        for (const q of r.balls) {
          if (q.hit && mc.cloud.length < 1200) mc.cloud.push(q);
        }
      }
      updateScoreSub();
      if (!anim.playing) drawTop();
      if (mc.n < MC_TOTAL) setTimeout(step, 0);
    })();
  }

  // ---- readouts -----------------------------------------------------------------

  const pct = (p) => (p * 100).toFixed(p > 0.9995 || p < 0.0005 ? 2 : 1) + ' %';
  const signedCm = (m) => (m >= 0 ? '+' : '−') + Math.abs(m * 100).toFixed(1);

  function note(text, warn) {
    const el = $('stageNote');
    el.textContent = text || '';
    el.classList.toggle('warn', !!warn);
  }

  function burstTally() {
    if (!R || !R.burst) return null;
    let landed = 0, scored = 0;
    for (const b of R.burst.balls) {
      if (anim.showAll || anim.t >= b.tEnd) { landed++; if (b.scored) scored++; }
    }
    return { landed, scored };
  }

  function updateScoreSub() {
    const tally = burstTally();
    const parts = [];
    if (mc.n > 0) {
      parts.push('Simulated ' + mc.n.toLocaleString('en-US') + ' balls: ' + pct(mc.scored / mc.n));
    }
    if (tally) {
      parts.push(anim.showAll ? 'This burst: ' + tally.scored + ' of ' + BURST_N
                 : tally.landed ? 'This burst: ' + tally.scored + ' of ' + tally.landed + ' so far'
                 : 'This burst: in the air');
    }
    $('rScoreSub').textContent = parts.join('. ');
  }

  function updateReadouts() {
    const set = (id, html) => { $(id).innerHTML = html; };
    const g = R && R.grade;
    if (!R || !R.cmd) {
      for (const id of ['rScore', 'rHood', 'rFly', 'rTof', 'rEntry', 'rLand']) set(id, '–');
      for (const id of ['rHoodSub', 'rFlySub', 'rTofSub', 'rEntrySub', 'rLandSub']) set(id, '');
      note('No hood angle lands a ball in the opening from ' + state.d.toFixed(2) + ' m at this exit height.', true);
      updateScoreSub(); updateBudget();
      return;
    }
    const fly = R.vBall / DEP.speed_transfer;
    const rpm = fly / (Math.PI * M.wheel_diameter_m) * 60;
    set('rHood', R.cmd.hood.toFixed(1) + '°');
    set('rHoodSub', 'above horizontal');
    set('rFly', fly.toFixed(2) + '<small>m/s</small>');
    set('rFlySub', Math.round(rpm).toLocaleString('en-US') + ' rpm on a ' +
        (M.wheel_diameter_m * 1000).toFixed(0) + ' mm wheel. Ball leaves at ' + R.vBall.toFixed(2) + ' m/s');
    const score = $('rScore');
    if (!g.hit) {
      score.textContent = '0 %'; score.className = 'poor';
      for (const id of ['rTof', 'rEntry', 'rLand']) set(id, '–');
      set('rTofSub', 'peak ' + g.apex.toFixed(2) + ' m'); set('rEntrySub', ''); set('rLandSub', '');
      note('The ball never comes down through the opening at this angle and speed.', true);
      updateScoreSub(); updateBudget();
      return;
    }
    score.textContent = pct(g.pScore);
    score.className = g.pScore >= 0.99 ? 'good' : g.pScore >= 0.9 ? 'fair' : 'poor';
    set('rTof', g.tof.toFixed(2) + '<small>s</small>');
    set('rTofSub', 'peak height ' + g.apex.toFixed(2) + ' m');
    set('rEntry', g.entry.toFixed(0) + '°');
    set('rEntrySub', 'leaves ±' + (g.halfWidth * 100).toFixed(0) + ' cm of room');
    const centred = Math.abs(g.offset) < 0.005;
    set('rLand', centred ? 'on centre' : signedCm(g.offset) + '<small>cm</small>');
    $('rLand').style.fontSize = centred ? '20px' : '';
    set('rLandSub', 'spread ±' + (g.spread.along * 100).toFixed(1) + ' cm, margin ' +
        g.margin.toFixed(2) + ' σ');
    if (Math.abs(g.offset) > 0.04) {
      note('Balls land ' + Math.abs(g.offset * 100).toFixed(0) + ' cm ' + (g.offset > 0 ? 'long' : 'short') +
           '. ' + (g.offset > 0 ? 'Lower' : 'Raise') + ' kSpeed to centre them.', true);
    } else note('Each streak is one ball. The ticks above the opening show where it came down.');
    updateScoreSub(); updateBudget();
  }

  const SOURCES = [['speed', 'Ball speed', '--src-speed'], ['angle', 'Launch angle', '--src-angle'],
                   ['vortex', 'Air wobble', '--src-wake'], ['range', 'Distance estimate', '--src-range']];

  function updateBudget() {
    const bars = $('bars'), key = $('barsKey');
    const sp = R && R.grade && R.grade.hit ? R.grade.spread : null;
    if (!sp) { bars.innerHTML = ''; key.innerHTML = ''; $('budTotal').textContent = '–'; return; }
    $('budTotal').textContent = '±' + (sp.along * 100).toFixed(1) + ' cm';
    // Sources add in quadrature, so each one's share of the total is its variance.
    const total = SOURCES.reduce((sum, [k]) => sum + sp[k] * sp[k], 0) || 1;
    bars.innerHTML = SOURCES.map(([k, , colour]) =>
      '<span style="flex:' + (1000 * sp[k] * sp[k] / total).toFixed(2) + ' 1 0;background:var(' + colour + ')"></span>').join('');
    key.innerHTML = SOURCES.map(([k, label, colour]) =>
      '<span><i style="background:var(' + colour + ')"></i>' + label + '<b>' + (sp[k] * 100).toFixed(1) + ' cm</b></span>').join('');
  }

  // ---- drawing --------------------------------------------------------------------

  let COL = {};
  function readColours() {
    const cs = getComputedStyle(document.documentElement);
    for (const n of ['ink-0', 'ink-1', 'ink-2', 'line', 'line-soft', 'text', 'text-2', 'text-3', 'fuel',
                     'score', 'miss', 'accent', 'src-speed', 'src-angle', 'src-wake', 'src-range']) {
      COL[n] = cs.getPropertyValue('--' + n).trim();
    }
  }
  const FONT = (w, px) => w + ' ' + px + 'px "Barlow Semi Condensed", "Barlow", system-ui, sans-serif';

  function fit(canvas) {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function alpha(ctx, a, fn) { ctx.save(); ctx.globalAlpha = a; fn(); ctx.restore(); }

  function ballAt(ball, t) {
    const p = ball.path, tau = t - ball.t0;
    if (tau <= 0) return null;
    const last = p.length - 1;
    if (tau >= p[last][0]) return { i: last, x: p[last][1], y: p[last][2], done: true };
    const i = Math.min(last - 1, Math.floor(tau / PATH_DT));
    const f = (tau - p[i][0]) / (p[i + 1][0] - p[i][0]);
    return { i, x: p[i][1] + f * (p[i + 1][1] - p[i][1]), y: p[i][2] + f * (p[i + 1][2] - p[i][2]), done: false };
  }

  function drawSide() {
    const { ctx, w, h } = fit($('side'));
    const d = state.d, H = M.target_height_m, a = M.apothem_m;
    let xMin = -0.95, xMax = Math.max(d + 1.05, 4.3);
    const yTop = Math.max(3.25, (R ? R.maxApex : 0) + 0.5);
    const padL = 0, padR = 0, padT = 50, padB = 44;
    const scale = Math.min((w - padL - padR) / (xMax - xMin), (h - padT - padB) / yTop);
    // When height is the limit, spend the spare width on more floor either side.
    const spare = (w - padL - padR) / scale - (xMax - xMin);
    xMin -= spare * 0.42; xMax += spare * 0.58;
    const ox = padL - xMin * scale;
    const oy = h - padB;
    const X = (x) => ox + x * scale, Y = (y) => oy - y * scale;

    // metre grid
    ctx.lineWidth = 1; ctx.strokeStyle = COL['line-soft'];
    ctx.font = FONT(500, 12); ctx.fillStyle = COL['text-3']; ctx.textBaseline = 'middle';
    for (let ym = 1; ym <= Math.floor(yTop); ym++) {
      ctx.beginPath(); ctx.moveTo(X(xMin), Y(ym) + 0.5); ctx.lineTo(X(xMax), Y(ym) + 0.5); ctx.stroke();
      if (Y(ym) > 58) { ctx.textAlign = 'left'; ctx.fillText(ym + ' m', 12, Y(ym) - 8); }
    }
    // floor
    ctx.strokeStyle = COL['line']; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(xMin), oy + 0.5); ctx.lineTo(X(xMax), oy + 0.5); ctx.stroke();

    // distance dimension under the floor
    ctx.strokeStyle = COL['text-3']; ctx.fillStyle = COL['text-2']; ctx.lineWidth = 1;
    const dimY = oy + 20;
    ctx.beginPath();
    ctx.moveTo(X(0), oy + 6); ctx.lineTo(X(0), dimY + 5); ctx.moveTo(X(d), oy + 6); ctx.lineTo(X(d), dimY + 5);
    ctx.moveTo(X(0), dimY); ctx.lineTo(X(d), dimY); ctx.stroke();
    ctx.font = FONT(600, 13.5); ctx.textAlign = 'center';
    const label = d.toFixed(2) + ' m';
    const lw = ctx.measureText(label).width + 12;
    ctx.fillStyle = COL['ink-1']; ctx.fillRect(X(d / 2) - lw / 2, dimY - 9, lw, 18);
    ctx.fillStyle = COL['text']; ctx.fillText(label, X(d / 2), dimY + 1);

    // hub: base box, funnel, opening
    ctx.fillStyle = COL['ink-2']; ctx.strokeStyle = COL['line']; ctx.lineWidth = 1.5;
    const boxTop = 1.42, baseHalf = 0.597, neck = 0.37;
    ctx.beginPath();
    ctx.moveTo(X(d - baseHalf), oy); ctx.lineTo(X(d - baseHalf), Y(boxTop)); ctx.lineTo(X(d - neck), Y(boxTop));
    ctx.lineTo(X(d - a), Y(H)); ctx.lineTo(X(d + a), Y(H)); ctx.lineTo(X(d + neck), Y(boxTop));
    ctx.lineTo(X(d + baseHalf), Y(boxTop)); ctx.lineTo(X(d + baseHalf), oy); ctx.closePath();
    ctx.fill(); ctx.stroke();
    // inside of the funnel reads as open
    ctx.fillStyle = COL['ink-1'];
    ctx.beginPath();
    ctx.moveTo(X(d - a) + 2, Y(H)); ctx.lineTo(X(d + a) - 2, Y(H)); ctx.lineTo(X(d + neck) - 2, Y(boxTop) - 2);
    ctx.lineTo(X(d - neck) + 2, Y(boxTop) - 2); ctx.closePath(); ctx.fill();
    // usable window at this entry angle
    const g = R && R.grade && R.grade.hit ? R.grade : null;
    if (g && g.halfWidth > 0) {
      ctx.strokeStyle = COL['score']; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(X(d - g.halfWidth), Y(H)); ctx.lineTo(X(d + g.halfWidth), Y(H)); ctx.stroke();
      ctx.lineCap = 'butt';
    }
    ctx.fillStyle = COL['text'];
    for (const s of [-1, 1]) { ctx.beginPath(); ctx.arc(X(d + s * a), Y(H), 3, 0, 7); ctx.fill(); }
    ctx.font = FONT(500, 12.5); ctx.fillStyle = COL['text-2']; ctx.textAlign = 'left';
    const openLabel = 'opening ' + H.toFixed(2) + ' m up';
    if (X(d + a) + 14 + ctx.measureText(openLabel).width < w) ctx.fillText(openLabel, X(d + a) + 9, Y(H));

    // robot: bumpers, wheels, launcher
    const h0 = state.world.exitHeight;
    ctx.fillStyle = COL['ink-2']; ctx.strokeStyle = COL['line']; ctx.lineWidth = 1.5;
    const rx0 = X(-0.82), rx1 = X(0.04);
    ctx.beginPath(); ctx.roundRect(rx0, Y(0.2), rx1 - rx0, scale * 0.14, 3); ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(X(-0.7), Y(0.2)); ctx.lineTo(X(-0.56), Y(Math.max(0.3, h0 - 0.03)));
    ctx.lineTo(X(-0.06), Y(Math.max(0.3, h0 - 0.03))); ctx.lineTo(X(-0.02), Y(0.2)); ctx.closePath();
    ctx.fill(); ctx.stroke();
    for (const wx of [-0.68, -0.1]) {
      ctx.beginPath(); ctx.arc(X(wx), Y(0.055), scale * 0.055, 0, 7); ctx.fill(); ctx.stroke();
    }
    if (R && R.cmd) {
      const th = R.cmd.hood * Math.PI / 180, len = 0.3 * scale;
      ctx.strokeStyle = COL['text-2']; ctx.lineWidth = Math.max(5, scale * 0.085); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(X(0) - Math.cos(th) * len, Y(h0) + Math.sin(th) * len);
      ctx.lineTo(X(0), Y(h0)); ctx.stroke(); ctx.lineCap = 'butt';
      ctx.fillStyle = COL['text']; ctx.font = FONT(600, 13.5); ctx.textAlign = 'right';
      ctx.fillText(R.cmd.hood.toFixed(1) + '°', X(0) - Math.cos(th) * len - 8, Y(h0) + Math.sin(th) * len + 2);
    }
    if (!R || !R.burst) return;

    // ball streaks (long exposure)
    const t = anim.showAll ? Infinity : anim.t;
    ctx.strokeStyle = COL['fuel']; ctx.lineWidth = 1.15; ctx.lineJoin = 'round';
    alpha(ctx, document.documentElement.dataset.theme === 'light' ? 0.3 : 0.2, () => {
      for (const ball of R.burst.balls) {
        const pos = ballAt(ball, t);
        if (!pos) continue;
        const p = ball.path;
        ctx.beginPath(); ctx.moveTo(X(p[0][1]), Y(p[0][2]));
        for (let i = 1; i <= pos.i; i++) ctx.lineTo(X(p[i][1]), Y(p[i][2]));
        if (!pos.done) ctx.lineTo(X(pos.x), Y(pos.y));
        ctx.stroke();
      }
    });

    // the nominal ball, dashed, and its peak
    if (R.nominalPath.length > 1) {
      ctx.strokeStyle = COL['text']; ctx.lineWidth = 1.3; ctx.setLineDash([5, 5]);
      alpha(ctx, 0.75, () => {
        ctx.beginPath();
        R.nominalPath.forEach((p, i) => (i ? ctx.lineTo(X(p[1]), Y(p[2])) : ctx.moveTo(X(p[1]), Y(p[2]))));
        ctx.stroke();
      });
      ctx.setLineDash([]);
      const ax = X(R.nominal.apexX), ay = Y(R.nominal.apexY);
      ctx.strokeStyle = COL['text-2']; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ax, ay - 5); ctx.lineTo(ax, ay - 16); ctx.stroke();
      ctx.fillStyle = COL['text']; ctx.font = FONT(600, 13); ctx.textAlign = 'center';
      ctx.fillText('peak ' + R.nominal.apexY.toFixed(2) + ' m', clamp(ax, 46, w - 46), ay - 25);
    }

    // where each landed ball crossed the opening plane
    for (const ball of R.burst.balls) {
      if (!ball.hit || t < ball.tEnd) continue;
      ctx.strokeStyle = ball.scored ? COL['score'] : COL['miss']; ctx.lineWidth = 1.6;
      const px = X(d + ball.along);
      alpha(ctx, 0.85, () => { ctx.beginPath(); ctx.moveTo(px, Y(H) - 9); ctx.lineTo(px, Y(H) - 3); ctx.stroke(); });
    }

    // balls in the air
    if (!anim.showAll) {
      const rad = Math.max(2.6, M.ball_radius_m * scale);
      ctx.fillStyle = COL['fuel']; ctx.strokeStyle = COL['ink-1']; ctx.lineWidth = 1;
      for (const ball of R.burst.balls) {
        const pos = ballAt(ball, t);
        if (!pos || pos.done) continue;
        ctx.beginPath(); ctx.arc(X(pos.x), Y(pos.y), rad, 0, 7); ctx.fill(); ctx.stroke();
      }
    }
  }

  function drawTop() {
    const { ctx, w, h } = fit($('top'));
    const a = M.apothem_m, r = M.ball_radius_m, bearing = state.world.bearing;
    const ext = 0.86, s = Math.min(w / (2 * ext), h / (2 * 0.70));
    const cx = w / 2, cy = h / 2;
    const X = (along) => cx + along * s, Y = (lat) => cy - lat * s;

    // rim hexagon
    const circ = a / Math.cos(Math.PI / 6);
    ctx.beginPath();
    for (let k = 0; k < 6; k++) {
      const ang = (bearing + 30 + 60 * k) * Math.PI / 180;
      const px = X(circ * Math.cos(ang)), py = Y(circ * Math.sin(ang));
      if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = COL['ink-0']; ctx.fill();
    ctx.strokeStyle = COL['text-2']; ctx.lineWidth = 2; ctx.stroke();

    const g = R && R.grade && R.grade.hit ? R.grade : null;
    if (g) {
      // region where a ball at this entry angle clears every edge
      const opening = R.world.opening, lo = [], hi = [];
      for (let z = -0.7; z <= 0.7001; z += 0.01) {
        const lim = C.alongTrackLimits(opening, z, g.entry);
        if (lim[1] > lim[0]) { lo.push([lim[0], z]); hi.push([lim[1], z]); }
      }
      if (lo.length) {
        ctx.beginPath();
        lo.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
        for (let i = hi.length - 1; i >= 0; i--) ctx.lineTo(X(hi[i][0]), Y(hi[i][1]));
        ctx.closePath();
        alpha(ctx, 0.16, () => { ctx.fillStyle = COL['score']; ctx.fill(); });
        alpha(ctx, 0.55, () => { ctx.strokeStyle = COL['score']; ctx.lineWidth = 1; ctx.stroke(); });
      }
    }
    // centre mark
    ctx.strokeStyle = COL['text-3']; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6); ctx.stroke();

    // simulated cloud, then this burst
    for (const q of mc.cloud) {
      ctx.fillStyle = q.scored ? COL['score'] : COL['miss'];
      alpha(ctx, 0.3, () => { ctx.beginPath(); ctx.arc(X(q.along), Y(q.lateral), 1.2, 0, 7); ctx.fill(); });
    }
    if (R && R.burst) {
      const t = anim.showAll ? Infinity : anim.t;
      ctx.lineWidth = 1; ctx.strokeStyle = COL['ink-1'];
      for (const ball of R.burst.balls) {
        if (!ball.hit || t < ball.tEnd) continue;
        if (Math.abs(ball.along) > ext || Math.abs(ball.lateral) > 0.7) continue;
        ctx.fillStyle = ball.scored ? COL['score'] : COL['miss'];
        ctx.beginPath(); ctx.arc(X(ball.along), Y(ball.lateral), 3.3, 0, 7); ctx.fill(); ctx.stroke();
      }
    }
    if (g) {
      const sa = g.spread.along, L = R.budget.lateral_half_width_m;
      const sl = Math.sqrt(g.spread.lateral * g.spread.lateral + L * L / 3);
      ctx.strokeStyle = COL['fuel']; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
      for (const k of [1, 2]) {
        ctx.beginPath(); ctx.ellipse(X(g.offset), cy, k * sa * s, Math.max(1, k * sl * s), 0, 0, 7); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    // shot direction
    ctx.fillStyle = COL['text-2']; ctx.font = FONT(500, 12.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    const ay = h - 12;
    ctx.fillText('shot direction', 10, ay);
    const tx = 10 + ctx.measureText('shot direction').width + 8;
    ctx.strokeStyle = COL['text-2']; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(tx, ay); ctx.lineTo(tx + 26, ay); ctx.moveTo(tx + 20, ay - 4);
    ctx.lineTo(tx + 26, ay); ctx.lineTo(tx + 20, ay + 4); ctx.stroke();
    // scale bar
    const bar = 0.25 * s;
    ctx.beginPath(); ctx.moveTo(w - 12 - bar, ay); ctx.lineTo(w - 12, ay);
    ctx.moveTo(w - 12 - bar, ay - 4); ctx.lineTo(w - 12 - bar, ay + 4); ctx.moveTo(w - 12, ay - 4); ctx.lineTo(w - 12, ay + 4); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText('25 cm', w - 18 - bar, ay);
    void r;
  }

  let whyGeom = null;
  function drawWhy() {
    const { ctx, w, h } = fit($('why'));
    whyGeom = null;
    if (!R || !R.sol || !R.sol.choice) {
      ctx.fillStyle = COL['text-2']; ctx.font = FONT(500, 14); ctx.textAlign = 'center';
      ctx.fillText('No shot reaches the opening from here.', w / 2, h / 2);
      return;
    }
    const ch = R.sol.choice;
    // Start where a shot first has any margin at all; flatter angles only graze the rim.
    const fam = R.sol.family.filter((f) => f.ok && f.margin > 0.02);
    if (fam.length < 3) return;
    const thLo = Math.floor(fam[0].theta), thHi = M.theta_max;
    const x0 = 44, x1 = w - 12, labelRows = 34;
    const topA = labelRows, topB = labelRows + (h - labelRows - 24) * 0.62;
    const botA = topB + 14, botB = h - 24;
    const X = (th) => x0 + (th - thLo) / (thHi - thLo) * (x1 - x0);
    whyGeom = { x0, x1, thLo, thHi };

    const mMax = Math.max(ch.bestMargin, 0.5) * 1.12;
    let vMin = Infinity; for (const f of fam) vMin = Math.min(vMin, f.v);
    const vLo = vMin * 0.985, vHi = vMin * 1.42;
    const Ym = (m) => topB - clamp(m, 0, mMax) / mMax * (topB - topA);
    const Yv = (v) => botB - (clamp(v, vLo, vHi) - vLo) / (vHi - vLo) * (botB - botA);

    // frames + grid
    ctx.font = FONT(500, 12); ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
    ctx.strokeStyle = COL['line-soft']; ctx.fillStyle = COL['text-3'];
    const mStep = mMax > 6 ? 2 : 1;
    for (let m = 0; m <= mMax; m += mStep) {
      ctx.beginPath(); ctx.moveTo(x0, Ym(m) + 0.5); ctx.lineTo(x1, Ym(m) + 0.5); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(m + ' σ', x0 - 7, Ym(m));
    }
    for (let v = Math.ceil(vLo); v <= vHi; v += 1) {
      ctx.strokeStyle = COL['line-soft'];
      ctx.beginPath(); ctx.moveTo(x0, Yv(v) + 0.5); ctx.lineTo(x1, Yv(v) + 0.5); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(v.toFixed(0), x0 - 7, Yv(v));
    }
    ctx.beginPath(); ctx.moveTo(x0, botB + 0.5); ctx.lineTo(x1, botB + 0.5); ctx.stroke();
    ctx.textAlign = 'center';
    for (let th = Math.ceil(thLo / 10) * 10; th <= thHi; th += 10) ctx.fillText(th + '°', X(th), botB + 12);
    ctx.textAlign = 'left'; ctx.fillStyle = COL['text-2'];
    ctx.fillText('margin', x0 + 6, topA + 9);
    ctx.fillText('ball speed, m/s', x0 + 6, botA + 8);

    // acceptable band (best shot mode only)
    if (state.concession > 0 && ch.hi > ch.lo) {
      alpha(ctx, 0.14, () => { ctx.fillStyle = COL['accent']; ctx.fillRect(X(ch.lo), topA, X(ch.hi) - X(ch.lo), botB - topA); });
    }

    // curves
    const curve = (yOf, colour, width) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.beginPath();
      let pen = false;
      for (const f of fam) {
        const y = yOf(f);
        if (y === null) { pen = false; continue; }
        if (pen) ctx.lineTo(X(f.theta), y); else { ctx.moveTo(X(f.theta), y); pen = true; }
      }
      ctx.stroke();
    };
    curve((f) => Ym(f.margin), COL['accent'], 2.2);
    curve((f) => (f.v <= vHi ? Yv(f.v) : null), COL['text-2'], 1.8);

    // guides
    const guides = [
      { th: ch.slow, text: 'slowest', colour: COL['text-3'], dash: [3, 4] },
      { th: ch.best, text: 'best odds', colour: COL['text-3'], dash: [3, 4] },
    ];
    if (R.cmd) guides.push({ th: R.cmd.hood, text: state.aim === 'best' ? 'pick' : 'this shot', colour: COL['fuel'], dash: [] });
    ctx.font = FONT(600, 12.5);
    const placed = [];
    for (const gd of guides.sort((p, q) => p.th - q.th)) {
      if (gd.th < thLo || gd.th > thHi) continue;
      const gx = X(gd.th);
      ctx.strokeStyle = gd.colour; ctx.lineWidth = gd.dash.length ? 1 : 1.6; ctx.setLineDash(gd.dash);
      ctx.beginPath(); ctx.moveTo(gx + 0.5, topA - 2); ctx.lineTo(gx + 0.5, botB); ctx.stroke(); ctx.setLineDash([]);
      const tw = ctx.measureText(gd.text).width;
      const tx = clamp(gx, x0 + tw / 2, x1 - tw / 2);
      let row = 0;
      while (row < 1 && placed.some((p) => p.row === row && Math.abs(p.x - tx) < (p.w + tw) / 2 + 8)) row++;
      placed.push({ x: tx, w: tw, row });
      ctx.fillStyle = gd.dash.length ? COL['text-2'] : COL['fuel']; ctx.textAlign = 'center';
      ctx.fillText(gd.text, tx, row === 0 ? 9 : 24);
    }
    // the shot on screen, as dots on both plots
    if (R.cmd && R.grade && R.grade.hit && R.cmd.hood >= thLo && R.cmd.hood <= thHi) {
      ctx.fillStyle = COL['fuel']; ctx.strokeStyle = COL['ink-1']; ctx.lineWidth = 1.5;
      for (const y of [Ym(R.grade.margin), R.vBall <= vHi && R.vBall >= vLo ? Yv(R.vBall) : null]) {
        if (y === null) continue;
        ctx.beginPath(); ctx.arc(X(R.cmd.hood), y, 4.5, 0, 7); ctx.fill(); ctx.stroke();
      }
    }
    // hover read-out
    if (hoverAngle !== null) {
      let near = fam[0];
      for (const f of fam) if (Math.abs(f.theta - hoverAngle) < Math.abs(near.theta - hoverAngle)) near = f;
      const gx = X(near.theta);
      ctx.strokeStyle = COL['text']; ctx.lineWidth = 1;
      alpha(ctx, 0.5, () => { ctx.beginPath(); ctx.moveTo(gx + 0.5, topA); ctx.lineTo(gx + 0.5, botB); ctx.stroke(); });
      const tip = near.theta.toFixed(1) + '°   margin ' + near.margin.toFixed(2) + ' σ   ' +
                  pct(near.pScore) + '   ' + near.v.toFixed(2) + ' m/s';
      ctx.font = FONT(600, 12.5);
      const tw = ctx.measureText(tip).width + 14;
      const bx = clamp(gx - tw / 2, x0, x1 - tw), by = topB - 26;
      ctx.fillStyle = COL['ink-2']; ctx.strokeStyle = COL['line'];
      ctx.beginPath(); ctx.roundRect(bx, by, tw, 22, 5); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COL['text']; ctx.textAlign = 'left'; ctx.fillText(tip, bx + 7, by + 11.5);
    }
  }

  function drawAll() { drawSide(); drawTop(); drawWhy(); }
  function themeChanged() {
    $('theme').textContent = document.documentElement.dataset.theme === 'light' ? 'Dark' : 'Light';
    readColours(); drawAll();
  }

  // ---- animation --------------------------------------------------------------------

  function stopAnim() { anim.playing = false; anim.showAll = true; anim.id++; }

  function play() {
    if (!R || !R.burst) return;
    const id = ++anim.id;
    anim.playing = true; anim.showAll = false; anim.t = 0; anim.last = 0;
    const frame = (now) => {
      if (id !== anim.id) return;
      if (!anim.last) anim.last = now;
      anim.t += Math.min(0.05, (now - anim.last) / 1000) * (state.slow ? 0.25 : 1);
      anim.last = now;
      if (anim.t >= R.tTotal) { anim.playing = false; anim.showAll = true; }
      drawSide(); drawTop(); updateScoreSub();
      if (anim.playing) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // ---- start --------------------------------------------------------------------------

  $('foot').innerHTML =
    'The ball is a point mass with quadratic drag (coefficient ' + M.drag_coefficient.toFixed(2) +
    ') and no spin, launched from ' + DEP.shooter_exit_height_m.toFixed(2) + ' m toward a ' +
    (2 * M.apothem_m).toFixed(2) + ' m hexagonal opening ' + M.target_height_m.toFixed(2) +
    ' m up. A ball scores if it passes the opening without touching the rim. The scatter values are ' +
    'estimates until someone measures them on the robot. The <a href="research_report.html">report</a> ' +
    'explains the model, the solver and what to measure.';

  readColours();
  bind();
  syncControls();
  themeChanged();
  recompute();
  startMC();
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => { if (!anim.playing) drawAll(); });
    for (const id of ['side', 'top', 'why']) ro.observe($(id));
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawAll);
  if (!reduceMotion) setTimeout(play, 350);
  window.AlphaSimUI = { state, get result() { return R; }, play, recompute };
})();
