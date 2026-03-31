// src/components/charts.js
// Canvas chart helpers — violet/indigo/teal palette

const Charts = {
  colors: {
    violet:'#8b5cf6', indigo:'#6366f1', teal:'#2dd4bf', sky:'#38bdf8',
    rose:'#fb7185', amber:'#fbbf24', emerald:'#34d399',
    muted:'#3a3d6a', text:'#6366a0', grid:'rgba(99,102,241,0.06)',
    bg:'#080b16',
  },

  drawLine(canvasId, datasets, labels) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top:18, right:18, bottom:28, left:40 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (cH / 5) * i;
      ctx.beginPath(); ctx.strokeStyle = this.colors.grid; ctx.lineWidth = 1;
      ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
      ctx.fillStyle = this.colors.text; ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'right'; ctx.fillText((1 - i / 5).toFixed(1), pad.left - 5, y + 3);
    }
    labels.forEach((lbl, i) => {
      const x = pad.left + (i / Math.max(labels.length - 1, 1)) * cW;
      ctx.fillStyle = this.colors.text; ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'center'; ctx.fillText(lbl, x, H - 6);
    });
    datasets.forEach(ds => {
      if (ds.data.length < 1) return;
      const pts = ds.data.map((v, i) => ({
        x: pad.left + (i / Math.max(ds.data.length - 1, 1)) * cW,
        y: pad.top  + (1 - v) * cH,
      }));
      // Area fill (only if 2+ points)
      if (ds.data.length >= 2) {
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
        grad.addColorStop(0, ds.color + '25');
        grad.addColorStop(1, ds.color + '02');
        ctx.beginPath(); ctx.moveTo(pts[0].x, pad.top + cH);
        pts.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.lineTo(pts[pts.length-1].x, pad.top + cH);
        ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
        // Line
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
        pts.forEach(p => ctx.lineTo(p.x, p.y));
        ctx.strokeStyle = ds.color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
      }
      // Dots
      pts.forEach(p => {
        ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = ds.color; ctx.fill();
        ctx.strokeStyle = this.colors.bg; ctx.lineWidth = 1.5; ctx.stroke();
      });
    });
  },

  drawBars(canvasId, values, labels, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top:12, right:14, bottom:28, left:40 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
    const maxV = Math.max(...values, 0.5);

    ctx.clearRect(0, 0, W, H);
    [0.25, 0.5, 0.75, 1].forEach(t => {
      const y = pad.top + (1 - t) * cH;
      ctx.beginPath(); ctx.strokeStyle = this.colors.grid; ctx.lineWidth = 1;
      ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
      ctx.fillStyle = this.colors.text; ctx.font = '9px JetBrains Mono';
      ctx.textAlign = 'right'; ctx.fillText((t * maxV).toFixed(2), pad.left - 4, y + 3);
    });
    const gap = 3, barW = (cW / values.length) - gap;
    values.forEach((v, i) => {
      const barH = (v / maxV) * cH;
      const x = pad.left + i * (barW + gap), y = pad.top + cH - barH;
      const grad = ctx.createLinearGradient(x, y, x, y + barH);
      grad.addColorStop(0, (colors && colors[i]) || this.colors.violet);
      grad.addColorStop(1, (colors && colors[i]) || this.colors.indigo + '60');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]); ctx.fill();
      ctx.fillStyle = this.colors.text; ctx.font = '8px JetBrains Mono';
      ctx.textAlign = 'center'; ctx.fillText(labels[i], x + barW / 2, H - 6);
    });
  },

  drawDonut(canvasId, slices, centerText) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const outer = Math.min(W, H) / 2 - 6, inner = outer * 0.6;
    const total = slices.reduce((s, x) => s + x.value, 0);
    ctx.clearRect(0, 0, W, H);
    let angle = -Math.PI / 2;
    slices.forEach(sl => {
      const sweep = (sl.value / total) * Math.PI * 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, outer, angle, angle + sweep);
      ctx.fillStyle = sl.color; ctx.fill(); angle += sweep;
    });
    ctx.beginPath(); ctx.arc(cx, cy, inner, 0, Math.PI * 2);
    ctx.fillStyle = '#0d1225'; ctx.fill();
    ctx.fillStyle = '#e2e4f0'; ctx.font = '700 18px Syne'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(centerText || total, cx, cy - 5);
    ctx.fillStyle = '#6366a0'; ctx.font = '9px JetBrains Mono'; ctx.fillText('nodes', cx, cy + 10);
  },

  drawRadar(canvasId, values, labels, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const r = Math.min(W, H) / 2 - 28, n = values.length;
    ctx.clearRect(0, 0, W, H);

    [0.25, 0.5, 0.75, 1].forEach(t => {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 - Math.PI / 2;
        const px = cx + Math.cos(a) * r * t, py = cy + Math.sin(a) * r * t;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath(); ctx.strokeStyle = 'rgba(99,102,241,0.08)'; ctx.lineWidth = 1; ctx.stroke();
    });
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r);
      ctx.strokeStyle = 'rgba(99,102,241,0.06)'; ctx.stroke();
    }
    ctx.beginPath();
    values.forEach((v, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      const px = cx + Math.cos(a) * r * v, py = cy + Math.sin(a) * r * v;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = color + '20'; ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

    labels.forEach((lbl, i) => {
      const a = (i / n) * Math.PI * 2 - Math.PI / 2;
      const px = cx + Math.cos(a) * (r + 18), py = cy + Math.sin(a) * (r + 18);
      ctx.fillStyle = '#6366a0'; ctx.font = '8px JetBrains Mono';
      ctx.textAlign = Math.cos(a) > 0.1 ? 'left' : Math.cos(a) < -0.1 ? 'right' : 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(lbl.replace(/_/g,' ').substring(0,12), px, py);
    });
  },

  // ── Animated Waveforms (Continuous 60fps) ───────────────────────────
  activeWaveforms: {},

  startAnims() {
    const loop = () => {
      Object.keys(this.activeWaveforms).forEach(id => {
        const wf = this.activeWaveforms[id];
        this.drawWaveformFrame(id, wf.color, wf.freq, wf.amp);
      });
      requestAnimationFrame(loop);
    };
    loop();
  },

  drawWaveform(canvasId, color, freq, amp) {
    this.activeWaveforms[canvasId] = { color, freq, amp };
  },

  drawWaveformFrame(canvasId, color, freq, amplitude) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const mid = H / 2;
    const time = Date.now() * 0.002;
    
    // Main line
    ctx.beginPath();
    for (let x = 0; x < W; x++) {
      const noise = Math.sin(x * 0.2 + time) * 2 + Math.sin(x * 0.5) * 1;
      const y = mid + Math.sin(x * freq * 0.03 + time * freq) * amplitude * (H * 0.35) + noise;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

    // Outer glow
    ctx.strokeStyle = color + '20'; ctx.lineWidth = 6; ctx.stroke();
  },

  drawHeatmap(canvasId, data, xLabels, yLabels) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = { top:6, right:6, bottom:24, left:50 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
    const cellW = cW / xLabels.length, cellH = cH / yLabels.length;

    ctx.clearRect(0, 0, W, H);
    data.forEach((row, yi) => {
      row.forEach((val, xi) => {
        const x = pad.left + xi * cellW, y = pad.top + yi * cellH;
        const hue = 260 - val * 120; // purple → green
        ctx.fillStyle = `hsla(${hue}, 70%, ${30 + val * 30}%, 0.9)`;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      });
    });
    yLabels.forEach((lbl, i) => {
      ctx.fillStyle = '#6366a0'; ctx.font = '8px JetBrains Mono';
      ctx.textAlign = 'right'; ctx.fillText(lbl, pad.left - 4, pad.top + i * cellH + cellH / 2 + 3);
    });
    xLabels.forEach((lbl, i) => {
      ctx.fillStyle = '#6366a0'; ctx.font = '8px JetBrains Mono';
      ctx.textAlign = 'center'; ctx.fillText(lbl, pad.left + i * cellW + cellW / 2, H - 6);
    });
  },

  drawTopologyAnim(canvasId, nodes) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const packets = [];
    const nodePos = nodes.map(n => {
      const el = document.getElementById(`node-cell-${n.id}`);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      const pr = canvas.getBoundingClientRect();
      return { 
        x: r.left - pr.left + r.width / 2, 
        y: r.top - pr.top + r.height / 2,
        active: n.selected
      };
    }).filter(p => p !== null);

    const hub = { x: canvas.width / 2, y: canvas.height + 50 }; // Hub is off-screen 'below' or central

    const loop = () => {
      if (document.getElementById(canvasId) !== canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const time = Date.now() * 0.002;

      nodePos.forEach(p => {
        if (!p.active) return;
        
        // Draw connection line
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.bezierCurveTo(p.x, p.y + 50, hub.x, hub.y - 100, hub.x, hub.y);
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.setLineDash([5, 5]);
        ctx.lineDashOffset = -time * 20;
        ctx.stroke();
        ctx.setLineDash([]);

        // Animated packets
        if (Math.random() < 0.05) packets.push({ t: 0, x: p.x, y: p.y });
      });

      for (let i = packets.length - 1; i >= 0; i--) {
        const pkt = packets[i];
        pkt.t += 0.01;
        if (pkt.t >= 1) { packets.splice(i, 1); continue; }
        
        // Bezier path for packet
        const t = pkt.t;
        const cx1 = pkt.x, cy1 = pkt.y + 50;
        const cx2 = hub.x, cy2 = hub.y - 100;
        const x = Math.pow(1-t, 3)*pkt.x + 3*Math.pow(1-t, 2)*t*cx1 + 3*(1-t)*Math.pow(t, 2)*cx2 + Math.pow(t, 3)*hub.x;
        const y = Math.pow(1-t, 3)*pkt.y + 3*Math.pow(1-t, 2)*t*cy1 + 3*(1-t)*Math.pow(t, 2)*cy2 + Math.pow(t, 3)*hub.y;

        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#8b5cf6';
        ctx.shadowBlur = 8;
        ctx.shadowColor = '#8b5cf6';
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      requestAnimationFrame(loop);
    };
    loop();
  },
};
