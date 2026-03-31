// src/components/pages.js
// 6G-ISAC Federated Transformer — unique page renderers

const Pages = {

  // ═══════════════════════════════════════════════════════════════════════
  //  DASHBOARD
  // ═══════════════════════════════════════════════════════════════════════
  dashboard() {
    const m = ISAC.metricsHistory[ISAC.metricsHistory.length - 1];
    const prev = ISAC.metricsHistory.length > 1 ? ISAC.metricsHistory[ISAC.metricsHistory.length - 2] : m;
    const dAcc = ((m.accuracy - prev.accuracy)*100).toFixed(1);
    const dF1  = ((m.f1 - prev.f1)*100).toFixed(1);
    const dLoss= (m.loss - prev.loss).toFixed(3);
    const selNodes = ISAC.nodes.filter(n=>n.selected).length;
    const sig = ISAC.signalSnapshot;

    return `
    <div class="page-header">
      <div>
        <div class="page-title">Dashboard</div>
        <div class="page-subtitle">FedTransformer · 6G Integrated Sensing & Communication</div>
      </div>
      <div class="page-controls">
        <button class="btn btn-primary" onclick="App.runStep()">▶ Step</button>
        <button class="btn" onclick="App.toggleAuto()" id="auto-btn">${ISAC.isRunning?'⏹ Stop':'⏩ Auto'}</button>
        <button class="btn" onclick="App.reset()">↺ Reset</button>
      </div>
    </div>

    <div class="sim-bar">
      <span class="sim-label">Round ${ISAC.currentRound} ·</span>
      <div class="sim-step-dots">
        ${ISAC.steps.map((s,i) => `<div class="sim-dot ${i<ISAC.currentStep?'done':i===ISAC.currentStep?'active':''}" title="${s.title}">${i+1}</div>`).join('')}
      </div>
      <span class="sim-label">${ISAC.steps[Math.min(ISAC.currentStep,10)].title}</span>
    </div>

    <!-- KPIs -->
    <div class="grid-4" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 24px;">
      <div class="card"><div class="card-label">Avg Accuracy</div><div class="card-value" style="color:var(--violet)">${(m.accuracy*100).toFixed(1)}%</div><div class="card-delta ${+dAcc>=0?'delta-up':'delta-down'}">${+dAcc>=0?'▲':'▼'} ${Math.abs(dAcc)}%</div></div>
      <div class="card"><div class="card-label">Global Loss</div><div class="card-value" style="color:var(--rose)">${m.loss.toFixed(3)}</div><div class="card-delta ${+dLoss<=0?'delta-up':'delta-down'}">${+dLoss<=0?'▼':'▲'} ${Math.abs(dLoss)}</div></div>
      <div class="card"><div class="card-label">Energy Contrib.</div><div class="card-value" style="color:var(--amber)">${ISAC.energyConsumed.toFixed(1)}<span style="font-size:12px"> mWh</span></div><div class="card-delta">estimated 6G footprint</div></div>
      <div class="card"><div class="card-label">Comm. Overhead</div><div class="card-value" style="color:var(--sky)">${ISAC.commOverhead.toFixed(1)}<span style="font-size:12px"> MB</span></div><div class="card-delta">total sync traffic</div></div>
    </div>

    <div class="grid-3">
      <div class="card"><div class="card-label">F1 Score</div><div class="card-value" style="color:var(--teal)">${m.f1.toFixed(3)}</div><div class="card-delta ${+dF1>=0?'delta-up':'delta-down'}">${+dF1>=0?'▲':'▼'} ${Math.abs(dF1)}%</div></div>
      <div class="card"><div class="card-label">Active Nodes</div><div class="card-value" style="color:var(--sky)">${selNodes}<span style="font-size:14px;color:var(--muted)">/${ISAC.nodes.length}</span></div><div class="card-delta">${ISAC.nodes.filter(n=>n.type==='base_station').length} base stations</div></div>
      <div class="card"><div class="card-label">Avg SNR</div><div class="card-value" style="color:var(--amber)">${(ISAC.nodes.reduce((s,n)=>s+n.snr,0)/ISAC.nodes.length).toFixed(1)}<span style="font-size:12px"> dB</span></div><div class="card-delta">across all nodes</div></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Model performance</div>
        <canvas id="c-metrics" width="500" height="180"></canvas>
        <div style="display:flex;gap:14px;margin-top:8px;font-size:10px">
          <span style="color:var(--violet)">● Accuracy</span>
          <span style="color:var(--teal)">● F1</span>
          <span style="color:var(--rose)">● Loss</span>
          <span style="color:var(--amber)">● Precision</span>
          <span style="color:var(--sky)">● Recall</span>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Per-class accuracy</div>
        <canvas id="c-class" width="500" height="180"></canvas>
        <div style="display:flex;gap:14px;margin-top:8px;font-size:10px">
          ${ISAC.classNames.map((cn,i)=>`<span style="color:${ISAC.classColors[i]}">● ${cn}</span>`).join('')}
        </div>
      </div>
    </div>

    <div class="grid-2">
      <!-- Live signal readings -->
      <div class="card">
        <div class="card-title">Live 6G-ISAC signal readings</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px">
          ${[
            {k:'snr_db',l:'SNR',u:'dB',c:'var(--violet)'},
            {k:'rssi_dbm',l:'RSSI',u:'dBm',c:'var(--sky)'},
            {k:'throughput_mbps',l:'Throughput',u:'Mbps',c:'var(--teal)'},
            {k:'rtt_ms',l:'RTT',u:'ms',c:'var(--amber)'},
            {k:'csi_magnitude',l:'CSI Mag',u:'',c:'var(--rose)'},
            {k:'ber',l:'BER',u:'',c:'var(--emerald)'},
            {k:'jitter_ms',l:'Jitter',u:'ms',c:'var(--indigo)'},
            {k:'packet_loss_rate',l:'Pkt Loss',u:'%',c:'var(--amber)'},
            {k:'radar_range_m',l:'Radar Range',u:'m',c:'var(--violet)'},
            {k:'radar_velocity',l:'Velocity',u:'m/s',c:'var(--sky)'},
            {k:'doppler_shift',l:'Doppler',u:'Hz',c:'var(--teal)'},
            {k:'active_users',l:'Users',u:'',c:'var(--rose)'},
          ].map(f=>`
            <div style="background:var(--bg3);border-radius:6px;padding:6px 8px">
              <div style="font-size:8.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em">${f.l}</div>
              <div style="font-family:var(--font-display);font-weight:700;font-size:14px;color:${f.c}">${typeof sig[f.k]==='number'?sig[f.k].toFixed?sig[f.k].toFixed(sig[f.k]>100?0:sig[f.k]>1?1:3):sig[f.k]:sig[f.k]}<span style="font-size:9px;color:var(--muted)"> ${f.u}</span></div>
            </div>
          `).join('')}
        </div>
      </div>
      <!-- Event log -->
      <div class="card">
        <div class="card-title">Event log</div>
        <div class="terminal">${ISAC.logs.map(l=>`<div class="log-${l.type}">${l.msg}</div>`).join('')}</div>
      </div>
    </div>`;
  },

  // ═══════════════════════════════════════════════════════════════════════
  //  NETWORK TOPOLOGY
  // ═══════════════════════════════════════════════════════════════════════
  topology() {
    const types = {base_station:'🗼 Base Station', iot_sensor:'📱 IoT Sensor', uav_relay:'🛩️ UAV Relay', v2x:'🚗 V2X Vehicle', radar:'📡 Radar Hub'};
    return `
    <div class="page-header">
      <div>
        <div class="page-title">Network Topology</div>
        <div class="page-subtitle">6G edge node roster — signal quality, throughput & latency per node</div>
      </div>
    </div>

    <div style="position:relative; margin-bottom: 24px;">
      <canvas id="c-topo-bg" style="position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:0;"></canvas>
      <div class="grid-3" id="topo-grid" style="position:relative; z-index:1;">
        ${ISAC.nodes.map(n => `
          <div class="node-card" id="node-cell-${n.id}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
              <div>
                <div style="display:flex;align-items:center;gap:6px">
                  <span style="font-size:18px">${n.icon}</span>
                  <div class="node-name">${n.name}</div>
                </div>
                <div class="node-type">${types[n.type] || n.type} · Client ${n.id}</div>
              </div>
              <div style="display:flex;flex-direction:column;gap:3px;align-items:flex-end">
                ${n.selected?'<span class="badge badge-violet">Selected</span>':'<span class="badge badge-muted">Standby</span>'}
                <span class="badge ${this._statusBadge(n.status)}">${n.status}</span>
              </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;font-size:10.5px">
              <div style="background:var(--bg3);border-radius:4px;padding:4px 6px"><span style="color:var(--muted)">SNR</span><div style="color:var(--violet);font-weight:600">${n.snr} dB</div></div>
              <div style="background:var(--bg3);border-radius:4px;padding:4px 6px"><span style="color:var(--muted)">RSSI</span><div style="color:var(--sky);font-weight:600">${n.rssi} dBm</div></div>
              <div style="background:var(--bg3);border-radius:4px;padding:4px 6px"><span style="color:var(--muted)">RTT</span><div style="color:var(--amber);font-weight:600">${n.rtt} ms</div></div>
            </div>
            <div style="margin-top:6px;font-size:10px;color:var(--muted)">Samples: ${n.samples} · Training: ${n.time?n.time+'s':'—'}</div>
          </div>
        `).join('')}
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Node accuracy after training</div>
        <canvas id="c-node-acc" width="500" height="160"></canvas>
      </div>
      <div class="card">
        <div class="card-title">Node type distribution</div>
        <canvas id="c-type-donut" width="500" height="160"></canvas>
      </div>
    </div>`;
  },

  // ═══════════════════════════════════════════════════════════════════════
  //  SIGNAL ANALYSIS (replaces blockchain)
  // ═══════════════════════════════════════════════════════════════════════
  signals() {
    const sig = ISAC.signalSnapshot;
    const groups = ISAC.featureGroups;

    return `
    <div class="page-header">
      <div>
        <div class="page-title">Signal Analysis</div>
        <div class="page-subtitle">All 16 ISAC features — channel state, signal quality, radar, timing & traffic</div>
      </div>
    </div>

    <!-- Waveforms -->
    <div class="grid-3" style="margin-bottom:16px">
      <div class="card">
        <div class="card-title">CSI waveform</div>
        <canvas id="c-wave-csi" width="300" height="70"></canvas>
      </div>
      <div class="card">
        <div class="card-title">Radar return signal</div>
        <canvas id="c-wave-radar" width="300" height="70"></canvas>
      </div>
      <div class="card">
        <div class="card-title">Throughput signal</div>
        <canvas id="c-wave-thru" width="300" height="70"></canvas>
      </div>
    </div>

    <!-- Feature groups -->
    ${Object.entries(groups).map(([group, features]) => `
      <div class="card" style="margin-bottom:12px">
        <div class="card-title">${group}</div>
        <div style="display:grid;grid-template-columns:repeat(${features.length},1fr);gap:8px">
          ${features.map(f => {
            const val = sig[f.key];
            const display = typeof val === 'number' ? (Math.abs(val) > 100 ? val.toFixed(0) : Math.abs(val) > 1 ? val.toFixed(1) : val.toFixed(3)) : val;
            return `
            <div style="background:var(--bg3);border-radius:8px;padding:10px 12px">
              <div style="font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">${f.label}</div>
              <div style="font-family:var(--font-display);font-weight:700;font-size:20px;color:var(--text)">${display}<span style="font-size:10px;color:var(--muted)"> ${f.unit}</span></div>
            </div>`;
          }).join('')}
        </div>
      </div>
    `).join('')}

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Feature importance (SHAP)</div>
        ${Object.entries(ISAC.featureImportance).slice(0,10).map(([k,v]) => `
          <div class="feat-row">
            <span class="feat-label">${k.replace(/_/g,' ')}</span>
            <div class="prog-wrap"><div class="prog-bar" style="width:${v*100}%;background:linear-gradient(90deg,var(--violet),var(--indigo))"></div></div>
            <span class="feat-val">${(v*100).toFixed(0)}%</span>
          </div>
        `).join('')}
      </div>
      <div class="card">
        <div class="card-title">Feature radar — 6G-ISAC spectrum</div>
        <canvas id="c-feat-radar" width="400" height="260"></canvas>
      </div>
    </div>`;
  },

  // ═══════════════════════════════════════════════════════════════════════
  //  TRAINING
  // ═══════════════════════════════════════════════════════════════════════
  training() {
    return `
    <div class="page-header">
      <div>
        <div class="page-title">Training</div>
        <div class="page-subtitle">11-step federated pipeline — Transformer + FedAvg + AES-256-GCM</div>
      </div>
      <div class="page-controls"><button class="btn btn-primary" onclick="App.runStep()">▶ Next step</button></div>
    </div>

    <div class="grid-2-1">
      <div class="card">
        <div class="card-title">Round ${ISAC.currentRound} — Pipeline</div>
        <div style="display:flex;flex-direction:column;gap:0">
          ${ISAC.steps.map((s,i) => {
            const done=i<ISAC.currentStep, active=i===ISAC.currentStep;
            return `
            <div style="display:flex;gap:12px">
              <div style="display:flex;flex-direction:column;align-items:center">
                <div style="width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:3px;border:2px solid ${done?'var(--violet)':active?'var(--amber)':'var(--border2)'};background:${done?'var(--violet)':active?'var(--amber)':'var(--bg3)'};${done||active?'box-shadow:0 0 8px '+(done?'var(--violet)':'var(--amber)'):''};"></div>
                ${i<ISAC.steps.length-1?'<div style="width:1px;flex:1;background:var(--border);min-height:16px;margin:2px 0"></div>':''}
              </div>
              <div style="padding-bottom:14px">
                <div style="font-size:11.5px;font-weight:500;color:${done?'var(--violet)':active?'var(--amber)':'var(--text)'}">
                  ${i+1}. ${s.title}
                  ${done?'<span class="badge badge-violet" style="margin-left:5px">done</span>':''}
                  ${active?'<span class="badge badge-amber" style="margin-left:5px">running</span>':''}
                </div>
                <div style="font-size:10px;color:var(--muted)">${s.desc}</div>
              </div>
            </div>`;
          }).join('')}
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:12px">
        <div class="card">
          <div class="card-title">Per-node loss</div>
          <canvas id="c-node-loss" width="280" height="130"></canvas>
        </div>
        <div class="card">
          <div class="card-title">Equations</div>
          <div class="eq-box">w_{t+1} = Σ (n_i/N) · w_i</div>
          <div style="font-size:9px;color:var(--muted);margin-bottom:6px">FedAvg — weighted aggregation</div>
          <div class="eq-box">Attn(Q,K,V) = softmax(QKᵀ/√d_k)·V</div>
          <div style="font-size:9px;color:var(--muted);margin-bottom:6px">Scaled dot-product attention</div>
          <div class="eq-box">L = -(1/m) Σ y·log(ŷ)</div>
          <div style="font-size:9px;color:var(--muted);margin-bottom:6px">Cross-entropy loss</div>
          <div class="eq-box">sim = (w_g · w_l) / (‖w_g‖ · ‖w_l‖)</div>
          <div style="font-size:9px;color:var(--muted)">Cosine similarity (poisoning check)</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">Event log</div>
      <div class="terminal">${ISAC.logs.slice(0,25).map(l=>`<div class="log-${l.type}">${l.msg}</div>`).join('')}</div>
    </div>`;
  },

  // ═══════════════════════════════════════════════════════════════════════
  //  SECURITY
  // ═══════════════════════════════════════════════════════════════════════
  security() {
    const evts = ISAC.securityEvents;
    const clean = evts.filter(e=>e.status==='CLEAN');
    return `
    <div class="page-header">
      <div>
        <div class="page-title">Security</div>
        <div class="page-subtitle">AES-256-GCM encrypted transmission · Cosine similarity anomaly detection</div>
      </div>
    </div>

    <div class="grid-4">
      <div class="card"><div class="card-label">Checked</div><div class="card-value">${evts.length}</div><div class="card-delta">update checks</div></div>
      <div class="card"><div class="card-label">Accepted</div><div class="card-value" style="color:var(--emerald)">${clean.length}</div><div class="card-delta">clean updates</div></div>
      <div class="card"><div class="card-label">Threshold</div><div class="card-value" style="color:var(--amber)">0.50</div><div class="card-delta">cosine similarity</div></div>
      <div class="card"><div class="card-label">Encryption</div><div class="card-value" style="color:var(--violet);font-size:18px">AES-256-GCM</div><div class="card-delta">per-round keys</div></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Update similarity checks</div>
        ${evts.length===0?'<div style="color:var(--muted);font-size:11px">No security events yet — run rounds to see results.</div>':`
        <table class="tbl">
          <thead><tr><th>Round</th><th>Node</th><th>Similarity</th><th>Result</th><th>Time</th></tr></thead>
          <tbody>${evts.slice(0,20).map(e=>`
            <tr>
              <td>R${e.round}</td>
              <td style="font-family:var(--font-display);font-weight:700">${e.node}</td>
              <td><div style="display:flex;align-items:center;gap:6px">
                <div class="prog-wrap" style="max-width:70px"><div class="prog-bar" style="width:${e.similarity*100}%;background:${e.similarity>=0.5?'var(--emerald)':'var(--rose)'}"></div></div>
                <span style="font-size:10px">${e.similarity}</span>
              </div></td>
              <td><span class="badge ${e.status==='CLEAN'?'badge-emerald':'badge-rose'}">${e.status}</span></td>
              <td style="color:var(--muted)">${e.time}</td>
            </tr>`).join('')}
          </tbody>
        </table>`}
      </div>
      <div class="card">
        <div class="card-title">Security pipeline</div>
        <div style="display:flex;flex-direction:column;gap:10px;font-size:11.5px">
          ${[
            {t:'AES-256-GCM encryption',d:'Model weights encrypted before transmission'},
            {t:'Per-round session keys',d:'Fresh 256-bit key generated each communication round'},
            {t:'Authentication tag',d:'GCM authentication tag detects any payload tampering'},
            {t:'Nonce isolation',d:'Unique nonce per message prevents replay attacks'},
            {t:'Cosine similarity filter',d:'Model updates compared to global weights; rejection if sim < τ'},
            {t:'XAI coherence check',d:'SHAP explanations must be non-trivial (max|SHAP| > 0.001)'},
          ].map(item=>`
            <div style="display:flex;gap:8px;align-items:flex-start">
              <span style="color:var(--emerald);font-size:13px;margin-top:1px">✓</span>
              <div><div style="color:var(--text);font-weight:500">${item.t}</div><div style="font-size:10px;color:var(--muted)">${item.d}</div></div>
            </div>`).join('')}
        </div>
      </div>
    </div>`;
  },

  // ═══════════════════════════════════════════════════════════════════════
  //  MODEL INSIGHTS (replaces XAI page)
  // ═══════════════════════════════════════════════════════════════════════
  model() {
    const tc = ISAC.transformerConfig;
    const m = ISAC.metricsHistory[ISAC.metricsHistory.length - 1];
    const ca = ISAC.classAccHistory[ISAC.classAccHistory.length - 1];

    return `
    <div class="page-header">
      <div>
        <div class="page-title">Model Insights</div>
        <div class="page-subtitle">ISACTransformer architecture · SHAP explainability · class-level analysis</div>
      </div>
    </div>

    <!-- Architecture -->
    <div class="grid-2" style="margin-bottom:16px">
      <div class="card">
        <div class="card-title">ISACTransformer architecture</div>
        <table class="tbl">
          <tbody>
            ${Object.entries({
              'Input dimension':tc.input_dim+' features','Embedding (d_model)':tc.d_model,
              'Attention heads':tc.num_heads,'Encoder layers':tc.num_layers,
              'FFN dimension':tc.d_ff,'Sequence length':tc.seq_len+' time steps',
              'Dropout':tc.dropout,'Output classes':tc.num_classes+' ('+ISAC.classNames.join(', ')+')',
              'Total parameters':tc.params,
            }).map(([k,v])=>`<tr><td style="color:var(--muted)">${k}</td><td style="font-weight:600">${v}</td></tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="card-title">Classification breakdown (latest)</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:14px">
          ${ISAC.classNames.map((cn,i)=>`
            <div style="background:var(--bg3);border-radius:8px;padding:10px;border-left:3px solid ${ISAC.classColors[i]}">
              <div style="font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">${cn}</div>
              <div style="font-family:var(--font-display);font-size:22px;font-weight:700;color:${ISAC.classColors[i]}">${(ca[['normal','interference','target','congestion'][i]]*100).toFixed(1)}%</div>
            </div>`).join('')}
        </div>
        <div class="card-title">Per-class accuracy heatmap</div>
        <canvas id="c-heatmap" width="500" height="120"></canvas>
      </div>
    </div>

    <!-- XAI -->
    <div class="grid-2">
      <div class="card">
        <div class="card-title">SHAP feature importance — all 16 ISAC features</div>
        ${Object.entries(ISAC.featureImportance).map(([k,v],i) => {
          const barColors = ['var(--violet)','var(--indigo)','var(--sky)','var(--teal)','var(--emerald)','var(--amber)','var(--rose)','var(--violet)'];
          return `
          <div class="feat-row">
            <span class="feat-label">${k.replace(/_/g,' ')}</span>
            <div class="prog-wrap"><div class="prog-bar" style="width:${v*100}%;background:${barColors[i%barColors.length]}"></div></div>
            <span class="feat-val">${(v*100).toFixed(0)}%</span>
          </div>`;
        }).join('')}
      </div>
      <div class="card">
        <div class="card-title">How XAI works in this system</div>
        <div style="font-size:11px;color:var(--muted);line-height:2">
          <div><span style="color:var(--violet);font-weight:700">1.</span> Each selected node runs Kernel SHAP on a batch of local 6G-ISAC samples.</div>
          <div><span style="color:var(--violet);font-weight:700">2.</span> SHAP values quantify how each of the 16 features (CSI, RSSI, radar, traffic, etc.) shifts the Transformer's prediction.</div>
          <div><span style="color:var(--violet);font-weight:700">3.</span> Top features are logged and stored in per-round metrics.</div>
          <div><span style="color:var(--violet);font-weight:700">4.</span> Peer validators check that the explanation is non-trivial (max|SHAP| > threshold).</div>
          <div><span style="color:var(--violet);font-weight:700">5.</span> Incoherent explanations cause the update to be <span class="badge badge-rose">rejected</span>.</div>
        </div>
        <div style="margin-top:12px;background:var(--bg3);border-radius:8px;padding:10px;border-left:3px solid var(--violet)">
          <div style="font-size:10.5px;font-weight:500;color:var(--amber);margin-bottom:3px">⚠ Network anomaly detected</div>
          <div style="font-size:10px;color:var(--muted)">Primary driver: CSI magnitude (73%). Secondary: RSSI drop (65%). Radar velocity contributed 52%.</div>
        </div>
        <div style="margin-top:12px">
          <div class="card-title">Network state classes</div>
          <table class="tbl">
            <thead><tr><th>Label</th><th>Class</th><th>Description</th></tr></thead>
            <tbody>
              ${ISAC.classNames.map((cn,i)=>`
                <tr>
                  <td style="color:${ISAC.classColors[i]};font-weight:700">${i}</td>
                  <td style="font-family:var(--font-display);font-weight:600">${cn}</td>
                  <td style="color:var(--muted)">${['Typical 6G operation','Elevated noise / RF jamming','Radar target / sensing event','Network overload / congestion'][i]}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>`;
  },

  // ── Helpers ─────────────────────────────────────────────────────────
  _statusBadge(s) {
    return {idle:'badge-muted',selected:'badge-indigo',syncing:'badge-sky',trained:'badge-violet',validated:'badge-emerald',synced:'badge-teal',standby:'badge-muted'}[s]||'badge-muted';
  },

  afterRender(page) {
    if (page === 'dashboard') {
      const rounds = ISAC.metricsHistory.map(m=>m.round===0?'Init':`R${m.round}`);
      Charts.drawLine('c-metrics',[
        {data:ISAC.metricsHistory.map(m=>m.accuracy),color:'#8b5cf6'},
        {data:ISAC.metricsHistory.map(m=>m.f1),color:'#2dd4bf'},
        {data:ISAC.metricsHistory.map(m=>m.loss),color:'#fb7185'},
        {data:ISAC.metricsHistory.map(m=>m.precision),color:'#fbbf24'},
        {data:ISAC.metricsHistory.map(m=>m.recall),color:'#38bdf8'},
      ], rounds);
      Charts.drawLine('c-class',[
        {data:ISAC.classAccHistory.map(c=>c.normal),color:'#34d399'},
        {data:ISAC.classAccHistory.map(c=>c.interference),color:'#fbbf24'},
        {data:ISAC.classAccHistory.map(c=>c.target),color:'#8b5cf6'},
        {data:ISAC.classAccHistory.map(c=>c.congestion),color:'#fb7185'},
      ], ISAC.classAccHistory.map(c=>c.round===0?'Init':`R${c.round}`));
    }
    if (page === 'topology') {
      const canvas = document.getElementById('c-topo-bg');
      if (canvas) {
        // Simple delay to let DOM render so we can get card positions
        setTimeout(() => Charts.drawTopologyAnim('c-topo-bg', ISAC.nodes), 100);
      }
      const trained = ISAC.nodes.filter(n=>n.acc>0);
      if (trained.length) Charts.drawBars('c-node-acc',trained.map(n=>n.acc),trained.map(n=>n.name.substring(0,6)),trained.map(()=>'#8b5cf6'));
      const types = {};
      ISAC.nodes.forEach(n=>{types[n.type]=(types[n.type]||0)+1;});
      const typeColors = {base_station:'#8b5cf6',iot_sensor:'#2dd4bf',uav_relay:'#fbbf24',v2x:'#fb7185',radar:'#38bdf8'};
      Charts.drawDonut('c-type-donut',Object.entries(types).map(([t,c])=>({value:c,color:typeColors[t]||'#6366f1'})),ISAC.nodes.length);
    }
    if (page === 'signals') {
      Charts.drawWaveform('c-wave-csi','#8b5cf6',1.2,0.8);
      Charts.drawWaveform('c-wave-radar','#fb7185',2.5,0.5);
      Charts.drawWaveform('c-wave-thru','#2dd4bf',0.8,0.6);
      const topFeatures = Object.entries(ISAC.featureImportance).slice(0,8);
      Charts.drawRadar('c-feat-radar',topFeatures.map(([,v])=>v),topFeatures.map(([k])=>k),'#8b5cf6');
    }
    if (page === 'training') {
      const trained = ISAC.nodes.filter(n=>n.loss>0);
      if (trained.length) Charts.drawBars('c-node-loss',trained.map(n=>n.loss),trained.map(n=>n.name.substring(0,6)),trained.map(()=>'#fb7185'));
    }
    if (page === 'model') {
      const data = ISAC.dataSkew;
      const xLabels = ISAC.classNames;
      const yLabels = ISAC.nodes.map(n => n.name.substring(0, 8));
      Charts.drawHeatmap('c-heatmap', data, xLabels, yLabels);
    }
  }
};
