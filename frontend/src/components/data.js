// src/components/data.js
// 6G-ISAC Federated Transformer — Live simulation data store

const ISAC = {
  currentRound: 1,
  currentStep:  0,
  isRunning:    false,
  autoTimer:    null,

  // ── 16 ISAC Features ──────────────────────────────────────────────────
  featureGroups: {
    'Channel State (CSI)': [
      { key:'csi_magnitude', label:'CSI Magnitude',   unit:'', baseline:0.7 },
      { key:'csi_phase',     label:'CSI Phase',       unit:'rad', baseline:0 },
      { key:'doppler_shift', label:'Doppler Shift',   unit:'Hz', baseline:0 },
      { key:'multipath_delay',label:'Multipath Delay',unit:'ms', baseline:0.05 },
    ],
    'Signal Quality': [
      { key:'rssi_dbm', label:'RSSI',      unit:'dBm', baseline:-60 },
      { key:'snr_db',   label:'SNR',       unit:'dB',  baseline:25 },
      { key:'ber',      label:'Bit Error Rate', unit:'', baseline:0.01 },
    ],
    'Latency & Timing': [
      { key:'rtt_ms',    label:'Round-Trip Time', unit:'ms', baseline:10 },
      { key:'jitter_ms', label:'Jitter',          unit:'ms', baseline:2 },
    ],
    'Radar / Sensing': [
      { key:'radar_range_m', label:'Radar Range',   unit:'m',   baseline:150 },
      { key:'radar_velocity', label:'Radar Velocity',unit:'m/s', baseline:0 },
      { key:'radar_rcs',     label:'Radar RCS',     unit:'m²',  baseline:10 },
      { key:'angle_of_arrival',label:'AoA',         unit:'°',   baseline:180 },
    ],
    'Network Traffic': [
      { key:'throughput_mbps',  label:'Throughput',   unit:'Mbps', baseline:800 },
      { key:'packet_loss_rate', label:'Packet Loss',  unit:'%',    baseline:0.01 },
      { key:'active_users',     label:'Active Users', unit:'',     baseline:50 },
    ],
  },

  classNames: ['Normal', 'High Interference', 'Target Detected', 'Congestion'],
  classColors: ['#34d399','#fbbf24','#8b5cf6','#fb7185'],

  // ── Edge nodes ────────────────────────────────────────────────────────
  nodes: [
    { id:0, name:'BS-Alpha',   type:'base_station', icon:'🗼', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:500, snr:25, rssi:-60, throughput:800, rtt:10 },
    { id:1, name:'Sensor-01',  type:'iot_sensor',   icon:'📱', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:400, snr:22, rssi:-65, throughput:600, rtt:12 },
    { id:2, name:'BS-Bravo',   type:'base_station', icon:'🗼', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:500, snr:28, rssi:-55, throughput:900, rtt:8 },
    { id:3, name:'UAV-Relay',  type:'uav_relay',    icon:'🛩️', status:'idle', loss:0, acc:0, time:0, selected:false, samples:300, snr:18, rssi:-72, throughput:400, rtt:18 },
    { id:4, name:'Sensor-02',  type:'iot_sensor',   icon:'📱', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:450, snr:20, rssi:-68, throughput:550, rtt:14 },
    { id:5, name:'BS-Charlie', type:'base_station', icon:'🗼', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:500, snr:26, rssi:-58, throughput:850, rtt:9 },
    { id:6, name:'Vehicle-01', type:'v2x',          icon:'🚗', status:'idle', loss:0, acc:0, time:0, selected:false, samples:350, snr:15, rssi:-78, throughput:300, rtt:22 },
    { id:7, name:'Sensor-03',  type:'iot_sensor',   icon:'📱', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:400, snr:21, rssi:-66, throughput:580, rtt:13 },
    { id:8, name:'BS-Delta',   type:'base_station', icon:'🗼', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:500, snr:27, rssi:-56, throughput:880, rtt:9 },
    { id:9, name:'Radar-Hub',  type:'radar',        icon:'📡', status:'idle', loss:0, acc:0, time:0, selected:true,  samples:450, snr:24, rssi:-62, throughput:700, rtt:11 },
  ],

  // ── Metrics history ─────────────────────────────────────────────────
  metricsHistory: [
    { round:0, accuracy:0.25, loss:1.40, precision:0.24, recall:0.25, f1:0.23 }
  ],
  securityEvents: [],
  logs: [{ type:'info', msg:'6G-ISAC federated network initialised.' }],

  // ── Per-class accuracy over rounds ────────────────────────────────
  classAccHistory: [{ round:0, normal:0.30, interference:0.20, target:0.22, congestion:0.28 }],

  // ── Research Sustainability & Overhead ──────────────────────────
  energyConsumed: 42.5, // mWh
  commOverhead:   12.8, // MB
  dataSkew: [ // Heatmap data: row per node, column per class
    [0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.8, 0], [0.1, 0.1, 0.1, 0.7],
    [0.7, 0.1, 0.1, 0.1], [0.1, 0.8, 0.05, 0.05], [0, 0.2, 0.7, 0.1], [0.2, 0, 0.1, 0.7],
    [0.9, 0.05, 0.05, 0], [0.1, 0.1, 0.1, 0.7]
  ],

  // ── Signal snapshots (simulated live readings) ─────────────────────
  signalSnapshot: {
    csi_magnitude: 0.72, csi_phase: 1.23, doppler_shift: 0.05, multipath_delay: 0.06,
    rssi_dbm: -61, snr_db: 24.5, ber: 0.012,
    rtt_ms: 11.2, jitter_ms: 2.8,
    radar_range_m: 152, radar_velocity: 1.2, radar_rcs: 10.5, angle_of_arrival: 182,
    throughput_mbps: 785, packet_loss_rate: 0.013, active_users: 48,
  },

  // ── XAI / feature importance (SHAP values) ──────────────────────────
  featureImportance: {
    csi_magnitude:0.73, rssi_dbm:0.65, snr_db:0.58, radar_velocity:0.52,
    throughput_mbps:0.48, doppler_shift:0.42, rtt_ms:0.38, packet_loss_rate:0.35,
    radar_rcs:0.31, jitter_ms:0.27, active_users:0.23, ber:0.19,
    angle_of_arrival:0.15, multipath_delay:0.12, radar_range_m:0.08, csi_phase:0.05,
  },

  // ── Transformer config ──────────────────────────────────────────────
  transformerConfig: {
    input_dim:16, d_model:64, num_heads:4, num_layers:2, d_ff:128,
    num_classes:4, seq_len:32, dropout:0.1, params:'42,244',
  },

  // ── Training steps ──────────────────────────────────────────────────
  steps: [
    { title:'Initialise network',     desc:'Register 10 edge nodes across the 6G coverage area.' },
    { title:'Select clients',         desc:'Choose clients_per_round nodes based on channel quality.' },
    { title:'Broadcast global model', desc:'Distribute ISACTransformer weights (64-dim, 4-head) to nodes.' },
    { title:'Local training',         desc:'Each node trains on private CSI/radar/traffic data (E epochs).' },
    { title:'SHAP explanations',      desc:'Kernel SHAP attribution across all 16 ISAC features per node.' },
    { title:'Validate updates',       desc:'Peer validation: accuracy + XAI coherence check.' },
    { title:'Encrypt weights',        desc:'AES-256-GCM encryption with per-round session keys.' },
    { title:'Anomaly detection',      desc:'Cosine similarity check: sim(w_global, w_local) ≥ threshold.' },
    { title:'FedAvg aggregation',     desc:'w_{t+1} = Σ (n_i/N) · w_i — weighted by node sample count.' },
    { title:'Evaluate global model',  desc:'Test on held-out 6G-ISAC dataset: accuracy, F1, loss.' },
    { title:'Distribute & log',       desc:'Updated model broadcast; CSV metrics logged for analysis.' },
  ],

  // ── Helpers ─────────────────────────────────────────────────────────
  log(msg, type='info') {
    const ts = new Date().toLocaleTimeString('en',{hour12:false});
    this.logs.unshift({ type, msg: `[${ts}] ${msg}` });
    if (this.logs.length > 80) this.logs.pop();
  },

  // ── Real-time step handler (from Python Backend) ─────────────────────
  handleRealStep(data) {
    const s = data.step;
    const r = this.currentRound;
    this.currentStep = s;

    if (s === 'done') {
      this.isRunning = false;
      return;
    }

    if (s === 1) {
      this.log(`Network Initialised — PyTorch models ready on device`, 'ok');
    }
    else if (s === 2) {
      this.nodes.forEach(n => { n.selected = false; n.status = 'standby'; });
      if (data.selected) {
        data.selected.forEach(cid => {
          const node = this.nodes.find(n => n.id === cid);
          if (node) { node.selected = true; node.status = 'selected'; }
        });
        this.log(`${data.selected.length} nodes selected for Round ${r}`, 'ok');
      }
    }
    else if (s === 3) {
      this.nodes.filter(n => n.selected).forEach(n => {
        n.status = 'syncing';
        this._incrementalLogOnce(`broadcast_r${r}`, `Broadcasting global weights to participating nodes...`, 'info');
      });
    }
    else if (s === 4) {
      if (data.results) {
        data.results.forEach(res => {
          const node = this.nodes.find(n => n.id === res.client_id);
          if (node) {
            node.status = 'trained';
            node.loss = res.local_loss;
            node.acc = res.local_accuracy;
            node.time = res.training_time_s;
            this.log(`${node.name}: Trained | Loss: ${node.loss.toFixed(4)} | Acc: ${(node.acc * 100).toFixed(1)}%`, 'ok');
          }
        });
      }
      // Randomise signal snapshot for visual variety (since it's not strictly from backend)
      Object.keys(this.signalSnapshot).forEach(k => {
        this.signalSnapshot[k] = +(this.signalSnapshot[k] * (0.98 + Math.random()*0.04)).toFixed(3);
      });
    }
    else if (s === 5) {
      this.log('SHAP Attribution: Analysing feature importance for all nodes...', 'info');
      if (data.explanations) {
        // Just take the first one for the summary log
        const firstCid = Object.keys(data.explanations)[0];
        if (firstCid) {
          const feats = data.explanations[firstCid].top_features.slice(0,3).join(', ');
          this.log(`Primary Drivers: ${feats}`, 'warn');
        }
      }
    }
    else if (s === 6) {
      if (data.validated) {
        this.nodes.filter(n => n.selected).forEach(n => {
          if (data.validated.includes(n.id)) {
            n.status = 'validated';
            this.log(`${n.name}: ✓ Validated (performance + XAI coherent)`, 'ok');
          } else {
            n.status = 'rejected';
            this.log(`${n.name}: ✗ Rejected (low performance/coherence)`, 'err');
          }
        });
      }
    }
    else if (s === 7) {
      this.log('AES-256-GCM: Encrypting model updates with session keys...', 'info');
    }
    else if (s === 8) {
      if (data.similarities) {
        data.similarities.forEach(sim => {
          const node = this.nodes.find(n => n.id === sim.client_id);
          this.securityEvents.unshift({
            round: r, 
            node: node ? node.name : `ID-${sim.client_id}`, 
            similarity: sim.sim.toFixed(4), 
            status: sim.status, 
            time: new Date().toLocaleTimeString()
          });
          this.log(`${node ? node.name : sim.client_id}: Sim=${sim.sim.toFixed(4)} → ${sim.status}`, sim.status === 'CLEAN' ? 'ok' : 'err');
        });
      }
    }
    else if (s === 9) {
      this.log(`FedAvg: Aggregating local updates into global model...`, 'ok');
    }
    else if (s === 10) {
      if (data.metrics) {
        const m = data.metrics;
        this.metricsHistory.push({
          round: r,
          accuracy: m.accuracy,
          loss: m.loss,
          precision: m.precision,
          recall: m.recall,
          f1: m.f1_score
        });
        // Mock per-class split based on global accuracy for heatmap
        const cprev = this.classAccHistory[this.classAccHistory.length - 1];
        this.classAccHistory.push({
          round: r,
          normal:      +Math.min(0.98, m.accuracy + (Math.random()*0.05)).toFixed(4),
          interference:+Math.min(0.95, m.accuracy - (Math.random()*0.02)).toFixed(4),
          target:      +Math.min(0.94, m.accuracy - (Math.random()*0.03)).toFixed(4),
          congestion:  +Math.min(0.95, m.accuracy - (Math.random()*0.01)).toFixed(4),
        });
        this.log(`Evaluation Round ${r}: Acc=${(m.accuracy*100).toFixed(2)}% | Loss=${m.loss.toFixed(4)}`, 'ok');
      }
    }
    else if (s === 11) {
      this.log(`Updated model distributed to entire network.`, 'ok');
      this.currentRound++;
      this.currentStep = 0;
      document.getElementById('sidebar-round').textContent = `Round ${this.currentRound}`;
    }
  },

  _incrementalLogOnce(key, msg, type) {
     if (this._lastLogKey === key) return;
     this.log(msg, type);
     this._lastLogKey = key;
  },

  runStep() {
    // Deprecated for WebSocket flow
  }
};
