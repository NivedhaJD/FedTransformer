// src/components/app.js
// App controller — navigation, step simulation, rendering

let currentPage = 'dashboard';

let ws = null;

const App = {
  init() { 
    this.render('dashboard'); 
    this.connectWs();
    Charts.startAnims();
  },

  connectWs() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => {
        ISAC.log('WebSocket connected to PyTorch backend API', 'ok');
        this.render(currentPage);
    };
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleBackendEvent(data);
    };
    ws.onclose = () => {
      ISAC.log('WebSocket disconnected. Reconnecting in 3s...', 'warn');
      setTimeout(() => this.connectWs(), 3000);
    };
  },

  handleBackendEvent(data) {
    if (data.action === 'training_complete') {
        ISAC.isRunning = false;
        ISAC.log(`Training completed successfully. All rounds done.`, 'ok');
        const btn = document.getElementById('auto-btn');
        if (btn) btn.textContent = 'Training Finished';
        this.render(currentPage);
        return;
    }
    
    if (data.step) {
        ISAC.handleRealStep(data);
        this.render(currentPage);
    }
  },

  render(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const pageEl = document.getElementById(`page-${page}`);
    const navEl  = document.querySelector(`[data-page="${page}"]`);
    if (!pageEl) return;
    pageEl.innerHTML = Pages[page]();
    pageEl.classList.add('active');
    if (navEl) navEl.classList.add('active');
    requestAnimationFrame(() => Pages.afterRender(page));
    currentPage = page;
  },

  runStep() {
    this.startTraining();
  },

  toggleAuto() {
    this.startTraining();
  },
  
  startTraining() {
    if (ISAC.isRunning) return;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ISAC.log(`Sent START signal to PyTorch backend`, 'info');
        ws.send(JSON.stringify({ action: 'start' }));
        ISAC.isRunning = true;
        const btn = document.getElementById('auto-btn');
        if (btn) btn.textContent = 'Training...';
    } else {
        alert("Not connected to backend API");
    }
    this.render(currentPage);
  },

  reset() {
    ISAC.isRunning = false;
    ISAC.currentRound = 1;
    ISAC.currentStep = 0;
    ISAC.metricsHistory = [{ round:0, accuracy:0.25, loss:1.40, precision:0.24, recall:0.25, f1:0.23 }];
    ISAC.classAccHistory = [{ round:0, normal:0.30, interference:0.20, target:0.22, congestion:0.28 }];
    ISAC.securityEvents = [];
    ISAC.logs = [{ type:'info', msg:'System configuration reset.' }];
    ISAC.nodes.forEach(n => { n.loss=0; n.acc=0; n.time=0; n.status='idle'; n.selected=false; });
    document.getElementById('sidebar-round').textContent = 'Round 1';
    this.render(currentPage);
  }
};

function navigate(page) { App.render(page); }
window.addEventListener('DOMContentLoaded', () => App.init());
