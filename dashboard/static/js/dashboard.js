/* ═══════════════════════════════════════════════════════════════════════
   SMART HEALTHCARE MONITORING PLATFORM — DASHBOARD JAVASCRIPT v3
   Additions:
     - Zone polygon draw mode (click to place points, Save to submit)
     - Saline ROI drag-to-draw mode (drag rectangle over IV bag)
     - Clear Zone  /  Coma Mode toggle
     - Calibrate IV saline reference
     - Richer status updates (saline source, zone set, fell-from-bed)
   ═══════════════════════════════════════════════════════════════════════ */

// ── Socket.IO ──────────────────────────────────────────────────────────────
const socket = io();
socket.on('connect', () => showToast('Connected', 'Dashboard connected', 'low'));
socket.on('disconnect', () => showToast('Disconnected', 'Lost connection to server', 'critical'));

// ── State ──────────────────────────────────────────────────────────────────
let riskChartInstance = null;
let alertCount = 0;

// Draw-mode state machine
// mode: null | 'zone' | 'saline'
let drawMode    = null;
let zonePoints  = [];              // polygon points for zone draw
let salineRect  = null;            // { startX, startY, endX, endY }
let salineDragging = false;

// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock() {
    const el = document.getElementById('header-time');
    if (el) {
        const n = new Date();
        el.textContent =
            n.toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' })
            + '  ' + n.toLocaleTimeString('en-GB');
    }
}
setInterval(updateClock, 1000);
updateClock();

// ── API Polling ────────────────────────────────────────────────────────────
async function fetchStatus() {
    try { updateDashboard(await (await fetch('/api/status')).json()); }
    catch (e) { /* silent */ }
}
async function fetchAlerts() {
    try { updateAlertFeed((await (await fetch('/api/alerts')).json()).alerts || []); }
    catch (e) {}
}
async function fetchEvents() {
    try { updateEventTable((await (await fetch('/api/events?count=30')).json()).events || []); }
    catch (e) {}
}
async function fetchRiskHistory() {
    try {
        const d = await (await fetch('/api/risk')).json();
        if (d.history) updateRiskChart(d.history);
    } catch (e) {}
}
async function fetchZoneInfo() {
    try {
        const d = await (await fetch('/api/zone_info')).json();
        setText('zone-defined', d.zone_set ? `Yes (${(d.zone_pts||[]).length} pts)` : 'No — draw via Set Zone');
        setText('zone-fell', d.fell_from_bed ? '⚠ YES' : 'No');
    } catch (e) {}
}
async function fetchSalineInfo() {
    try {
        const d = await (await fetch('/api/saline_info')).json();
        const src = d.using_yolo ? 'YOLO (auto)' : d.roi ? 'Manual ROI set ✓' : 'Not set — click Set Saline';
        setText('saline-source', src);
    } catch (e) {}
}

setInterval(fetchStatus, 1000);
setInterval(fetchAlerts, 3000);
setInterval(fetchEvents, 5000);
setInterval(fetchRiskHistory, 5000);
setInterval(fetchZoneInfo, 2000);
setInterval(fetchSalineInfo, 2000);
setTimeout(() => { fetchStatus(); fetchAlerts(); fetchEvents(); fetchRiskHistory(); fetchZoneInfo(); fetchSalineInfo(); }, 500);

// ── Dashboard Update ───────────────────────────────────────────────────────
function updateDashboard(data) {
    // Stream
    const fps = data.stream?.fps || 0;
    setText('stream-fps', `${fps.toFixed(1)} FPS`);
    setPillState('pill-stream', fps > 5 ? 'ok' : fps > 0 ? 'warn' : 'error');

    // Vision
    const vision = data.hybrid?.mode || 'N/A';
    setText('vision-mode', vision.replace('_', ' '));
    setPillState('pill-vision', data.hybrid?.use_dl ? 'ok' : 'warn');

    // Persons
    const cnt = data.tracker?.active_count || 0;
    setText('person-count', cnt);
    setPillState('pill-persons', cnt > 0 ? 'ok' : 'warn');

    // Patient
    const lid = data.tracker?.locked_patient_id;
    setText('patient-id', lid != null ? `Patient #${lid}` : 'Not Locked');
    setText('patient-id-badge', lid != null ? `#${lid}` : '--');
    setText('patient-badge-text', lid != null ? `★ Patient #${lid} Locked` : 'No Patient Locked');
    setText('patient-persons', cnt);

    // Fall
    if (data.fall) {
        updateModuleCard('fall', data.fall.status, data.fall.confidence);
        setText('fall-angle', `${(data.fall.angle || 0).toFixed(0)}°`);
        setText('patient-posture', data.fall.posture || '--');
        setText('patient-angle', `${(data.fall.angle || 0).toFixed(0)}°`);
    }

    // Movement
    if (data.movement) {
        updateModuleCard('movement', data.movement.status, data.movement.confidence);
        const elapsed = data.movement.elapsed || 0;
        setText('movement-elapsed', `${elapsed.toFixed(0)}s`);
        setWidth('movement-progress', `${Math.min(elapsed / 30, 1) * 100}%`);
    }

    // Saline — basic update from polling (full update comes via WebSocket)
    if (data.saline) updateSalinePanel(data.saline);

    // Bed zone
    if (data.bed_zone) {
        updateModuleCard('zone', data.bed_zone.status, data.bed_zone.confidence);
        setText('zone-state', data.bed_zone.state || '--');
    }

    // Risk
    if (data.risk) {
        const score = data.risk.score || 0;
        updateRiskGauge(score, data.risk.level || 'SAFE');
        const c = data.risk.components || {};
        setWidth('risk-fall', `${c.fall || 0}%`);
        setWidth('risk-inactivity', `${c.inactivity || 0}%`);
        setWidth('risk-posture', `${c.posture || 0}%`);
        setWidth('risk-zone', `${c.zone || 0}%`);
        const trend = data.risk.trend || 'STABLE';
        const ta = document.getElementById('trend-arrow');
        const tt = document.getElementById('trend-text');
        if (ta && tt) {
            ta.className = 'trend-arrow ' + trend.toLowerCase();
            ta.textContent = trend === 'RISING' ? '↗' : trend === 'FALLING' ? '↘' : '→';
            tt.textContent = trend;
        }
    }
}

function updateModuleCard(module, status, confidence) {
    const sEl = document.getElementById(`${module}-status`);
    const cEl = document.getElementById(`${module}-confidence`);
    const vEl = document.getElementById(`${module}-conf-value`);
    if (sEl) { sEl.textContent = status; sEl.className = 'module-status ' + getStatusClass(status); }
    if (cEl) cEl.style.width = `${(confidence * 100).toFixed(0)}%`;
    if (vEl) vEl.textContent = `${(confidence * 100).toFixed(0)}%`;
}

function getStatusClass(status) {
    const s = (status || '').toUpperCase();
    if (['FALL DETECTED','NO MOVEMENT','EMPTY','OUT OF BED','FELL FROM BED','CRITICAL LOW'].some(w => s.includes(w))) return 'critical';
    if (['LOW','ON BED EDGE','STANDING NEAR','UNCERTAIN','WARNING','NO BOTTLE'].some(w => s.includes(w))) return 'warning';
    if (['NORMAL','ACTIVE','OK','IN BED','SITTING','LYING'].some(w => s.includes(w))) return 'ok';
    if (['NOT SET','NO ZONE','INITIALIZING','NOT DETECTED'].some(w => s.includes(w))) return 'normal';
    return 'normal';
}

function updateRiskGauge(score, level) {
    const fill = document.getElementById('gauge-fill');
    if (fill) fill.style.strokeDashoffset = 251 - (251 * score / 100);
    setText('risk-score-text', Math.round(score));
    const lt = document.getElementById('risk-level-text');
    if (lt) {
        lt.textContent = level;
        lt.style.fill = { SAFE:'#34d399', MODERATE:'#fbbf24', HIGH:'#fb923c', CRITICAL:'#f87171' }[level] || '#34d399';
    }
}

// ── Draw Mode Helpers ──────────────────────────────────────────────────────
function enterDrawMode(mode) {
    drawMode = mode;
    const banner = document.getElementById('draw-mode-banner');
    const text   = document.getElementById('draw-mode-text');
    const hint   = document.getElementById('draw-mode-hint');
    if (banner) banner.style.display = 'flex';

    const canvas = document.getElementById('overlay-canvas');
    const video  = document.getElementById('video-feed');
    const overlay = document.getElementById('video-overlay');
    canvas.width  = video.clientWidth;
    canvas.height = video.clientHeight;
    
    // Enable pointer events on the overlay so clicks go to the canvas
    if (overlay) overlay.style.pointerEvents = 'auto';
    canvas.style.cursor = 'crosshair';

    clearCanvas();

    if (mode === 'zone') {
        zonePoints = [];
        if (text) text.textContent = '🖊 Zone Draw Mode — Click to place corners';
        if (hint) hint.textContent = 'Click ≥3 points, then Save Zone';
        document.getElementById('btn-draw-zone').classList.add('btn-danger');
        document.getElementById('btn-draw-zone').innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
            Save Zone`;
    } else if (mode === 'saline') {
        salineRect = null; salineDragging = false;
        if (text) text.textContent = '📦 Saline Draw Mode — Drag to select IV bag';
        if (hint) hint.textContent = 'Click & drag a rectangle over the IV bag';
        document.getElementById('btn-set-saline').classList.add('btn-danger');
    }
}

function exitDrawMode(save = false) {
    const banner = document.getElementById('draw-mode-banner');
    if (banner) banner.style.display = 'none';
    
    const overlay = document.getElementById('video-overlay');
    const canvas = document.getElementById('overlay-canvas');
    if (overlay) overlay.style.pointerEvents = 'none';
    if (canvas) canvas.style.cursor = 'default';

    clearCanvas();

    if (drawMode === 'zone') {
        document.getElementById('btn-draw-zone').classList.remove('btn-danger');
        document.getElementById('btn-draw-zone').innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5"/></svg>
            Set Zone`;
        if (save) saveZone();
    } else if (drawMode === 'saline') {
        document.getElementById('btn-set-saline').classList.remove('btn-danger');
        if (save && salineRect) saveSalineROI();
    }
    drawMode = null;
}

function clearCanvas() {
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── Zone Drawing ───────────────────────────────────────────────────────────
function redrawZoneCanvas() {
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (zonePoints.length === 0) return;

    ctx.beginPath();
    ctx.moveTo(zonePoints[0].x, zonePoints[0].y);
    zonePoints.forEach((p, i) => { if (i > 0) ctx.lineTo(p.x, p.y); });

    if (zonePoints.length > 2) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(251,191,36,0.18)';
        ctx.fill();
    }
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Corner dots
    zonePoints.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(`P${i+1}`, p.x + 7, p.y - 5);
    });
}

async function saveZone() {
    if (zonePoints.length < 3) {
        showToast('Zone Error', 'Need at least 3 points to define bed zone', 'medium');
        return;
    }
    const video = document.getElementById('video-feed');
    const canvas = document.getElementById('overlay-canvas');
    const nw = video.naturalWidth  || 960;
    const nh = video.naturalHeight || 540;
    const scaled = zonePoints.map(p => [
        Math.round((p.x / canvas.width)  * nw),
        Math.round((p.y / canvas.height) * nh)
    ]);
    try {
        const res = await fetch('/api/set_zone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: scaled })
        });
        const data = await res.json();
        showToast('Bed Zone Saved', data.success ? `${zonePoints.length}-point zone saved ✓` : 'Failed', data.success ? 'ok' : 'critical');
    } catch(e) { showToast('Error', String(e), 'critical'); }
}

// ── Saline ROI Drawing ─────────────────────────────────────────────────────
function redrawSalineCanvas() {
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!salineRect) return;

    const { startX, startY, endX, endY } = salineRect;
    const x = Math.min(startX, endX), y = Math.min(startY, endY);
    const w = Math.abs(endX - startX), h = Math.abs(endY - startY);

    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(34,211,238,0.10)';
    ctx.fillRect(x, y, w, h);

    // Corner handles
    [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx,cy]) => {
        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, Math.PI*2);
        ctx.fillStyle = '#22d3ee';
        ctx.fill();
    });

    ctx.fillStyle = 'rgba(34,211,238,0.85)';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(`${Math.round(w)}×${Math.round(h)}px`, x + 4, y - 6 < 10 ? y + 14 : y - 6);
}

async function saveSalineROI() {
    if (!salineRect) return;
    const video  = document.getElementById('video-feed');
    const canvas = document.getElementById('overlay-canvas');
    const { startX, startY, endX, endY } = salineRect;

    try {
        const res = await fetch('/api/set_saline_roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x1: Math.min(startX, endX), y1: Math.min(startY, endY),
                x2: Math.max(startX, endX), y2: Math.max(startY, endY),
                natural_width:  video.naturalWidth  || 960,
                natural_height: video.naturalHeight || 540,
                display_width:  canvas.width,
                display_height: canvas.height,
            })
        });
        const data = await res.json();
        showToast('Saline ROI Set',
            data.success ? 'IV bag ROI locked. Tracking begins now.' : 'Failed to set ROI',
            data.success ? 'ok' : 'critical');
        if (data.success) fetchSalineInfo();
    } catch(e) { showToast('Error', String(e), 'critical'); }
}

// ── Canvas Mouse Events ────────────────────────────────────────────────────
const overlayCanvas = document.getElementById('overlay-canvas');

overlayCanvas?.addEventListener('click', (e) => {
    if (drawMode !== 'zone') return;
    const rect = overlayCanvas.getBoundingClientRect();
    zonePoints.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    redrawZoneCanvas();
});

overlayCanvas?.addEventListener('mousedown', (e) => {
    if (drawMode !== 'saline') return;
    const rect = overlayCanvas.getBoundingClientRect();
    salineRect = {
        startX: e.clientX - rect.left, startY: e.clientY - rect.top,
        endX:   e.clientX - rect.left, endY:   e.clientY - rect.top,
    };
    salineDragging = true;
});

overlayCanvas?.addEventListener('mousemove', (e) => {
    if (drawMode !== 'saline' || !salineDragging || !salineRect) return;
    const rect = overlayCanvas.getBoundingClientRect();
    salineRect.endX = e.clientX - rect.left;
    salineRect.endY = e.clientY - rect.top;
    redrawSalineCanvas();
});

overlayCanvas?.addEventListener('mouseup', (e) => {
    if (drawMode !== 'saline' || !salineDragging) return;
    salineDragging = false;
    // Auto-save if the rect is big enough
    if (salineRect) {
        const w = Math.abs(salineRect.endX - salineRect.startX);
        const h = Math.abs(salineRect.endY - salineRect.startY);
        if (w > 15 && h > 15) {
            exitDrawMode(true);
        } else {
            showToast('Too small', 'Drag a larger rectangle around the IV bag', 'medium');
        }
    }
});

// ── Button Handlers ────────────────────────────────────────────────────────
document.getElementById('btn-lock-patient')?.addEventListener('click', async () => {
    try {
        const data = await (await fetch('/api/status')).json();
        const ids = Object.keys(data.tracker?.persons || {});
        if (ids.length > 0) {
            await fetch('/api/lock_patient', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ track_id: parseInt(ids[0]) })
            });
            showToast('Patient Locked', `Locked onto patient #${ids[0]}`, 'ok');
        } else {
            showToast('No Person', 'No persons detected yet', 'medium');
        }
    } catch(e) { showToast('Error', String(e), 'critical'); }
});

document.getElementById('btn-unlock')?.addEventListener('click', async () => {
    await fetch('/api/unlock_patient', { method: 'POST' });
    showToast('Unlocked', 'Patient tracking unlocked', 'low');
});

// Zone draw mode toggle
document.getElementById('btn-draw-zone')?.addEventListener('click', function() {
    if (drawMode === 'zone') {
        exitDrawMode(true);   // Save
    } else {
        if (drawMode) exitDrawMode(false);
        enterDrawMode('zone');
    }
});

document.getElementById('btn-clear-zone')?.addEventListener('click', async () => {
    if (!confirm('Clear bed zone? Patient monitoring will show NO ZONE until a new zone is drawn.')) return;
    try {
        await fetch('/api/clear_zone', { method: 'POST' });
        showToast('Zone Cleared', 'Bed zone removed. Draw a new zone with Set Zone.', 'medium');
        fetchZoneInfo();
    } catch(e) { showToast('Error', String(e), 'critical'); }
});

// Saline draw mode
document.getElementById('btn-set-saline')?.addEventListener('click', function() {
    if (drawMode === 'saline') {
        exitDrawMode(false);  // Cancel
    } else {
        if (drawMode) exitDrawMode(false);
        enterDrawMode('saline');
    }
});

document.getElementById('btn-calibrate-saline')?.addEventListener('click', async () => {
    try {
        const res = await fetch('/api/calibrate_saline', { method: 'POST' });
        const d = await res.json();
        showToast('Calibration', d.success ? 'IV bag histogram locked as reference ✓' : (d.error || 'Failed'), d.success ? 'ok' : 'critical');
        if (d.success) fetchSalineInfo();
    } catch(e) { showToast('Error', String(e), 'critical'); }
});

document.getElementById('btn-clear-saline')?.addEventListener('click', async () => {
    if (!confirm('Clear Saline ROI? The IV tracking will stop until a new region is set.')) return;
    try {
        await fetch('/api/clear_saline', { method: 'POST' });
        showToast('Saline Cleared', 'IV ROI has been removed.', 'low');
        fetchSalineInfo();
    } catch(e) { showToast('Error', String(e), 'critical'); }
});

document.getElementById('btn-cancel-draw')?.addEventListener('click', () => exitDrawMode(false));

document.getElementById('btn-reset-fall')?.addEventListener('click', async () => {
    await fetch('/api/reset_fall', { method: 'POST' });
    showToast('Reset', 'Fall detector reset', 'low');
});

document.getElementById('btn-reset-movement')?.addEventListener('click', async () => {
    await fetch('/api/reset_movement', { method: 'POST' });
    showToast('Reset', 'Movement detector reset', 'low');
});

// ── Alert Feed ─────────────────────────────────────────────────────────────
function updateAlertFeed(alerts) {
    const list = document.getElementById('alert-list');
    if (!list) return;
    if (!alerts.length) { list.innerHTML = '<div class="alert-empty">No alerts yet</div>'; return; }

    const countEl = document.getElementById('alert-count');
    if (countEl) countEl.textContent = alerts.length;

    list.innerHTML = alerts.slice(0, 20).map(a => {
        const severity = (a.severity || 'low').toLowerCase();
        return `
        <div class="alert-item ${severity}">
            <span class="alert-time">${(a.timestamp || '').substring(11)}</span>
            <div class="alert-body">
                <div class="alert-type">${a.event_type || ''}</div>
                <div class="alert-detail">${(a.detail || '').substring(0, 80)}</div>
            </div>
            <span class="alert-conf">${((a.confidence||0)*100).toFixed(0)}%</span>
        </div>`;
    }).join('');
}

// ── Event Table ────────────────────────────────────────────────────────────
function updateEventTable(events) {
    const tbody = document.getElementById('event-tbody');
    if (!tbody) return;
    tbody.innerHTML = events.slice(0, 30).map(e => `
        <tr>
            <td>${(e.timestamp || '').substring(11)}</td>
            <td>${e.event_type || ''}</td>
            <td><span class="severity-badge ${(e.severity||'low').toLowerCase()}">${e.severity||''}</span></td>
            <td>${((e.confidence||0)*100).toFixed(0)}%</td>
            <td>${(e.detail || '').substring(0, 60)}</td>
        </tr>`).join('');
}

// ── Risk Chart ─────────────────────────────────────────────────────────────
function initRiskChart() {
    const ctx = document.getElementById('risk-chart');
    if (!ctx) return;
    riskChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [
            { label:'Risk Score', data:[], borderColor:'rgba(96,165,250,0.8)', backgroundColor:'rgba(96,165,250,0.1)', borderWidth:2, fill:true, tension:0.4, pointRadius:0, pointHitRadius:10 },
            { label:'Fall Risk',  data:[], borderColor:'rgba(248,113,113,0.6)', borderWidth:1, borderDash:[4,4], fill:false, tension:0.4, pointRadius:0 },
            { label:'Zone Risk',  data:[], borderColor:'rgba(251,191,36,0.6)',  borderWidth:1, borderDash:[4,4], fill:false, tension:0.4, pointRadius:0 },
        ]},
        options: {
            responsive:true, maintainAspectRatio:false, animation:{ duration:300 },
            scales:{
                x:{ grid:{ color:'rgba(255,255,255,0.03)' }, ticks:{ color:'#5a6478', maxTicksLimit:8, font:{size:10} } },
                y:{ min:0, max:100, grid:{ color:'rgba(255,255,255,0.03)' }, ticks:{ color:'#5a6478', font:{size:10} } }
            },
            plugins:{ legend:{ display:true, position:'top', labels:{ color:'#8994a8', boxWidth:12, font:{size:10} } } }
        }
    });
}

function updateRiskChart(history) {
    if (!riskChartInstance) initRiskChart();
    if (!riskChartInstance || !history.length) return;
    riskChartInstance.data.labels   = history.map(h => new Date(h.time*1000).toLocaleTimeString('en-GB'));
    riskChartInstance.data.datasets[0].data = history.map(h => h.score || 0);
    riskChartInstance.data.datasets[1].data = history.map(h => h.components?.fall || 0);
    riskChartInstance.data.datasets[2].data = history.map(h => h.components?.zone || 0);
    riskChartInstance.update('none');
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(title, body, severity = 'low') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${severity}`;
    toast.innerHTML = `<div class="toast-header"><span class="toast-title">${title}</span></div><div class="toast-body">${body}</div>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

// ── Socket.IO Events ───────────────────────────────────────────────────────
socket.on('alert',         d => { showToast(`⚠ ${d.event_type}`, d.detail || '', (d.severity||'low').toLowerCase()); fetchAlerts(); });
socket.on('status_update', d => updateDashboard(d));
socket.on('patient_locked',d => showToast('Patient Locked', `Monitoring patient #${d.locked_id}`, 'low'));
socket.on('zone_set',      d => showToast('Bed Zone Set', d.success ? 'Zone saved' : 'Failed', d.success ? 'ok' : 'critical'));

// ── Saline Real-time WebSocket Handler ──────────────────────────────────────
socket.on('saline_update', d => updateSalinePanel(d));
socket.on('saline_alert',  d => {
    showToast(`💉 ${d.status}`, `Level: ${(d.level||0).toFixed(0)}% — ${d.detail||''}`, 'critical');
    fetchAlerts();
});

// ── updateSalinePanel — full saline card update ────────────────────────────
function updateSalinePanel(d) {
    if (!d) return;
    const level   = Math.max(0, Math.min(100, d.level != null && d.level >= 0 ? d.level : 0));
    const status  = d.status || '--';
    const conf    = d.confidence || 0;
    const roiSet  = d.roi_set !== false && status.toUpperCase() !== 'NOT SET';

    // Module status badge
    updateModuleCard('saline', status, conf);

    // ── If ROI not set, show dormant state ──
    if (!roiSet) {
        const fluid = document.getElementById('saline-fluid-animated');
        if (fluid) { fluid.style.height = '0%'; fluid.className = 'saline-fluid-animated'; }
        const ll = document.getElementById('saline-level-line');
        if (ll) ll.style.bottom = '0%';
        setText('saline-pct-inner', '--');
        const lb = document.getElementById('saline-level-big');
        if (lb) { lb.textContent = '-- %'; lb.style.color = 'var(--text-muted)'; }
        setText('saline-track-id', '--');
        setText('saline-source', 'Not set — click Set Saline');
        setText('saline-cam-shift', '--');
        setText('saline-integrity', '--');
        setText('saline-n-methods', '0 / 4');
        ['canny','hough','gradient','gds'].forEach(n => {
            setWidth(`mbar-${n}`, '0%');
            setText(`mval-${n}`, '--');
        });
        setText('saline-spread', '--');
        setText('saline-integrity-inline', '--');
        const banner = document.getElementById('saline-alert-banner');
        if (banner) banner.style.display = 'none';
        return;
    }

    // Animated bottle
    const fluid = document.getElementById('saline-fluid-animated');
    const levelLine = document.getElementById('saline-level-line');
    const pctInner  = document.getElementById('saline-pct-inner');
    if (fluid) {
        fluid.style.height = `${level}%`;
        fluid.className = 'saline-fluid-animated';
        if (level <= 20) fluid.classList.add('critical');
        else if (level <= 30) fluid.classList.add('low');
    }
    if (levelLine) levelLine.style.bottom = `${level}%`;
    if (pctInner)  pctInner.textContent = `${level.toFixed(0)}%`;

    // Big level number
    const lvlBig = document.getElementById('saline-level-big');
    if (lvlBig) {
        lvlBig.textContent = `${level.toFixed(1)} %`;
        lvlBig.style.color = level <= 20 ? 'var(--accent-red)'
                           : level <= 30 ? 'var(--accent-yellow)'
                           : 'var(--accent-cyan)';
    }

    // Stats
    setText('saline-track-id',  d.tracking_id != null ? `#${d.tracking_id}` : '--');
    setText('saline-source',    d.using_yolo  ? 'YOLO (auto detect)' : 'Manual ROI ✓');
    setText('saline-cam-shift', d.camera_shifted ? '⚠ Detected' : '✓ Stable');
    setText('saline-integrity', d.integrity_ok !== false ? '✓ OK' : '⚠ Low');
    setText('saline-n-methods', `${d.n_methods || 0} / 4`);

    const camEl = document.getElementById('saline-cam-shift');
    if (camEl) camEl.style.color = d.camera_shifted ? 'var(--accent-yellow)' : 'var(--accent-green)';
    const intEl = document.getElementById('saline-integrity');
    if (intEl) intEl.style.color = d.integrity_ok !== false ? 'var(--accent-green)' : 'var(--accent-yellow)';

    // Method breakdown bars
    function updateMethodBar(name, val) {
        const bar = document.getElementById(`mbar-${name}`);
        const lbl = document.getElementById(`mval-${name}`);
        if (bar) bar.style.width = val != null ? `${Math.max(0, Math.min(100, val))}%` : '0%';
        if (lbl) lbl.textContent = val != null ? `${val.toFixed(0)}%` : '--';
    }
    updateMethodBar('canny',    d.canny_level);
    updateMethodBar('hough',    d.hough_level);
    updateMethodBar('gradient', d.gradient_level);
    updateMethodBar('gds',      d.gds_level);

    // Spread + integrity footer
    setText('saline-spread',           (d.spread || 0).toFixed(1));
    setText('saline-integrity-inline', d.integrity_ok !== false ? 'OK ✓' : 'Low ⚠');

    // Alert banner
    const banner = document.getElementById('saline-alert-banner');
    const alertText = document.getElementById('saline-alert-text');
    const isAlert = ['EMPTY','LOW','CRITICAL LOW'].includes(status.toUpperCase());
    if (banner) banner.style.display = isAlert ? 'block' : 'none';
    if (alertText && isAlert) {
        const msgs = {
            'EMPTY':        '🚨 IV EMPTY — Immediate attention needed!',
            'CRITICAL LOW': '⚠ CRITICAL — IV level below 20%',
            'LOW':          '⚠ LOW SALINE — Attend patient soon',
        };
        alertText.textContent = msgs[status.toUpperCase()] || `⚠ ${status}`;
    }
}

// ── Chart time window controls ─────────────────────────────────────────────
document.querySelectorAll('.chart-controls .btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.chart-controls .btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
    });
});

// ── Helpers ────────────────────────────────────────────────────────────────
function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
function setWidth(id, w)   { const el = document.getElementById(id); if (el) el.style.width  = w; }
function setHeight(id, h)  { const el = document.getElementById(id); if (el) el.style.height = h; }
function setPillState(id, state) {
    const pill = document.getElementById(id);
    if (!pill) return;
    const dot = pill.querySelector('.pill-dot');
    if (dot) dot.style.background = { ok:'#34d399', warn:'#fbbf24', error:'#f87171' }[state] || '#34d399';
}

// ── Resize canvas when video resizes ─────────────────────────────────────
new ResizeObserver(() => {
    if (drawMode) return;   // don't disrupt active draw
    const c = document.getElementById('overlay-canvas');
    const v = document.getElementById('video-feed');
    if (c && v) { c.width = v.clientWidth; c.height = v.clientHeight; }
}).observe(document.getElementById('video-container') || document.body);

// ── Init ──────────────────────────────────────────────────────────────────
initRiskChart();
console.log('Smart Healthcare Dashboard v3 loaded ✓');
