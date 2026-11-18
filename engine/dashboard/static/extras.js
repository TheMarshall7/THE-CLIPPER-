// Dashboard additions: Discover (§14 B), clip filters (C), clip editor (D), engine settings (E), Trends + Recipes (M15).
(() => {
  const addTab = (id, label, html, onShow) => {
    const btn = document.createElement('button'); btn.dataset.tab = id; btn.textContent = label;
    $('#nav').insertBefore(btn, document.querySelector('#nav button[data-tab=settings]'));
    const sec = document.createElement('section'); sec.id = id; sec.innerHTML = html; document.querySelector('main').appendChild(sec);
    btn.onclick = () => {
      document.querySelectorAll('#nav button').forEach(x => x.classList.toggle('active', x === btn));
      document.querySelectorAll('section').forEach(s => s.classList.toggle('active', s.id === id));
      tab = id; try { localStorage.setItem('tab', id); } catch (e) {}
      onShow && onShow();
    };
    try { if (localStorage.getItem('tab') === id) btn.click(); } catch (e) {}
    return sec;
  };
  const num = v => v === '' || v == null ? null : Number(v);
  const post = (url, body) => api(url, {body: body || {}});
  const pct = v => v == null ? 'n/a' : (v * 100).toFixed(0) + '%';

  // ================= B. Discover
  addTab('discover', 'Discover', `
    <div class="toolbar">
      <div><label>Search</label><input id="dq" value="GTA 6"></div>
      <div><label>Type</label><select id="dlive"><option value="true">Live now</option><option value="false">Recent videos / VODs</option></select></div>
      <div><label>Platforms</label><select id="dplat"><option value="twitch,youtube">Twitch + YouTube</option><option value="twitch">Twitch</option><option value="youtube">YouTube</option></select></div>
      <button class="btn" id="dgo">Search</button>
    </div>
    <div class="hint" id="dnotes" style="margin-bottom:10px"></div>
    <div class="card"><table><thead><tr><th>Score</th><th>Streamer</th><th>Platform</th><th>Title</th><th>Viewers</th><th>Length</th><th>Permission</th><th>Status</th><th></th></tr></thead>
    <tbody id="drows"><tr><td colspan="9" class="hint">Press Search. Discovery uses official Twitch / YouTube APIs and needs keys in config/secrets.env.</td></tr></tbody></table></div>`);
  $('#dgo').onclick = async () => {
    $('#dgo').disabled = true; $('#drows').innerHTML = '<tr><td colspan="9" class="hint">Searching…</td></tr>';
    try {
      const r = await api(`/api/discover?query=${encodeURIComponent($('#dq').value)}&live=${$('#dlive').value}&platforms=${$('#dplat').value}`);
      $('#dnotes').innerHTML = [...r.notes, r.kick].map(esc).join('<br>');
      $('#drows').innerHTML = r.results.length ? r.results.map((f, i) => `<tr>
        <td><b>${Math.round(f.score)}</b></td><td>${esc(f.streamer)}</td><td>${esc(f.platform)}</td><td>${esc(f.title)}${f.live ? ' <span class="badge run">live</span>' : ''}</td>
        <td>${f.viewers ?? 'n/a'}</td><td>${fmt(f.duration)}</td><td><span class="badge ${['explicitly_authorized','official_clipping_program'].includes(f.permission) ? 'good' : 'warn'}">${esc(f.permission.replace(/_/g, ' '))}</span></td>
        <td>${f.status ? esc(f.status) : '-'}</td>
        <td>${f.recording || !f.live ? `<button class="btn" data-i="${i}">Process</button>` : '<span class="hint">no recording yet</span>'}</td></tr>`).join('')
        : '<tr><td colspan="9" class="hint">No results.</td></tr>';
      document.querySelectorAll('#drows button[data-i]').forEach(b => b.onclick = async () => {
        const f = r.results[b.dataset.i]; b.disabled = true;
        try { const q = await post('/api/process', {source: f.url, streamer: f.streamer}); toast(`Queued: ${f.title}`); } catch (e) { toast('Could not queue: ' + e.message, 6000); b.disabled = false; }
      });
    } catch (e) { $('#drows').innerHTML = `<tr><td colspan="9" class="hint">${esc(e.message)}</td></tr>`; }
    $('#dgo').disabled = false;
  };

  // ================= C. clip library filters (client side, applied after each render)
  const tb = document.querySelector('#clips .toolbar');
  tb.insertAdjacentHTML('afterbegin', `
    <div><label>Streamer</label><select id="cf_streamer"><option value="">All</option></select></div>
    <div><label>Platform</label><select id="cf_plat"><option value="">All</option></select></div>
    <div><label>Length</label><select id="cf_dur"><option value="">Any</option><option value="0-20">under 20 s</option><option value="20-35">20-35 s</option><option value="35-999">over 35 s</option></select></div>
    <div><label>Date</label><select id="cf_date"><option value="">Any time</option><option value="1">Last 24 h</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option></select></div>
    <div><label>Sort</label><select id="cf_sort"><option value="score">Highest score</option><option value="new">Newest</option><option value="dur">Longest</option></select></div>`);
  window.applyClipFilters = () => {
    const cards = [...document.querySelectorAll('#cliplist .clip[data-id]')]; if (!cards.length) return;
    const fill = (sel, key) => { const el = $(sel), cur = el.value, vals = [...new Set(cards.map(c => c.dataset[key]).filter(Boolean))].sort();
      el.innerHTML = '<option value="">All</option>' + vals.map(v => `<option ${v === cur ? 'selected' : ''}>${esc(v)}</option>`).join(''); };
    if (!$('#cf_streamer')._f) { fill('#cf_streamer', 'streamer'); fill('#cf_plat', 'platform'); $('#cf_streamer')._f = 1; }
    const st = $('#cf_streamer').value, pl = $('#cf_plat').value, [d0, d1] = ($('#cf_dur').value || '0-9999').split('-').map(Number);
    const days = Number($('#cf_date').value) || 0, now = Date.now() / 1000;
    cards.forEach(c => {
      const dur = +c.dataset.dur, ts = +c.dataset.ts;
      c.style.display = (!st || c.dataset.streamer === st) && (!pl || c.dataset.platform === pl) && dur >= d0 && dur < d1 && (!days || now - ts < days * 86400) ? '' : 'none';
    });
    const key = $('#cf_sort').value, list = $('#cliplist');
    cards.sort((a, b) => key === 'new' ? b.dataset.ts - a.dataset.ts : key === 'dur' ? b.dataset.dur - a.dataset.dur : b.dataset.score - a.dataset.score).forEach(c => list.appendChild(c));
  };
  ['#cf_streamer', '#cf_plat', '#cf_dur', '#cf_date', '#cf_sort'].forEach(s => $(s).onchange = applyClipFilters);

  // ================= D. clip editor (modal) + recipe picker + performance entry, injected into each card
  document.body.insertAdjacentHTML('beforeend', `<div id="ed" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,.7); z-index:20; overflow:auto">
    <div class="card" style="max-width:760px; margin:30px auto"><h2 id="edtitle">Edit clip</h2><div id="edbody"></div>
    <div style="margin-top:16px; display:flex; gap:10px"><button class="btn" id="edsave">Save &amp; re-render</button><button class="btn ghost" id="edclose">Close</button></div></div></div>`);
  $('#edclose').onclick = () => $('#ed').style.display = 'none';
  let editing = null;
  window.openEditor = async id => {
    const d = await api(`/api/clips/${id}/detail`), rec = await api('/api/recipes').catch(() => []); editing = d;
    const cap = d.caption || {}, h = d.hooks || [];
    $('#edtitle').textContent = `Edit clip #${id}`;
    $('#edbody').innerHTML = `
      <div class="row"><div><label>Start (seconds in the source video)</label><input id="e_start" type="number" step="0.1" value="${d.start.toFixed(1)}"></div>
        <div><label>End</label><input id="e_end" type="number" step="0.1" value="${d.end.toFixed(1)}"></div></div>
      <div class="hint">Source video is ${fmt(d.stream_duration)} long. Changing the times re-cuts the clip and re-fits the music.</div>
      <div class="row"><div><label>Camera layout</label><select id="e_layout"><option value="">Automatic</option><option value="split">Face cam on top + gameplay</option><option value="fullcam">Face cam full screen</option><option value="gameplay_only">Gameplay only</option></select></div>
        <div><label>Gameplay framing</label><select id="e_gm"><option value="">Default</option><option value="fill">Fill the frame (crop)</option><option value="fit_blur">Whole frame (blurred sides)</option></select></div></div>
      <div class="row3"><div><label>Face cam crop: x</label><input id="e_cx" type="number" placeholder="${d.cam_box ? d.cam_box[0] : 'auto'}"></div>
        <div><label>y</label><input id="e_cy" type="number" placeholder="${d.cam_box ? d.cam_box[1] : 'auto'}"></div>
        <div><label>width × height</label><div style="display:flex; gap:6px"><input id="e_cw" type="number" placeholder="${d.cam_box ? d.cam_box[2] : 'w'}"><input id="e_ch" type="number" placeholder="${d.cam_box ? d.cam_box[3] : 'h'}"></div></div></div>
      <div class="hint">Source frame ${d.frame_size ? d.frame_size.join('×') : '?'} pixels. Leave the crop empty to keep the automatic face-cam detection.</div>
      <label>Gameplay focus (0 = left, 1 = right): <span id="e_fxv">${(+d.focus_x).toFixed(2)}</span></label><input id="e_fx" type="range" min="0" max="1" step="0.05" value="${d.focus_x}">
      <label>Hook text (shown for the first seconds)</label>
      <div class="row"><select id="e_hsel">${[0, 1, 2].map(i => `<option value="${i}" ${i === d.hook_choice ? 'selected' : ''}>Hook ${i + 1}${h[i] ? ': ' + esc(h[i].slice(0, 40)) : ''}</option>`).join('')}</select><input id="e_htext" value="${esc(h[d.hook_choice] || '')}"></div>
      <label>Captions</label>
      <div class="row3"><div><select id="e_cpos"><option value="">Position: keep</option><option value="lower">Lower</option><option value="seam">Between cam and game</option><option value="upper">Upper</option></select></div>
        <div><select id="e_canim"><option value="">Animation: keep</option><option value="pop">Pop</option><option value="highlight">Highlight</option><option value="none">None</option></select></div>
        <div><input id="e_cwords" type="number" min="1" max="6" placeholder="words per line (${cap.max_words_per_line || 3})"></div></div>
      <div class="row"><div><input id="e_cfont" placeholder="font (${esc(cap.font || 'default')})"></div><div><input id="e_ccol" placeholder="highlight colour (${esc(cap.highlight_color || '#FFD400')})"></div></div>
      <label>Title</label><input id="e_title" value="${esc(d.title || '')}">
      <label>Social media description</label><textarea id="e_desc" rows="4">${esc(d.description || '')}</textarea>
      <label>Style recipe</label><select id="e_rec"><option value="">Use the default recipe (if one is promoted)</option><option value="off" ${d.recipe_off ? 'selected' : ''}>No recipe</option>
        ${rec.filter(r => r.status !== 'retired').map(r => `<option value="${esc(r.recipe_id)}" ${d.recipe === r.recipe_id ? 'selected' : ''}>${esc(r.recipe_id)}${r.status === 'default' ? ' (default)' : ''}</option>`).join('')}</select>`;
    $('#e_layout').value = '';
    $('#e_gm').value = d.gameplay_mode || '';
    $('#e_fx').oninput = () => $('#e_fxv').textContent = (+$('#e_fx').value).toFixed(2);
    $('#e_hsel').onchange = () => $('#e_htext').value = h[$('#e_hsel').value] || '';
    $('#ed').style.display = 'block';
  };
  $('#edsave').onclick = async () => {
    const d = editing, b = {};
    const s = num($('#e_start').value), e = num($('#e_end').value);
    if (s !== d.start && s != null) b.start = s; if (e !== d.end && e != null) b.end = e;
    if ($('#e_layout').value) b.layout = $('#e_layout').value; if ($('#e_gm').value) b.gameplay_mode = $('#e_gm').value;
    const cb = ['#e_cx', '#e_cy', '#e_cw', '#e_ch'].map(x => num($(x).value));
    if (cb.every(v => v != null)) b.cam_box = cb; else if (cb.some(v => v != null)) return toast('Fill all four face-cam crop numbers, or none.');
    if (+$('#e_fx').value !== +d.focus_x) b.focus_x = +$('#e_fx').value;
    const hi = +$('#e_hsel').value; if (hi !== d.hook_choice) b.hook_choice = hi;
    if ($('#e_htext').value !== (d.hooks[hi] || '')) { b.hook_text = $('#e_htext').value; b.hook_choice = hi; }
    const cap = {position: $('#e_cpos').value || null, animation: $('#e_canim').value || null, max_words_per_line: num($('#e_cwords').value),
      font: $('#e_cfont').value || null, highlight_color: $('#e_ccol').value || null};
    if (Object.values(cap).some(v => v != null)) b.caption = cap;
    if ($('#e_title').value !== (d.title || '')) b.title = $('#e_title').value;
    if ($('#e_desc').value !== (d.description || '')) b.description = $('#e_desc').value;
    const rv = $('#e_rec').value; if (rv !== (d.recipe_off ? 'off' : (d.recipe || ''))) b.recipe = rv;
    if (!Object.keys(b).length) return toast('Nothing changed.');
    try { const r = await post(`/api/clips/${d.id}/edit`, b); toast(r.needs_render ? 'Saved. Re-rendering… the card updates when it finishes.' : 'Saved.', 6000); $('#ed').style.display = 'none'; }
    catch (err) { toast('Could not save: ' + err.message, 7000); }
  };
  new MutationObserver(() => {
    document.querySelectorAll('#cliplist .clip[data-id]:not([data-x])').forEach(c => {
      c.dataset.x = 1; const a = c.querySelector('.actions'); if (!a) return;
      a.insertAdjacentHTML('beforeend', `<button class="btn ghost" onclick="openEditor(${c.dataset.id})">Edit</button>
        <button class="btn ghost" onclick="logPerf(${c.dataset.id})" title="Enter how this clip performed after posting">Performance</button>`);
    });
    applyClipFilters();
  }).observe($('#cliplist'), {childList: true});
  window.logPerf = async id => {
    const v = prompt('Views so far (the Recipes tab compares recipes using these):'); if (v == null || v === '' || isNaN(+v)) return;
    const p = prompt('Platform (tiktok, youtube, instagram):', 'tiktok') || 'tiktok';
    await post(`/api/clips/${id}/performance`, {platform: p, views: +v}); toast('Recorded.');
  };

  // ================= E. engine settings (appended to the Settings tab)
  const sset = document.querySelector('#settings');
  sset.insertAdjacentHTML('beforeend', `<div class="card" style="max-width:760px; margin-top:16px" id="engset"><h2>Clips, video and processing</h2><div id="engbody" class="hint">loading…</div></div>`);
  const F = [
    ['Clips', [['clip_min', 'Shortest clip (s)', 'n'], ['clip_max', 'Longest clip (s)', 'n'], ['sensitivity', 'Highlight sensitivity (0 strict … 1 many clips)', 'n'], ['quality_threshold', 'Publish-ready quality bar (0-100)', 'n'], ['max_clips_per_stream', 'Clips per stream (minimum cap)', 'n']]],
    ['Transcription', [['whisper_model', 'Whisper model', 's'], ['whisper_device', 'Run on', 's']]],
    ['Video', [['width', 'Width', 's'], ['height', 'Height', 's'], ['encoder', 'Encoder', 's'], ['gameplay_mode', 'Gameplay framing', 's']]],
    ['Captions', [['cap_font', 'Font', 't'], ['cap_size', 'Size', 'n'], ['cap_highlight', 'Highlight colour', 't'], ['cap_words', 'Words per line', 'n'], ['cap_position', 'Position', 's'], ['cap_animation', 'Animation', 's']]],
    ['Processing', [['storage_dir', 'Storage location (folder)', 't'], ['max_concurrent_jobs', 'Streams at once', 'n'], ['priority', 'Processing priority', 's']]],
    ['Music extras', [['end_card', 'Add an end card with the song (YouTube Shorts)', 'b'], ['end_card_seconds', 'End card length (s)', 'n'], ['remove_source_music', 'Remove game radio / BGM so a song can be placed (needs Demucs)', 'b']]],
    ['Song under talking', [['ducking', 'When people talk on and off (smooth = steady level, tight = dips per phrase)', 's'], ['duck_hold_s', 'Pauses shorter than this still count as talking (s)', 'n'], ['beat_sync_cuts', 'Cuts land on the song's beat', 'b']]],
    ['Music promo (song first)', [['pre_drop_dropout', 'Silent beat before the chorus (auto = Music promo only)', 's'], ['pre_drop_seconds', 'Silence length (s, 0 = one beat)', 'n'], ['lyrics', 'Song lyrics on screen (auto = Music promo only)', 's'], ['promo_build_bars', 'Build-up before the chorus (bars)', 'n']]]];
  async function loadEng() {
    const r = await api('/api/settings/engine'); const v = r.values;
    $('#engbody').innerHTML = F.map(([g, fs]) => `<h3 style="margin-top:16px">${g}</h3><div class="row">${fs.map(([k, l, t]) =>
      t === 'b' ? `<label class="check" style="margin-top:14px"><input type="checkbox" data-k="${k}" ${v[k] ? 'checked' : ''}> ${esc(l)}</label>` :
      t === 's' ? `<div><label>${esc(l)}</label><select data-k="${k}">${(r.choices[k] || []).map(o => `<option ${String(o) === String(v[k]) ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select></div>` :
      `<div><label>${esc(l)}</label><input data-k="${k}" ${t === 'n' ? 'type="number" step="any"' : ''} value="${esc(v[k])}"></div>`).join('')}</div>`).join('')
      + `<div style="margin-top:16px; display:flex; gap:10px; align-items:center"><button class="btn" id="engsave">Save</button><span class="hint" id="engmsg">${r.demucs ? 'Demucs is installed.' : 'Demucs is not installed: source-music removal stays off until you run <code>.venv/Scripts/pip install demucs</code>.'}</span></div>`;
    $('#engsave').onclick = async () => {
      const b = {}; document.querySelectorAll('#engbody [data-k]').forEach(el => b[el.dataset.k] = el.type === 'checkbox' ? el.checked : el.value);
      try { await post('/api/settings/engine', b); toast('Saved. New jobs use these settings.'); } catch (e) { toast('Not saved: ' + e.message, 7000); }
    };
  }
  document.querySelector('#nav button[data-tab=settings]').addEventListener('click', loadEng); loadEng().catch(() => {});

  // ================= M15. Trends
  addTab('trends', 'Trends', `
    <div class="grid two">
      <div class="card"><h2>Reference clips</h2>
        <div class="hint">Paste links to clips you are allowed to analyze (or file paths). The engine measures layout, pacing, hook frame, captions, music and loudness. Nothing is copied.</div>
        <textarea id="tr_urls" rows="3" placeholder="https://www.youtube.com/shorts/…"></textarea>
        <div style="margin-top:10px; display:flex; gap:10px; align-items:center"><button class="btn" id="tr_go">Import &amp; analyze</button><span class="hint" id="tr_msg"></span></div></div>
      <div class="card"><h2>Trend report</h2><div class="toolbar" style="margin:0"><div><label>Window</label><select id="tr_win"><option>24h</option><option selected>7d</option><option>30d</option></select></div>
        <button class="btn ghost" id="tr_ref">Refresh</button></div><div id="tr_sum" class="hint"></div></div></div>
    <div id="tr_report" style="margin-top:16px"></div>
    <div class="card" style="margin-top:16px"><h3>Reference library</h3><table><thead><tr><th>#</th><th>Account</th><th>Length</th><th>Cuts/10 s</th><th>Layout</th><th>Music</th><th>LUFS</th><th>Views / day</th><th>Followers</th><th>× median</th><th></th></tr></thead><tbody id="tr_rows"></tbody></table></div>`, loadTrends);
  $('#tr_go').onclick = async () => {
    const urls = $('#tr_urls').value.split('\n').map(s => s.trim()).filter(Boolean); if (!urls.length) return;
    await post('/api/trends/import', {urls}); $('#tr_msg').textContent = 'Analyzing in the background…'; $('#tr_urls').value = ''; setTimeout(loadTrends, 8000);
  };
  $('#tr_ref').onclick = () => loadTrends(true); $('#tr_win').onchange = () => loadTrends();
  const patTable = (title, items, empty) => `<div class="card" style="margin-bottom:12px"><h3>${title}</h3>${items.length ? `<table><thead><tr><th>Pattern</th><th>Clips</th><th>× median</th><th>In top clips</th><th>Evidence</th></tr></thead><tbody>${items.map(i => `<tr>
    <td>${esc(i.pattern)}</td><td>${i.n} of ${i.of} ${i.early_signal ? '<span class="badge warn">early signal</span>' : ''}</td><td>${i.vs_baseline ?? 'n/a'}</td>
    <td>${pct(i.share_of_top_clips)}${i.prev_share_of_top != null ? ` <span class="hint">was ${pct(i.prev_share_of_top)}</span>` : ''}</td>
    <td>${i.examples.map(e => `<a href="${esc(e.url)}" target="_blank">#${e.id}</a>`).join(' ')}</td></tr>`).join('')}</tbody></table>` : `<div class="hint">${empty}</div>`}</div>`;
  async function loadTrends(refresh) {
    const [rep, refs] = await Promise.all([api(`/api/trends/report?window=${$('#tr_win').value}${refresh === true ? '&refresh=true' : ''}`), api('/api/trends/references')]);
    $('#tr_sum').innerHTML = `${rep.clips_in_window} clips in this window (${rep.clips_with_performance} with view data), ${rep.clips_in_previous_window} in the window before.<br>${esc(rep.note)}`;
    $('#tr_report').innerHTML = patTable("What's working now", rep.working_now, 'Nothing beats the baseline yet. Add reference clips with view counts.')
      + patTable('Rising', rep.rising, 'No pattern is gaining share among top clips yet (needs two windows of data).')
      + patTable('Saturated', rep.saturated, 'No pattern is both everywhere and flat yet.')
      + `<div class="card"><h3>Sounds in these clips</h3><div class="hint">${rep.sounds.clips_with_music} clips use added music · median ${rep.sounds.median_bpm ?? 'n/a'} BPM · entry styles: ${esc(JSON.stringify(rep.sounds.entry_styles))}. Trending songs are analysed for style only and never used in exports.</div></div>`;
    $('#tr_rows').innerHTML = refs.map(r => `<tr><td><a href="${esc(r.url)}" target="_blank">${r.id}</a></td><td>${esc(r.account || r.streamer || '-')}</td><td>${Math.round(r.duration || 0)} s</td>
      <td>${r.visual.cuts_per_10s ?? '-'}</td><td>${esc((r.visual.layout || {}).kind || '-')}</td><td>${esc((r.audio.music || {}).kind || '-')}</td><td>${r.audio.integrated_lufs ?? '-'}</td>
      <td>${r.metrics.views_per_day ?? 'n/a'}</td><td>${r.metrics.followers ?? 'n/a'}</td><td>${r.perf ?? 'n/a'}</td>
      <td><button class="btn ghost" style="padding:4px 8px" onclick="refMetrics(${r.id})">Metrics</button> <button class="btn ghost" style="padding:4px 8px" onclick="refDel(${r.id})">Remove</button></td></tr>`).join('')
      || '<tr><td colspan="11" class="hint">No reference clips yet.</td></tr>';
  }
  window.refMetrics = async id => {
    const v = prompt('Views:'); if (v == null || isNaN(+v)) return; const f = prompt('Account followers (blank if unknown):');
    await post(`/api/trends/references/${id}/metrics`, {views: +v, followers: f ? +f : null}); loadTrends();
  };
  window.refDel = async id => { if (confirm('Remove this reference clip?')) { await post(`/api/trends/references/${id}/delete`); loadTrends(); } };

  // ================= M15. Recipes
  addTab('recipes', 'Recipes', `
    <div class="toolbar"><div><label>Moment type</label><input id="rc_moment" value="any"></div><button class="btn" id="rc_make">Create recipe from the trend report</button></div>
    <div class="hint" style="margin-bottom:12px">A recipe is only style settings (layout, pacing, caption placement, music timing). Promote one to make it the default for new renders, or choose one per clip in the clip editor.</div>
    <div class="grid" id="rc_list"></div>`, loadRecipes);
  $('#rc_make').onclick = async () => { try { const r = await post('/api/recipes/create', {moment: $('#rc_moment').value || 'any'}); toast('Created ' + r.recipe_id); loadRecipes(); } catch (e) { toast(e.message, 6000); } };
  async function loadRecipes() {
    const rows = await api('/api/recipes');
    $('#rc_list').innerHTML = rows.map(r => { const s = r.score; return `<div class="card"><div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap"><h2 style="margin:0">${esc(r.recipe_id)}</h2>
      <span class="badge ${r.status === 'default' ? 'good' : r.status === 'retired' ? 'bad' : ''}">${esc(r.status)}</span>
      <span class="hint" style="margin:0 0 0 auto">${s.clips} clips measured · ${s.vs_baseline ? s.vs_baseline + '× baseline' : 'no data'} · ${esc(s.verdict)}</span></div>
      <div class="hint">${esc(r.based_on || '')}</div>
      <pre style="background:#0b0908; padding:10px; border-radius:8px; overflow:auto; font-size:12px">${esc(JSON.stringify({layout: r.layout, cam_ratio: r.cam_ratio, captions: r.captions, pacing: r.pacing, audio: r.audio, zoom: r.zoom}, null, 1))}</pre>
      <div><b>Our twist</b><ul>${(r.twists || []).map(t => `<li><span class="badge">${esc(t.kind.replace(/_/g, ' '))}</span> ${esc(t.text)}</li>`).join('')}</ul></div>
      <div class="actions">${r.status !== 'default' ? `<button class="btn good" onclick="rcAct('${esc(r.recipe_id)}','promote')">Make default</button>` : ''}
      ${r.status !== 'retired' ? `<button class="btn ghost" onclick="rcAct('${esc(r.recipe_id)}','retire')">Retire</button>` : `<button class="btn ghost" onclick="rcAct('${esc(r.recipe_id)}','restore')">Restore</button>`}</div></div>`; }).join('')
      || '<div class="empty">No recipes yet. Import reference clips on the Trends tab, then create one.</div>';
  }
  window.rcAct = async (id, a) => { await post(`/api/recipes/${encodeURIComponent(id)}/${a}`); loadRecipes(); };
})();
