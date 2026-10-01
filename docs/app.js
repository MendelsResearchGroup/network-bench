/* A static explorer: all aggregation and filtering happens in the browser. */
const $ = id => document.getElementById(id);
const MODELS = ['gns', 'edge_mlp', 'mlp', 'tiny_mlp', 'linear_floor', 'frozen', 'edge_mlp_delta', 'edge_mlp_attention'];
const DATASETS = {node_optimized: 'Node optimized', stiff_optimized: 'Stiffness optimized', noisy_lj: 'Noisy LJ'};
const DESCRIPTIONS = {gns: 'Graph message passing', edge_mlp: 'Local bond encodings', mlp: 'Node features only', tiny_mlp: 'Single hidden layer', linear_floor: 'Linear velocity predictor', frozen: 'Frozen-position baseline'};
DESCRIPTIONS.edge_mlp_delta = 'Edge MLP with latest velocity and velocity differences; same parameter count';
DESCRIPTIONS.edge_mlp_attention = 'Velocity-difference edge MLP with a 32-channel attention-pooled global context';
let data, dataset = 'node_optimized', cohorts = {}, enabled = new Set(), horizon = 100;
let tableSort = {key: 'r2', descending: true};
const finite = x => typeof x === 'number' && Number.isFinite(x);
const number = (v, digits = 3) => finite(v) ? v.toFixed(digits) : '—';
const label = key => DATASETS[key] ?? key.replaceAll('_', ' ');
const css = key => getComputedStyle(document.documentElement).getPropertyValue(key).trim();
const color = model => css(`--s${Math.max(0, MODELS.indexOf(model)) + 1}`);
const settings = run => Object.entries(run.hyperparameters).map(([key, value]) => `${key}=${value}`).join(', ') || 'No hyperparameters';
const datasets = () => [...new Set(data.runs.map(run => run.dataset))].sort();
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function stats(values) {
  values = values.filter(finite);
  if (!values.length) return {mean: null, sd: null, n: 0};
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = values.length > 1 ? Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1)) : null;
  return {mean, sd, n: values.length};
}
function selectedRuns() {
  return data.runs.filter(run => run.dataset === dataset
    && run.comparison_id === cohorts[run.dataset] && enabled.has(run.model)
    && ($('plot-seed').value === 'all' || String(run.seed) === $('plot-seed').value));
}
function groups(runs) {
  const grouped = new Map();
  for (const run of runs) {
    const key = JSON.stringify([run.dataset, run.comparison_id, run.model, Object.entries(run.hyperparameters).sort()]);
    if (!grouped.has(key)) grouped.set(key, {...run, runs: []});
    grouped.get(key).runs.push(run);
  }
  return [...grouped.values()].map(group => {
    const steps = [...new Set(group.runs.flatMap(run => Object.keys(run.poisson_r2).map(Number)))].sort((a, b) => a - b);
    return {...group, points: Object.fromEntries(steps.map(step => [step, stats(group.runs.map(run => run.poisson_r2[step]))]))};
  });
}
function setDataset(value) {
  const changed = dataset !== value;
  dataset = value;
  for (const button of $('datasets').children) button.setAttribute('aria-pressed', String(button.dataset.value === dataset));
  const comparisons = [...new Set(data.runs.filter(run => run.dataset === dataset).map(run => run.comparison_id))];
  $('cohort-label').hidden = comparisons.length < 2;
  $('cohort').replaceChildren(...comparisons.map(id => new Option(id, id)));
  $('cohort').value = cohorts[dataset] ?? '';
  render(changed);
}
function render(animate = false) {
  const runs = selectedRuns(), grouped = groups(runs);
  const protocols = [...new Set(runs.map(run => {
    const p = run.protocol;
    const training = p.training_frames !== null && p.first_frame === 0 && p.frame_stride === 1
      ? `Train: first ${p.training_frames} frames per trajectory`
      : `Train: ${p.window_mode} windows · start frame ${p.first_frame} · stride ${p.frame_stride}`;
    return `${training} · ${run.rollout_steps}-step ${p.evaluation_split} rollout from ${p.history_frames} initial frames`;
  }))];
  $('protocol').textContent = protocols.join(' / ');
  $('protocol').title = 'Training, validation and test use separate trajectories. Checkpoints are selected on validation rollouts. Graph models use bond edges only, including on Noisy LJ.';
  const steps = [...new Set(runs.flatMap(run => Object.keys(run.poisson_r2).map(Number)))].sort((a, b) => a - b);
  const previous = horizon;
  if (steps.length) horizon = steps.reduce((best, step) => Math.abs(step - previous) < Math.abs(best - previous) ? step : best);
  $('horizon').max = Math.max(0, steps.length - 1);
  $('horizon').value = Math.max(0, steps.indexOf(horizon));
  $('horizon').disabled = !steps.length;
  $('horizon').dataset.steps = JSON.stringify(steps);
  $('horizon-value').textContent = `${horizon} steps`;
  $('first-step').textContent = steps.length ? `${steps[0]} steps` : '—';
  $('last-step').textContent = steps.length ? `${steps.at(-1)} steps` : '—';
  renderRanking(grouped);
  renderChart(animate === true);
  renderTable();
  for (const button of $('legend').children) {
    button.setAttribute('aria-pressed', String(enabled.has(button.dataset.model)));
    button.querySelector('.swatch').style.background = color(button.dataset.model);
  }
}
function renderRanking(grouped) {
  const ranking = grouped.map(group => ({...group, score: group.points[horizon] ?? stats([])}))
    .sort((a, b) => (b.score.mean ?? -Infinity) - (a.score.mean ?? -Infinity));
  $('ranking').replaceChildren();
  ranking.forEach((group, index) => {
    const item = node('li'), line = node('div', undefined, 'rank-line'), name = node('span', undefined, 'rank-name');
    const dot = node('i', undefined, 'swatch'); dot.style.background = color(group.model);
    name.append(node('span', String(index + 1).padStart(2, '0'), 'rank-number'), dot, node('span', group.model));
    const score = node('span', number(group.score.mean), 'rank-score');
    score.append(node('small', ` ± ${number(group.score.sd)}`));
    line.append(name, score);
    const meta = node('div', undefined, 'rank-meta');
    meta.append(node('span', `${group.parameters.toLocaleString()} parameters`), node('span', `${group.score.n}/${group.runs.length} finite seeds`));
    const bar = node('div', undefined, 'rank-bar'), fill = node('span');
    fill.style.width = `${Math.max(0, Math.min(1, group.score.mean ?? 0)) * 100}%`; fill.style.background = color(group.model); bar.append(fill);
    item.title = settings(group); item.append(line, meta, bar); $('ranking').append(item);
  });
  if (!ranking.length) $('ranking').append(node('li', 'No models selected.', 'empty'));
}
function rgba(hex, alpha) {
  return `rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${alpha})`;
}
function renderChart(animate) {
  if (!window.Plotly) { $('chart').textContent = 'The plot could not load. Results are available below.'; return; }
  const plot = $('plot'), traces = [];
  const individual = $('line-mode').value === 'individual';
  const all = groups(data.runs.filter(run => run.comparison_id === cohorts[run.dataset]
    && enabled.has(run.model) && ($('plot-seed').value === 'all' || String(run.seed) === $('plot-seed').value)));
  const identity = group => JSON.stringify([group.model, Object.entries(group.hyperparameters).sort()]);
  const slots = [...new Map(all.map(group => [identity(group), group])).entries()].sort((a,b) => a[0].localeCompare(b[0]));
  const x = [...new Set(data.runs.flatMap(run => Object.keys(run.poisson_r2).map(Number)))].sort((a,b) => a-b);
  const values = [];
  for (const [id, template] of slots) {
    const group = all.find(group => group.dataset === dataset && identity(group) === id);
    const c = color(template.model);
    const uid = 'model-' + [...id].map(char => char.codePointAt(0).toString(16)).join('-');
    const base = {x, ids:x.map(String), type:'scatter', connectgaps:false};
    if (individual) {
      const seeds = [...new Set(data.runs.filter(run => identity(run) === id).map(run => run.seed))].sort((a,b) => a-b);
      for (const seed of seeds) {
        const run = group?.runs.find(run => run.seed === seed);
        traces.push({...base, uid:`${uid}-${seed}`, y:x.map(step => run?.poisson_r2[step] ?? null),
          mode:'lines+markers', name:`${template.model} · seed ${seed}`,
          line:{color:c,width:2,dash:['solid','dash','dot'][seed % 3],simplify:false},marker:{size:4},
          hovertemplate:'%{y:.3f}<extra>%{fullData.name}</extra>'});
      }
    } else {
      const points = x.map(step => group?.points[step] ?? stats([]));
      const band = sign => points.map(point => $('bands').checked && point.n > 1 ? point.mean + sign * point.sd : null);
      traces.push({...base,uid:`${uid}-lower`,y:band(-1),mode:'lines',line:{width:0,simplify:false},hoverinfo:'skip',showlegend:false},
        {...base,uid:`${uid}-upper`,y:band(1),mode:'lines',line:{width:0,simplify:false},fill:'tonexty',fillcolor:rgba(c,.12),hoverinfo:'skip',showlegend:false},
        {...base,uid:`${uid}-mean`,y:points.map(point => point.mean),mode:'lines+markers',name:template.model,
          line:{color:c,width:2.6,simplify:false},marker:{size:4},customdata:points.map(point => [number(point.sd),point.n]),
          hovertemplate:'%{y:.3f} ± %{customdata[0]} (n=%{customdata[1]})<extra>%{fullData.name}</extra>'});
    }
  }
  // Shared axes and stable trace slots let each model morph into itself.
  for (const group of all) for (const point of Object.values(group.points)) {
    if (finite(point.mean)) values.push(point.mean + ($('bands').checked ? point.sd ?? 0 : 0));

  }
  if (individual) for (const group of all) for (const run of group.runs) values.push(...Object.values(run.poisson_r2).filter(finite));
  const mobile = window.innerWidth < 600;
  const layout = {
    height:mobile ? 380 : 520, margin:{l:46,r:16,t:20,b:48},paper_bgcolor:css('--surface'),plot_bgcolor:css('--surface'),
    font:{family:'-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',size:11,color:css('--muted')},
    showlegend:false,hovermode:'x unified',dragmode:'zoom',uirevision:'single-plot',
    hoverlabel:{bgcolor:css('--surface'),bordercolor:css('--line'),font:{size:11,color:css('--ink')}},
    xaxis:{title:{text:'ROLLOUT STEP',font:{size:9}},range:[x[0] ?? 0,x.at(-1) ?? 100],gridcolor:css('--grid'),zeroline:false,nticks:mobile?5:10},
    yaxis:{title:{text:'TEST R²',font:{size:9}},range:[0,Math.max(1,...values)+.05],gridcolor:css('--grid'),zerolinecolor:css('--muted'),tickformat:'.2f',nticks:5},
    shapes:[{type:'line',x0:horizon,x1:horizon,y0:0,y1:1,yref:'paper',line:{color:css('--muted'),width:1,dash:'dot'}}],
    annotations:traces.some(trace => trace.y.some(finite)) ? [] : [{text:'No results for these filters.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]
  };
  plot.setAttribute('aria-label', `${label(dataset)}: Poisson’s ratio R squared against rollout steps. Values are available in the runs table.`);
  const sameTraces = plot.data && plot.data.length === traces.length && traces.every((trace,i) => trace.uid === plot.data[i].uid);
  if (animate && sameTraces && plot.data.some(trace => trace.y.some(finite)) && traces.some(trace => trace.y.some(finite)) && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    Plotly.animate(plot,{data:traces,traces:traces.map((_,i) => i)},
      {mode:'immediate',transition:{duration:650,easing:'cubic-in-out'},frame:{duration:650,redraw:true}});
  } else {
    Plotly.react(plot,traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,displayModeBar:true,
      modeBarButtonsToRemove:['select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d'],
      toImageButtonOptions:{format:'svg',filename:'network-bench',width:1100,height:550}});
  }
  $('range-note').textContent = individual
    ? 'Individual seeds · Plot starts at 0; negative scores remain in the table.'
    : 'Mean ± sample SD across seeds · Plot starts at 0; negative scores remain in the table.';
}

function tableRuns() {
  return selectedRuns().filter(run => run.model.toLowerCase().includes($('search').value.trim().toLowerCase())
    && ($('seed').value === 'all' || String(run.seed) === $('seed').value));
}
const COLUMNS = [
  {key:'model',label:'Model / settings',value:run=>run.model},
  {key:'dataset',label:'Dataset',value:run=>label(run.dataset)},
  {key:'seed',label:'Seed',value:run=>run.seed},
  {key:'parameters',label:'Parameters',value:run=>run.parameters},
  {key:'epoch',label:'Epoch',value:run=>run.selected_epoch ?? run.epochs},
  {key:'val',label:'Validation score',value:run=>run.val_score},
  {key:'relative',label:'Relative MSE ↓',value:run=>run.relative_mse},
  {key:'r2',label:'Test R² ↑',value:run=>run.poisson_r2[horizon]},
  {key:'diverged',label:'Diverged',value:run=>run.diverged}
];
function renderTable() {
  const runs = tableRuns(), column = COLUMNS.find(column=>column.key===tableSort.key);
  runs.sort((a,b)=>{
    const av=column.value(a),bv=column.value(b);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const order=typeof av==='string'?av.localeCompare(bv):av-bv;
    return tableSort.descending ? -order:order;
  });
  $('table-count').textContent=`${runs.length} runs · test R² at step ${horizon}`;
  const head=node('tr');
  for (const column of COLUMNS) {
    const th=node('th'); th.scope='col';
    th.setAttribute('aria-sort',column.key===tableSort.key?(tableSort.descending?'descending':'ascending'):'none');
    const button=node('button',column.key==='r2'?`Test R² @${horizon} ↑`:column.label);
    button.onclick=()=>{tableSort={key:column.key,descending:column.key===tableSort.key?!tableSort.descending:true};renderTable();};
    th.append(button); head.append(th);
  }
  $('table-head').replaceChildren(head); $('table-body').replaceChildren();
  for (const run of runs) {
    const row=node('tr');
    for (const column of COLUMNS) {
      const cell=node('td'), value=column.value(run);
      if (column.key==='model') {
        const details=node('details'), summary=node('summary'), swatch=node('i',undefined,'swatch'); swatch.style.background=color(run.model);
        summary.append(swatch,node('span',run.model)); details.append(summary);
        const dl=node('dl');
        for (const [key,value] of Object.entries({...run.hyperparameters,selection:run.select_by ?? 'last epoch',run:run.name})) dl.append(node('dt',key),node('dd',String(value)));
        details.append(dl); cell.append(details);
      } else if (column.key==='seed') cell.append(node('span',String(value),'seed-pill'));
      else if (column.key==='parameters') cell.textContent=value.toLocaleString();
      else if (['val','relative','r2'].includes(column.key)) cell.textContent=number(value);
      else cell.textContent=value ?? '—';
      if (column.key==='r2') cell.className='score-cell';
      if (column.key==='val') cell.title=run.select_by ?? 'Last epoch; no selection metric';
      row.append(cell);
    }
    $('table-body').append(row);
  }
  if (!runs.length) {const row=node('tr'),cell=node('td','No runs match these filters.','empty');cell.colSpan=COLUMNS.length;row.append(cell);$('table-body').append(row);}
}
function downloadCSV() {
  const columns=['dataset','model','seed','hyperparameters','parameters','selected_epoch','select_by','val_score','relative_mse','position_mse','diverged',`poisson_r2_${horizon}`];
  const quote=value=>'"'+String(value ?? '').replaceAll('"','""')+'"';
  const rows=tableRuns().map(run=>columns.map(key=>quote(key==='hyperparameters'?JSON.stringify(run.hyperparameters):key.startsWith('poisson_r2_')?run.poisson_r2[horizon]:run[key])).join(','));
  const url=URL.createObjectURL(new Blob([[columns.join(','),...rows].join('\r\n')],{type:'text/csv;charset=utf-8'}));
  const link=node('a');link.href=url;link.download='network-bench-results.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function setTheme(dark) {
  document.documentElement.dataset.theme=dark?'dark':'light';
  $('theme').setAttribute('aria-label',`Switch to ${dark?'light':'dark'} theme`);
  if(data) render();
}
setTheme(false);
$('theme').onclick=()=>setTheme(document.documentElement.dataset.theme!=='dark');
fetch('results.json').then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();}).then(payload=>{
  data={...payload, runs:payload.runs.filter(run=>run.model!=='tiny_mlp')};
  $('generated').textContent=`Updated ${new Date(data.generated).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})} · ${data.runs.length} runs`;
  if(!data.runs.length){$('notice').textContent='No completed runs have been published yet.';return;}
  enabled=new Set(data.runs.map(run=>run.model));
  for(const key of datasets()){const button=node('button',label(key));button.dataset.value=key;button.onclick=()=>setDataset(key);$('datasets').append(button);}
  for(const key of datasets()){
    cohorts[key]=data.runs.find(run=>run.dataset===key).comparison_id;
  }
  for(const model of [...enabled].sort((a,b)=>MODELS.indexOf(a)-MODELS.indexOf(b))){
    const button=node('button'),dot=node('i',undefined,'swatch');dot.style.background=color(model);button.dataset.model=model;button.title=DESCRIPTIONS[model] ?? model;
    button.append(dot,node('span',model));button.onclick=()=>{enabled.has(model)?enabled.delete(model):enabled.add(model);render();};$('legend').append(button);
  }
  for(const seed of [...new Set(data.runs.map(run=>run.seed))].sort((a,b)=>a-b)) for(const id of ['plot-seed','seed']) $(id).append(new Option(`Seed ${seed}`,String(seed)));
  $('cohort').onchange=()=>{cohorts[dataset]=$('cohort').value;render();};
  for(const id of ['bands','plot-seed','line-mode'])$(id).onchange=render;
  $('horizon').oninput=()=>{horizon=JSON.parse($('horizon').dataset.steps)[$('horizon').value];render();};
  $('reset-models').onclick=()=>{enabled=new Set(data.runs.map(run=>run.model));render();};
  $('search').oninput=renderTable;$('seed').onchange=renderTable;$('download').onclick=downloadCSV;
  $('notice').hidden=true;$('dashboard').hidden=false;setDataset(datasets()[0]);
}).catch(error=>{$('notice').hidden=false;$('notice').textContent=`Results could not be loaded (${error.message}). Please reload, or download the JSON from the link above.`;});
