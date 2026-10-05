/* A static explorer: all aggregation and filtering happens in the browser. */
const $ = id => document.getElementById(id);
const MODELS = ['gns', 'edge_mlp', 'mlp', 'tiny_mlp', 'linear_floor', 'frozen', 'edge_mlp_delta', 'edge_mlp_attention'];
const MODEL_STYLES = [
  {symbol:'circle', glyph:'●', dash:'solid'},
  {symbol:'square', glyph:'■', dash:'dash'},
  {symbol:'diamond', glyph:'◆', dash:'dot'},
  {symbol:'triangle-up', glyph:'▲', dash:'dashdot'},
  {symbol:'triangle-down', glyph:'▼', dash:'longdash'},
  {symbol:'cross', glyph:'✚', dash:'longdashdot'},
  {symbol:'star', glyph:'★', dash:'solid'},
  {symbol:'hexagon', glyph:'⬡', dash:'dash'}
];
const DATASETS = {node_optimized: 'Node optimized', stiff_optimized: 'Stiffness optimized', noisy_lj: 'Noisy LJ'};
const DESCRIPTIONS = {gns: 'Graph message passing', edge_mlp: 'Local bond encodings', mlp: 'Node features only', tiny_mlp: 'Single hidden layer', linear_floor: 'Linear velocity predictor', frozen: 'Frozen-position baseline'};
DESCRIPTIONS.edge_mlp_delta = 'Edge MLP with latest velocity and velocity differences; same parameter count';
DESCRIPTIONS.edge_mlp_attention = 'Velocity-difference edge MLP with a 32-channel attention-pooled global context';
let data, studies, study = 'rollouts', trainSize = 100, dataset = 'node_optimized', mode = 'normal', trainingMode = 'one_step', cohorts = {}, enabled = new Set(), horizon = 100, topModels = true;
let tableSort = {key: 'r2', descending: true};
const finite = x => typeof x === 'number' && Number.isFinite(x);
const number = (v, digits = 3) => finite(v) ? v.toFixed(digits) : '—';
const scientific = v => finite(v) ? v.toExponential(3) : '—';
const metricValues = (run, metric) => metric === 'poisson_r2' ? run.poisson_r2
  : run[metric] && Object.keys(run[metric]).length ? run[metric]
  : {[run.rollout_steps]: run[metric.replace('_by_step', '')]};
const label = key => DATASETS[key] ?? key.replaceAll('_', ' ');
const css = key => getComputedStyle(document.documentElement).getPropertyValue(key).trim();
const color = model => css(`--s${Math.max(0, MODELS.indexOf(model)) + 1}`);
const modelStyle = model => MODEL_STYLES[Math.max(0, MODELS.indexOf(model))];
const settings = run => Object.entries(run.hyperparameters).map(([key, value]) => `${key}=${value}`).join(', ') || 'No hyperparameters';
const datasets = () => [...new Set(data.runs.map(run => run.dataset))].sort();
const cohortKey = (key = dataset) => `${study}:${key}:${mode}:${trainingMode}`;
const learning = () => study === 'learning';
const matchesMode = run => (run.mode ?? 'normal') === mode && (run.training_mode ?? 'one_step') === trainingMode;
const modeRuns = (key = dataset) => data.runs.filter(run => run.dataset === key && matchesMode(run));
const missingLearning = () => learning() && !modeRuns().length;
const emptyResults = () => missingLearning()
  ? `${mode === 'ood' ? 'OOD' : 'Normal'} / ${trainingMode === 'multi_step' ? 'MST' : 'One-step'} learning curves have not been run yet. Available: Normal / One-step.`
  : 'No results for these filters.';
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
  return modeRuns().filter(run => run.comparison_id === cohorts[cohortKey()]
    && (!learning() || run.train_networks === trainSize)
    && ($('plot-seed').value === 'all' || String(run.seed) === $('plot-seed').value));
}
function groups(runs, metric = 'poisson_r2', trainingAxis = false) {
  const grouped = new Map();
  for (const run of runs) {
    const key = JSON.stringify([run.dataset, run.comparison_id, run.model, Object.entries(run.hyperparameters).sort()]);
    if (!grouped.has(key)) grouped.set(key, {...run, runs: []});
    grouped.get(key).runs.push(run);
  }
  return [...grouped.values()].map(group => {
    const steps = [...new Set(group.runs.flatMap(run => trainingAxis ? [run.train_networks] : Object.keys(metricValues(run, metric)).map(Number)))].sort((a, b) => a - b);
    return {...group, points: Object.fromEntries(steps.map(step => [step, stats(group.runs
      .filter(run => !trainingAxis || run.train_networks === step)
      .map(run => metricValues(run, metric)[trainingAxis ? horizon : step]))]))};
  });
}
function setStudy(value, resetModels = true) {
  study = value;
  data = studies[study];
  $('train-size-label').hidden = !learning();
  $('split-seed-label').hidden = !learning();
  $('plot-seed-label').textContent = learning() ? 'Split & training seed' : 'Training seed';
  for (const button of $('studies').children) button.setAttribute('aria-pressed', String(button.dataset.study === study));
  const sizes = [...new Set(data.runs.map(run => run.train_networks).filter(finite))].sort((a,b) => a-b);
  $('train-size').replaceChildren(...sizes.map(size => new Option(`${size} networks`, String(size))));
  if (sizes.length && !sizes.includes(trainSize)) trainSize = sizes.at(-1);
  $('train-size').value = String(trainSize);
  $('study-note').textContent = learning()
    ? 'Learning curves: how accuracy changes with 10–100 training networks. Both plots score the selected rollout horizon. The dotted line marks the training size inspected in the ranking, table and split details.'
    : 'Rollouts: how accuracy changes over prediction steps for a fixed training set.';
  $('generated').textContent = `Updated ${new Date(data.generated).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})} · ${data.runs.length} runs in this view`;
  $('json-download').href = learning() ? 'learning-curves.json' : 'results.json';
  setDataset(datasets().includes(dataset) ? dataset : datasets()[0], resetModels);
}
function setDataset(value, resetModels = true) {
  const changed = dataset !== value;
  dataset = value;
  if (resetModels) topModels = true;
  for (const button of $('datasets').children) button.setAttribute('aria-pressed', String(button.dataset.value === dataset));
  const comparisons = [...new Set(modeRuns().map(run => run.comparison_id))];
  cohorts[cohortKey()] ??= comparisons[0];
  $('cohort-label').hidden = comparisons.length < 2;
  $('cohort').replaceChildren(...comparisons.map(id => new Option(id, id)));
  $('cohort').value = cohorts[cohortKey()] ?? '';
  for (const button of $('modes').children) button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  for (const button of $('training-modes').children) button.setAttribute('aria-pressed', String(button.dataset.training === trainingMode));
  render(changed);
}
function restoreURL() {
  const params = new URL(location.href).searchParams;
  study = params.get('view') === 'learning' ? 'learning' : 'rollouts';
  data = studies[study];
  trainSize = Number(params.get('train') ?? 100);
  $('split-seed').value = ['0','1','2'].includes(params.get('split_seed')) ? params.get('split_seed') : '0';
  dataset = datasets().includes(params.get('dataset')) ? params.get('dataset') : datasets()[0];
  mode = params.get('mode') === 'ood' ? 'ood' : 'normal';
  trainingMode = params.get('training') === 'mst' ? 'multi_step' : 'one_step';
  const step = Number(params.get('step') ?? 100);
  horizon = finite(step) ? step : 100;
  const comparisons = modeRuns().map(run => run.comparison_id);
  cohorts[cohortKey()] = comparisons.includes(params.get('cohort')) ? params.get('cohort') : comparisons[0];
  for (const [id, key, fallback] of [['plot-seed','seed','all'], ['line-mode','lines','mean'], ['seed','table_seed','all']]) {
    const value = params.get(key) ?? fallback;
    $(id).value = [...$(id).options].some(option => option.value === value) ? value : fallback;
  }
  $('bands').checked = params.get('bands') !== 'false';
  $('search').value = params.get('search') ?? '';
  topModels = !params.has('models');
  const available = new Set(data.runs.map(run => run.model));
  enabled = topModels ? available : new Set(params.get('models').split(',').filter(model => available.has(model)));
  setStudy(study, false);
}
function updateURL() {
  const url = new URL(location.href), params = url.searchParams;
  params.set('dataset', dataset);
  params.set('mode', mode);
  params.set('training', trainingMode === 'multi_step' ? 'mst' : 'one_step');
  const optional = {
    view:learning() ? 'learning' : null,
    train:learning() ? String(trainSize) : null,
    split_seed:learning() ? $('split-seed').value : null,
    step:horizon === 100 ? null : String(horizon),
    seed:$('plot-seed').value === 'all' ? null : $('plot-seed').value,
    lines:$('line-mode').value === 'mean' ? null : $('line-mode').value,
    bands:$('bands').checked ? null : 'false',
    models:topModels ? null : [...enabled].join(','),
    cohort:$('cohort-label').hidden ? null : cohorts[cohortKey()],
    table_seed:$('seed').value === 'all' ? null : $('seed').value,
    search:$('search').value || null
  };
  for (const [key, value] of Object.entries(optional)) value == null ? params.delete(key) : params.set(key, value);
  history.replaceState(null, '', url);
}
window.addEventListener('popstate', () => { if (data) restoreURL(); });
window.addEventListener('resize', () => {
  if (data) { renderChart(false); renderChart(false, 'position_mse_by_step'); }
});
function renderSplit() {
  if (learning()) {
    if ($('plot-seed').value !== 'all') $('split-seed').value = $('plot-seed').value;
    $('split-seed').disabled = $('plot-seed').value !== 'all';
  }
  const run = learning() ? modeRuns().find(run => run.train_networks === trainSize && String(run.split_seed) === $('split-seed').value) : null;
  const split = data.splits[learning() ? run?.split_id : cohorts[cohortKey()]];
  $('ranking-audit').hidden = learning() || mode !== 'ood';
  $('ranking-audit').href = `https://github.com/MendelsResearchGroup/network-bench/blob/main/configs/ood/${dataset}.ranking.json`;
  $('split-rule').textContent = missingLearning() ? emptyResults() : learning()
    ? 'Normal · one-step training. Each seed shuffles the full dataset roster: 100 networks in a training pool, 70 for validation and 70 for test. Training sizes use nested prefixes of the pool; validation and test stay fixed within a seed. Seeds change both membership and training randomness.'
    : mode === 'ood'
    ? 'OOD: rank networks by ground-truth Poisson’s ratio at 100 steps. Train and validate within the highest 30%; test on up to 100 networks from the lower 70%.'
    : 'Normal: shuffle networks with a fixed data-split seed, then assign separate training, validation and test networks.';
  $('split-counts').textContent = split
    ? `· ${learning() ? `seed ${split.seed} · ` : ''}${split.systems.train.length} train / ${split.systems.val.length} validation / ${split.systems.test.length} test / ${split.unused.length} unused`
    : missingLearning() ? '· not run yet' : '· awaiting completed runs';
  $('split-note').textContent = split
    ? learning()
      ? `Showing split and training seed ${split.seed} at ${trainSize} training networks. Training uses the first ${trainSize} of this seed's 100-network pool. The same 70 validation and 70 test networks are used at every training size and for every model within this seed. Other seeds have different memberships, which can overlap across seeds. Unused IDs include the unselected part of the training pool. Bands show sample SD of per-seed scores, including split and training variation; they are not confidence intervals. Checkpoints are selected using validation R² at 100 steps. IDs below match the saved runs exactly.`
      : mode === 'ood'
      ? `The upper 30% is rounded up, then shuffled with seed ${split.seed}. Its first ${split.systems.train.length} networks are training; the remaining ${split.systems.val.length} are validation. The lower 70% is shuffled using the same generator; its first ${split.systems.test.length} networks are test, and the rest are unused. Ranking uses the benchmark’s position-based ratio from initial frame 3 to frame 103; equal ratios are ordered by network ID. Checkpoints are selected only on high-ratio validation networks. Sample counts differ from Normal; R² uses this mode’s own test population. Network IDs below match the saved run exactly.`
      : `Data-split seed ${split.seed}: ${split.systems.train.length} training, ${split.systems.val.length} validation and ${split.systems.test.length} test networks; remaining networks are unused. Training seeds change model initialization, while these network assignments stay fixed. Network IDs below match the saved run exactly.`
    : missingLearning() ? 'There are no saved splits or runs for this learning-curve selection.' : 'No completed results have been published for this split yet.';
  $('split-membership').replaceChildren();
  if (split) for (const [part, title] of [['train', 'Training'], ['val', 'Validation'], ['test', 'Test'], ['unused', 'Unused']]) {
    const ids = part === 'unused' ? split.unused : split.systems[part];
    if (!ids.length) continue;
    const section = node('section'), list = node('ol');
    section.append(node('h3', `${title} · ${ids.length} networks`));
    for (const id of [...ids].sort()) list.append(node('li', id));
    section.append(list); $('split-membership').append(section);
  }
}
function render(animate = false) {
  const runs = selectedRuns(), grouped = groups(runs);
  renderSplit();
  $('ranking-note').textContent = `Ranked by mean R² at step ${horizon}${learning() ? ` · ${trainSize} training networks` : ''}.`;
  $('legend-note').textContent = `The same models appear in both plots. Top 3 by mean R² at step ${horizon}${learning() ? ` and ${trainSize} training networks` : ''} shown by default. Click a model to toggle both lines; numbers show parameter counts.`;
  const protocols = [...new Set(runs.map(run => {
    const p = run.protocol;
    const training = p.training_frames !== null && p.first_frame === 0 && p.frame_stride === 1
      ? `Train: first ${p.training_frames} frames per trajectory`
      : `Train: ${p.window_mode} windows · start frame ${p.first_frame} · stride ${p.frame_stride}`;
    const objective = (run.training_mode ?? 'one_step') === 'multi_step'
      ? `MST: ${run.rollout_schedule.map(([epoch, steps]) => `${steps} steps from epoch ${epoch + 1}`).join(', ')}; ${run.detach_rollout ? 'detached predictions' : 'gradients through rollout'}`
      : 'One-step training';
    return `${training} · ${objective} · ${run.rollout_steps}-step ${p.evaluation_split} rollout from ${p.history_frames} initial frames (${run.rollout_steps + p.history_frames} frames total)`;
  }))];
  $('protocol').textContent = protocols.join(' / ') || (missingLearning() ? 'Select Normal and One-step to see the completed training-size sweep.' : `Awaiting completed ${trainingMode === 'multi_step' ? 'MST' : 'one-step'} results for this dataset and split.`);
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
  if (topModels) enabled = new Set(rankedGroups(grouped).filter(group => finite(group.score.mean)).slice(0, 3).map(group => group.model));
  renderRanking(grouped);
  renderChart(animate === true);
  renderChart(animate === true, 'position_mse_by_step');
  renderTable();
  for (const button of $('legend').children) {
    const model = button.dataset.model;
    const counts = [...new Set(data.runs.filter(run => run.dataset === dataset
      && matchesMode(run) && run.comparison_id === cohorts[cohortKey()] && run.model === model).map(run => run.parameters))];
    button.hidden = !counts.length;
    button.querySelector('.model-parameters').textContent = counts.map(count => count.toLocaleString()).join(' / ');
    button.title = `${DESCRIPTIONS[model] ?? model} · ${counts.map(count => count.toLocaleString()).join(' / ')} parameters`;
    button.setAttribute('aria-pressed', String(enabled.has(model)));
    button.querySelector('.swatch').style.color = color(model);
  }
}
function rankedGroups(grouped) {
  return grouped.map(group => ({...group, score: group.points[horizon] ?? stats([])}))
    .sort((a, b) => (b.score.mean ?? -Infinity) - (a.score.mean ?? -Infinity));
}
function renderRanking(grouped) {
  const ranking = rankedGroups(grouped);
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
  if (!ranking.length) $('ranking').append(node('li', missingLearning() ? emptyResults() : 'No models selected.', 'empty'));
}
function rgba(hex, alpha) {
  return `rgba(${parseInt(hex.slice(1,3),16)},${parseInt(hex.slice(3,5),16)},${parseInt(hex.slice(5,7),16)},${alpha})`;
}
function renderChart(animate, metric = 'poisson_r2') {
  const mse = metric === 'position_mse_by_step';
  if (!window.Plotly) { $(mse ? 'mse-chart' : 'chart').textContent = 'The plot could not load. Results are available below.'; return; }
  const plot = $(mse ? 'mse-plot' : 'plot'), traces = [];
  const individual = $('line-mode').value === 'individual';
  const plottedScore = value => finite(value) ? Math.max(0, value) : null;
  const all = groups(data.runs.filter(run => matchesMode(run) && run.comparison_id === cohorts[cohortKey(run.dataset)]
    && enabled.has(run.model) && ($('plot-seed').value === 'all' || String(run.seed) === $('plot-seed').value)), metric, learning());
  const identity = group => JSON.stringify([group.model, Object.entries(group.hyperparameters).sort()]);
  const slots = [...new Map(all.map(group => [identity(group), group])).entries()].sort((a,b) => a[0].localeCompare(b[0]));
  const x = [...new Set(data.runs.flatMap(run => learning() ? [run.train_networks] : Object.keys(run.poisson_r2).map(Number)))].sort((a,b) => a-b);
  const values = [];
  for (const [id, template] of slots) {
    const group = all.find(group => group.dataset === dataset && identity(group) === id);
    const c = color(template.model);
    const appearance = modelStyle(template.model);
    const uid = 'model-' + [...id].map(char => char.codePointAt(0).toString(16)).join('-');
    const base = {x, ids:x.map(String), type:'scatter', connectgaps:false};
    if (individual) {
      const seeds = [...new Set(data.runs.filter(run => identity(run) === id).map(run => run.seed))].sort((a,b) => a-b);
      for (const seed of seeds) {
        const scores = x.map(step => {
          const run = group?.runs.find(run => run.seed === seed && (!learning() || run.train_networks === step));
          return run ? metricValues(run, metric)[learning() ? horizon : step] ?? null : null;
        });
        traces.push({...base, uid:`${uid}-${seed}`, y:scores.map(plottedScore), customdata:scores,
          mode:'lines+markers', name:`${template.model} · seed ${seed}`,
          line:{color:c,width:2.4,dash:['solid','dash','dot'][seed % 3],simplify:false},marker:{size:6,symbol:appearance.symbol},
          hovertemplate:`${learning() ? '%{x} training networks · ' : ''}%{customdata:${mse ? '.3e' : '.3f'}}<extra>%{fullData.name}</extra>`});
      }
    } else {
      const points = x.map(step => group?.points[step] ?? stats([]));
      const band = sign => points.map(point => $('bands').checked && point.n > 1 ? plottedScore(point.mean + sign * point.sd) : null);
      traces.push({...base,uid:`${uid}-lower`,y:band(-1),mode:'lines',line:{width:0,simplify:false},hoverinfo:'skip',showlegend:false},
        {...base,uid:`${uid}-upper`,y:band(1),mode:'lines',line:{width:0,simplify:false},fill:'tonexty',fillcolor:rgba(c,.12),hoverinfo:'skip',showlegend:false},
        {...base,uid:`${uid}-mean`,y:points.map(point => plottedScore(point.mean)),mode:'lines+markers',name:template.model,
          line:{color:c,width:3,dash:appearance.dash,simplify:false},marker:{size:7,symbol:appearance.symbol},customdata:points.map(point => [(mse ? scientific : number)(point.sd),point.n,point.mean]),
          hovertemplate:`${learning() ? '%{x} training networks · ' : ''}%{customdata[2]:${mse ? '.3e' : '.3f'}} ± %{customdata[0]} (n=%{customdata[1]})<extra>%{fullData.name}</extra>`});
    }
  }
  // Shared axes and stable trace slots let each model morph into itself.
  const visible = all.filter(group => group.dataset === dataset);
  for (const group of visible) for (const point of Object.values(group.points)) {
    if (finite(point.mean)) values.push(point.mean + ($('bands').checked ? point.sd ?? 0 : 0));

  }
  if (individual) for (const group of visible) for (const run of group.runs) values.push(...(learning() ? [metricValues(run, metric)[horizon]] : Object.values(metricValues(run, metric))).filter(finite));
  const mobile = window.innerWidth < 600;
  const maximum = Math.max(0, ...values);
  const hasResults = traces.some(trace => trace.y.some(finite));
  const layout = {
    height:mobile ? 440 : 560, margin:{l:90,r:20,t:26,b:68},paper_bgcolor:css('--surface'),plot_bgcolor:css('--surface'),
    font:{family:'-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',size:16,color:css('--ink')},
    showlegend:false,hovermode:'x unified',dragmode:'zoom',uirevision:`${study}:${dataset}:${metric}:${learning() ? horizon : ''}`,
    hoverlabel:{bgcolor:css('--surface'),bordercolor:css('--line'),font:{size:20,color:css('--ink')}},
    xaxis:{visible:hasResults,title:{text:learning() ? 'TRAINING NETWORKS' : 'ROLLOUT STEP',font:{size:16}},range:[x[0] ?? 0,x.at(-1) ?? 100],gridcolor:css('--grid'),zeroline:false,nticks:mobile?4:10},
    yaxis:{visible:hasResults,title:{text:mse ? 'POSITION MSE' : 'TEST R²',font:{size:16}},range:[0,mse ? (maximum > 0 ? maximum * 1.05 : 1) : Math.max(1,maximum)+.05],gridcolor:css('--grid'),zerolinecolor:css('--muted'),tickformat:mse ? '.1e' : '.2f',nticks:5},
    shapes:hasResults ? [{type:'line',x0:learning() ? trainSize : horizon,x1:learning() ? trainSize : horizon,y0:0,y1:1,yref:'paper',line:{color:css('--muted'),width:1,dash:'dot'}}] : [],
    annotations:hasResults ? [] : [{text:emptyResults().replace(' learning curves have not been run yet. Available:', '<br>Learning curves not run yet.<br>Available:'),xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}]
  };
  plot.setAttribute('aria-label', `${label(dataset)}: ${mse ? 'position mean squared error' : 'Poisson’s ratio R squared'} against ${learning() ? `training networks, scored at rollout step ${horizon}` : 'rollout steps'}. Values are available in the runs table.`);
  const sameTraces = plot.data && plot.data.length === traces.length && traces.every((trace,i) => trace.uid === plot.data[i].uid);
  if (!learning() && !mse && animate && sameTraces && plot.data.some(trace => trace.y.some(finite)) && traces.some(trace => trace.y.some(finite)) && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    Plotly.animate(plot,{data:traces,traces:traces.map((_,i) => i)},
      {mode:'immediate',transition:{duration:650,easing:'cubic-in-out'},frame:{duration:650,redraw:true}});
  } else {
    Plotly.react(plot,traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,displayModeBar:true,
      modeBarButtonsToRemove:['select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d'],
      toImageButtonOptions:{format:'svg',filename:mse ? 'network-bench-mse' : 'network-bench-r2',width:1100,height:550}});
  }
  $(mse ? 'mse-range-note' : 'range-note').textContent = `${learning() ? `Test scores at step ${horizon} · ` : ''}${individual ? 'Individual seeds' : learning() ? 'Mean ± sample SD across split and training seeds' : 'Mean ± sample SD across training seeds'} · ${mse ? 'Lower MSE is better. Values use squared coordinate units.' : 'Negative scores are drawn at 0; hover and table show actual values.'}`;
}

function tableRuns() {
  return selectedRuns().filter(run => run.model.toLowerCase().includes($('search').value.trim().toLowerCase())
    && ($('seed').value === 'all' || String(run.seed) === $('seed').value));
}
const COLUMNS = [
  {key:'model',label:'Model / settings',value:run=>run.model},
  {key:'dataset',label:'Dataset',value:run=>label(run.dataset)},
  {key:'train_networks',label:'Training networks',value:run=>run.train_networks},
  {key:'seed',label:'Seed',value:run=>run.seed},
  {key:'parameters',label:'Parameters',value:run=>run.parameters},
  {key:'epoch',label:'Best epoch',value:run=>run.selected_epoch ?? run.trained_epochs ?? run.epochs},
  {key:'trained',label:'Trained epochs',value:run=>run.trained_epochs ?? run.epochs},
  {key:'val',label:'Validation score',value:run=>run.val_score},
  {key:'mse',label:'Position MSE ↓',value:run=>metricValues(run,'position_mse_by_step')[horizon]},
  {key:'relative',label:'Relative MSE ↓',value:run=>metricValues(run,'relative_mse_by_step')[horizon]},
  {key:'r2',label:'Test R² ↑',value:run=>run.poisson_r2[horizon]},
  {key:'diverged',label:'Diverged',value:run=>run.diverged}
];
function renderTable() {
  const columns = COLUMNS.filter(column => learning() || column.key !== 'train_networks');
  const runs = tableRuns(), column = columns.find(column=>column.key===tableSort.key) ?? columns.find(column=>column.key==='r2');
  runs.sort((a,b)=>{
    const av=column.value(a),bv=column.value(b);
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const order=typeof av==='string'?av.localeCompare(bv):av-bv;
    return tableSort.descending ? -order:order;
  });
  $('table-count').textContent=`${runs.length} runs · ${learning() ? `${trainSize} training networks · ` : ''}test metrics at step ${horizon}`;
  const head=node('tr');
  for (const column of columns) {
    const th=node('th'); th.scope='col';
    th.setAttribute('aria-sort',column.key===tableSort.key?(tableSort.descending?'descending':'ascending'):'none');
    const button=node('button',column.key==='r2'?`Test R² @${horizon} ↑`:column.key==='mse'?`Position MSE @${horizon} ↓`:column.key==='relative'?`Relative MSE @${horizon} ↓`:column.label);
    button.onclick=()=>{tableSort={key:column.key,descending:column.key===tableSort.key?!tableSort.descending:!['mse','relative'].includes(column.key)};renderTable();};
    th.append(button); head.append(th);
  }
  $('table-head').replaceChildren(head); $('table-body').replaceChildren();
  for (const run of runs) {
    const row=node('tr');
    for (const column of columns) {
      const cell=node('td'), value=column.value(run);
      if (column.key==='model') {
        const details=node('details'), summary=node('summary'), swatch=node('i',undefined,'swatch'); swatch.style.background=color(run.model);
        summary.append(swatch,node('span',run.model)); details.append(summary);
        const dl=node('dl');
        for (const [key,value] of Object.entries({...run.hyperparameters,training:run.training_mode ?? 'one_step',rollout_schedule:JSON.stringify(run.rollout_schedule ?? [[0,1]]),detach_rollout:run.detach_rollout ?? false,selection:run.select_by ?? 'last epoch',epoch_limit:run.epochs,early_stopping:run.early_stopping_patience == null ? 'disabled' : `${run.early_stopping_patience} validation checks`,early_stopped:run.early_stopped ?? false,run:run.name})) dl.append(node('dt',key),node('dd',String(value)));
        details.append(dl); cell.append(details);
      } else if (column.key==='seed') cell.append(node('span',String(value),'seed-pill'));
      else if (column.key==='parameters') cell.textContent=value.toLocaleString();
      else if (column.key==='mse') cell.textContent=scientific(value);
      else if (['val','relative','r2'].includes(column.key)) cell.textContent=number(value);
      else cell.textContent=value ?? '—';
      if (column.key==='r2') cell.className='score-cell';
      if (column.key==='val') cell.title=run.select_by ?? 'Last epoch; no selection metric';
      row.append(cell);
    }
    $('table-body').append(row);
  }
  if (!runs.length) {const row=node('tr'),cell=node('td',missingLearning() ? emptyResults() : 'No runs match these filters.','empty');cell.colSpan=columns.length;row.append(cell);$('table-body').append(row);}
  updateURL();
}
function downloadCSV() {
  const columns=['dataset','mode','training_mode','rollout_schedule','detach_rollout','comparison_id','model','seed',...(learning() ? ['split_seed','train_networks','split_id'] : []),'hyperparameters','parameters','epochs','trained_epochs','early_stopped','early_stopping_patience','selected_epoch','select_by','val_score','relative_mse','position_mse','diverged',`poisson_r2_${horizon}`,`position_mse_${horizon}`,`relative_mse_${horizon}`];
  const quote=value=>'"'+String(value ?? '').replaceAll('"','""')+'"';
  const rows=tableRuns().map(run=>columns.map(key=>quote(['hyperparameters','rollout_schedule'].includes(key)?JSON.stringify(run[key]):key.startsWith('poisson_r2_')?run.poisson_r2[horizon]:key.startsWith('position_mse_')?metricValues(run,'position_mse_by_step')[horizon]:key.startsWith('relative_mse_')?metricValues(run,'relative_mse_by_step')[horizon]:run[key])).join(','));
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
Promise.all(['results.json','learning-curves.json'].map(path => fetch(path).then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json();}))).then(([payload, learningPayload])=>{
  studies={rollouts:{...payload, runs:payload.runs.filter(run=>run.model!=='tiny_mlp')}, learning:learningPayload};
  data=studies.rollouts;
  if(!data.runs.length){$('notice').textContent='No completed runs have been published yet.';return;}
  enabled=new Set([...data.runs,...learningPayload.runs].map(run=>run.model));
  for(const key of datasets()){const button=node('button',label(key));button.dataset.value=key;button.onclick=()=>setDataset(key);$('datasets').append(button);}
  for (const [view, payload] of Object.entries(studies)) for(const key of datasets()){
    for (const distribution of ['normal', 'ood']) for (const training of ['one_step', 'multi_step']) cohorts[`${view}:${key}:${distribution}:${training}`]=payload.runs.find(run=>run.dataset===key && (run.mode ?? 'normal') === distribution && (run.training_mode ?? 'one_step') === training)?.comparison_id;
  }
  for (const button of $('studies').children) button.onclick=()=>setStudy(button.dataset.study);
  for (const button of $('modes').children) button.onclick=()=>{mode=button.dataset.mode;setDataset(dataset);};
  for (const button of $('training-modes').children) button.onclick=()=>{trainingMode=button.dataset.training;setDataset(dataset);};
  for(const model of [...enabled].sort((a,b)=>MODELS.indexOf(a)-MODELS.indexOf(b))){
    const button=node('button'),dot=node('i',modelStyle(model).glyph,'swatch');dot.style.color=color(model);button.dataset.model=model;button.title=DESCRIPTIONS[model] ?? model;
    button.append(dot,node('span',model),node('span',undefined,'model-parameters'));button.onclick=()=>{topModels=false;enabled.has(model)?enabled.delete(model):enabled.add(model);render();};$('legend').append(button);
  }
  for(const seed of [...new Set(data.runs.map(run=>run.seed))].sort((a,b)=>a-b)) for(const id of ['plot-seed','seed']) $(id).append(new Option(`Seed ${seed}`,String(seed)));
  $('cohort').onchange=()=>{cohorts[cohortKey()]=$('cohort').value;topModels=true;render();};
  $('train-size').onchange=()=>{trainSize=Number($('train-size').value);render();};
  $('split-seed').onchange=()=>{renderSplit();updateURL();};
  for(const id of ['bands','plot-seed','line-mode'])$(id).onchange=render;
  $('horizon').oninput=()=>{horizon=JSON.parse($('horizon').dataset.steps)[$('horizon').value];render();};
  $('top-models').onclick=()=>{topModels=true;render();};
  $('reset-models').onclick=()=>{topModels=false;enabled=new Set(data.runs.map(run=>run.model));render();};
  $('search').oninput=renderTable;$('seed').onchange=renderTable;$('download').onclick=downloadCSV;
  $('notice').hidden=true;$('dashboard').hidden=false;restoreURL();
}).catch(error=>{$('notice').hidden=false;$('notice').textContent=`Results could not be loaded (${error.message}). Please reload, or download the JSON from the link above.`;});
