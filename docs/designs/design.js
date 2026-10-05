/* Atlas switches metric panels; each design uses the shared benchmark engine. */
for (const button of document.querySelectorAll('[data-metric]')) {
  button.onclick = () => {
    document.body.dataset.activeMetric = button.dataset.metric;
    for (const tab of document.querySelectorAll('[data-metric]')) tab.setAttribute('aria-pressed', String(tab === button));
    renderChart(false, button.dataset.metric === 'mse' ? 'position_mse_by_step' : 'poisson_r2');
    Plotly.Plots.resize(document.getElementById(button.dataset.metric === 'mse' ? 'mse-plot' : 'plot'));
  };
}
