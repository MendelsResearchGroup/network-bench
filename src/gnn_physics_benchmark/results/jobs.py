"""Execute one prepared experiment job and record completion or failure."""
import json
import time
import traceback
from . import cache


def run_job(config, job, index, root):
    from .. import benchmark, models
    from ..evaluate import evaluate_model
    from ..normalization import Normalizer

    done = root / 'done' / f'{index}.json'
    started = time.monotonic()
    try:
        if job.get('kind', 'train') == 'train':
            result = benchmark.run(config, root=root / 'runs')
        else:
            model = models.build(config.model, config.graph, Normalizer(config.graph.dim))
            measured = evaluate_model(model, benchmark.load_data(config, config.split.explicit['test']), config)
            measured.update(trained_epochs=0, early_stopped=False)
            directory = cache.run_dir(config, config.split.explicit, root / 'runs')
            cache.write_result(directory, config, config.split.explicit, metrics=measured)
            result = cache.read_result(directory)
        record = {**job, 'index': index, 'directory': result['directory'],
                  'metrics': result['metrics'], 'seconds': time.monotonic() - started}
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(record, indent=1) + '\n')
        (root / 'failed' / f'{index}.txt').unlink(missing_ok=True)
        print('COMPLETED', index, json.dumps(result['metrics']), flush=True)
        return record
    except Exception:
        (root / 'failed').mkdir(parents=True, exist_ok=True)
        (root / 'failed' / f'{index}.txt').write_text(traceback.format_exc())
        raise
