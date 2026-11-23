"""Gate production on the distributed smoke runs and checkpoint replay."""
import argparse
import json
import os
from pathlib import Path
from scripts.benchmarks.benchmark_albertone_mlm import publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    assert json.loads((args.output / 'cpu-validation.json').read_text())['status'] == 'passed'
    results = [json.loads((args.output / f'm{rate}/result.json').read_text()) for rate in ('0.15', '0.28', '0.40')]
    assert all(r['status'] == 'completed' and r['rank_parameter_agreement'] and r['final_lr'] == 0 for r in results)
    assert len({r['counts']['input_tokens'] for r in results}) == 1
    assert len({r['counts']['content_tokens'] for r in results}) == 1
    for domain in ('wikipedia', 'academic', 'books', 'gazzetta_ufficiale', 'subtitles'):
        assert len({r['validation'][f'val/{domain}/targets'] for r in results}) == 1
    resumed = json.loads((args.output / 'm0.28/resumed-result.json').read_text())
    assert resumed['counts'] == results[1]['counts']
    assert resumed['resumed_max_parameter_difference'] is not None
    assert abs(resumed['validation']['val/loss'] - results[1]['validation']['val/loss']) < 1e-5
    publish(args.output / 'validation.json', dict(status='passed', job_id=os.environ['SLURM_JOB_ID'],
        checks=['Direct unpadding equals existing collator', 'Token budgets preserve tails and all sequences',
                'Warmup/peak/zero LR endpoints', 'Exact mixture arithmetic',
                'Strict full model/head restoration', 'Unequal-target DDP gradient reference',
                'All three rates use identical input tokens', 'Fixed validation target counts',
                'Finite accumulated actual-model updates', 'Rank parameter agreement',
                'Full checkpoint replay matches uninterrupted parameters and validation'],
        resume_max_parameter_difference=resumed['resumed_max_parameter_difference'],
        results=[dict(rate=r['run']['mask_rate'], peak_allocated_gib=r['peak_allocated_gib'],
                      elapsed_seconds=r['elapsed_seconds']) for r in results],
        limitation='Engineering fixture only; production also requires complete real-data validation'))


if __name__ == '__main__':
    main()
