"""Account for every planned episode in a frozen prompt comparison, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
from .experiment import load_config


def comparison_report(config_directory, output_directory):
    configs=Path(config_directory);outputs=Path(output_directory);rows=[]
    for path in sorted(configs.glob('*.json')):
        cfg=load_config(path);directory=outputs/path.stem
        saved=directory/'config.json'
        if saved.exists() and json.loads(saved.read_text())!=cfg:
            raise ValueError(f'Configuration differs from planned trial: {path.stem}')
        for case in cfg['cases']:
            row={'trial':path.stem,'case_id':case,'model':cfg['model']['name'],
                 'phase':cfg['phase'],'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'status':'not_started','scored':False}
            episode=directory/case;result=episode/'result.json'
            failure=episode/'failure.json'
            if result.exists() or failure.exists() or (episode/'intent.json').exists():
                if not saved.exists():raise ValueError(f'Missing configuration provenance: {path.stem}')
                row['status']='incomplete'
            if failure.exists():row['status']='failed'
            if result.exists():
                data=json.loads(result.read_text());row['status']=data['status']
                harness=data.get('harness',{})
                row.update(actions=harness.get('actions'),usage=harness.get('usage_summary'))
                scores=data.get('scores',{})
                if 'human' in scores and 'certified' in scores:
                    human,cert=scores['human'],scores['certified']
                    before,after=human['initial']['vi_total_bits'],human['final']['vi_total_bits']
                    merges=cert['final']['false_merge_voxel_pairs']-cert['initial']['false_merge_voxel_pairs']
                    row.update(scored=True,human_initial_vi=before,human_final_vi=after,
                        human_vi_improvement=before-after,
                        target_partition_matches=after<=1e-12,
                        certified_false_merge_pair_delta=merges,
                        certified_merge_pairs_increased=merges>0,
                        certified_scored_voxels=cert['final']['scored_voxels'],
                        unknown_reference_voxels=cert['final'].get('unknown_reference_voxels_in_scope'),
                        scope_attachment=human.get('scope_attachment'),
                        human_spec_sha256=human.get('frozen_spec_sha256'),
                        certified_spec_sha256=cert.get('frozen_spec_sha256'))
            if failure.exists():row['status']='failed'
            rows.append(row)
    if not rows:raise ValueError('No planned episodes')
    counts={}
    for row in rows:counts[row['status']]=counts.get(row['status'],0)+1
    return {'planned_episodes':len(rows),'status_counts':counts,'episodes':rows,
            'interpretation':'All planned episodes are included. Target agreement is not whole-neuron correctness. '
            'Certified merge-pair deltas are aggregate changes, not an exhaustive extension audit. '
            'Unknown reference material is unresolved; absent results are not successes. '
            'Training comparisons guide iteration and are not held-out performance estimates.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('configs',type=Path);p.add_argument('outputs',type=Path)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=comparison_report(a.configs,a.outputs)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
