import json
import pytest
from proofreading.trial_report import comparison_report
from proofreading.experiment import load_config


def setup(tmp_path):
    configs=tmp_path/'configs';configs.mkdir()
    outputs=tmp_path/'outputs';run=outputs/'trial';run.mkdir(parents=True)
    cfg={'schema':1,'mode':'cb2','benchmark':'../benchmark','model':{'provider':'responses','name':'test'},'phase':'train','cases':['corrected','failed','missing','preflight']}
    (configs/'trial.json').write_text(json.dumps(cfg))
    (run/'config.json').write_text(json.dumps(load_config(configs/'trial.json')))
    return configs,outputs,run


def test_report_keeps_failures_missing_and_preflight_out_of_successes(tmp_path):
    configs,outputs,run=setup(tmp_path)
    human={'initial':{'vi_total_bits':1.},'final':{'vi_total_bits':0.}}
    cert={'initial':{'false_merge_voxel_pairs':2**60},
          'final':{'false_merge_voxel_pairs':2**60+1,'scored_voxels':500}}
    for name,data in [('corrected',{'status':'finished','scores':{'human':human,'certified':cert}}),
                      ('preflight',{'status':'preflight','initial_scores':{'human':human}})]:
        d=run/name;d.mkdir();(d/'result.json').write_text(json.dumps(data))
    d=run/'failed';d.mkdir();(d/'failure.json').write_text('{}')
    report=comparison_report(configs,outputs)
    assert report['planned_episodes']==4
    assert report['status_counts']=={'finished':1,'failed':1,'not_started':1,'preflight':1}
    row=report['episodes'][0]
    assert row['target_partition_matches'] and row['certified_merge_pairs_increased']
    assert row['certified_false_merge_pair_delta']==1
    assert not report['episodes'][-1]['scored']
    (run/'corrected'/'failure.json').write_text('{}')
    assert comparison_report(configs,outputs)['episodes'][0]['status']=='failed'


def test_report_rejects_mixed_configuration(tmp_path):
    configs,outputs,run=setup(tmp_path)
    (run/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='differs'):comparison_report(configs,outputs)
