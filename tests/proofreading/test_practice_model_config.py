import pytest
from scripts.proofreading.run_cb2_practice import model_configuration


def test_provider_specific_reasoning_and_baseline_defaults():
    assert model_configuration('gpt-6-astra')==('responses',{'reasoning':{'effort':'max'}})
    assert model_configuration('claude-opus-5-5')==('anthropic',{'cache_control':{'type':'ephemeral','ttl':'5m'}})
    assert model_configuration('claude-opus-5-5','max')==('anthropic',{'output_config':{'effort':'max'},'cache_control':{'type':'ephemeral','ttl':'5m'}})
    with pytest.raises(ValueError):model_configuration('unknown')
    with pytest.raises(ValueError):model_configuration('gpt-6-astra','xhigh')


def test_cache_opt_out_and_long_ttl():
    assert model_configuration('claude-opus-5-5', prompt_cache='off')[1]['cache_control'] is None
    assert model_configuration('claude-opus-5-5', prompt_cache='1h')[1]['cache_control']['ttl']=='1h'
