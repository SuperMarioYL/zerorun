import json
from zerorun.probe import GPUProbe
from zerorun.registry.schema import ModelEntry,QuantConfig
from zerorun.picker import pick,best_pick
entry=ModelEntry(model='demo-model',hf_id='example/demo',chat_template_ref='demo',quants=(QuantConfig(format='Q8_0',backend='llama.cpp',min_vram_gb=32),QuantConfig(format='Q4_K_M',backend='llama.cpp',min_vram_gb=16)))
gpu=GPUProbe(vram_gb=24,compute_cap=8.9,arch_family='ada',name='simulated',source='fixture')
for p in pick(gpu,entry):print(json.dumps({'quant':p.quant.format,'feasible':p.feasible,'reasons':p.reasons}))
print('selected:',best_pick(gpu,entry).quant.format)
