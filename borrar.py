import json, pathlib
FUGA = ['stage 1', 'stage 2', 'reasoning', 'which exact string', 'copy it verbatim', 'a) ', 'b) ']
for f in pathlib.Path('data').rglob('2026-09-21/chain_of_thought/**/scenario_1/ollama_prompts_combined.json'):
    tot = fuga = 0
    for t in json.loads(f.read_text(encoding='utf-8'))['tasks']:
        tot += 1
        tx = t['generated_output'].lower()
        if any(k in tx for k in FUGA) or len(tx) > 2500:
            fuga += 1
            print('  FUGA:', t['task_id'], repr(t['generated_output'][:150]))
    print(f.parent.parent.name, f'-> {fuga}/{tot} contaminadas')