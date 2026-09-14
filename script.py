import json, pathlib, re, collections
p = pathlib.Path(r'data\2026-09-14\constellation_dataset_qwen2_7b')
c = collections.Counter()
for s in p.glob('scenario_*/ollama_prompts_combined.json'):
    for t in json.loads(s.read_text(encoding='utf-8'))['tasks']:
        for m in set(re.findall(r'\[[^\]]{1,40}\]', t['generated_output'])):
            c[m] += 1
for k, v in c.most_common(15):
    print(v, k)