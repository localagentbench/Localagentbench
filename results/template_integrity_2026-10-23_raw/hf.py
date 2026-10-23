import json,urllib.request,os
C={
'deepseek-r1-14b':['deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'],
'deepseek-r1-8b':['deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'],
'gemma3-4b':['google/gemma-3-4b-it'],
'gemma4-12b':['google/gemma-4-12b-it','google/gemma-4-12B-it'],
'gemma4-26b':['google/gemma-4-26b-a4b-it','google/gemma-4-26B-A4B-it'],
'gemma4-e4b':['google/gemma-4-e4b-it','google/gemma-4-E4B-it'],
'gpt-oss-20b':['openai/gpt-oss-20b'],
'hermes3-8b':['NousResearch/Hermes-3-Llama-3.1-8B'],
'llama3.1-8b':['meta-llama/Llama-3.1-8B-Instruct'],
'llama3.2-3b':['meta-llama/Llama-3.2-3B-Instruct'],
'mistral-7b':['mistralai/Mistral-7B-Instruct-v0.3'],
'mistral-nemo-12b':['mistralai/Mistral-Nemo-Instruct-2407'],
'mistral-small-24b':['mistralai/Mistral-Small-24B-Instruct-2501'],
'mistral-small3.2-24b':['mistralai/Mistral-Small-3.2-24B-Instruct-2506'],
'phi4-14b':['microsoft/phi-4'],
'qwen2.5-14b':['Qwen/Qwen2.5-14B-Instruct'],'qwen2.5-7b':['Qwen/Qwen2.5-7B-Instruct'],
'qwen2.5-coder-14b':['Qwen/Qwen2.5-Coder-14B-Instruct'],'qwen2.5-coder-7b':['Qwen/Qwen2.5-Coder-7B-Instruct'],
'qwen3-14b':['Qwen/Qwen3-14B'],'qwen3-8b':['Qwen/Qwen3-8B'],
'qwen3-30b-a3b':['Qwen/Qwen3-30B-A3B-Thinking-2507'],'qwen3-4b':['Qwen/Qwen3-4B-Thinking-2507'],
'qwen3-coder-30b':['Qwen/Qwen3-Coder-30B-A3B-Instruct'],'qwen3-vl-8b':['Qwen/Qwen3-VL-8B-Instruct','Qwen/Qwen3-VL-8B-Thinking'],
'qwen3.5-27b':['Qwen/Qwen3.5-27B'],'qwen3.5-9b':['Qwen/Qwen3.5-9B'],
'qwen3.6-27b':['Qwen/Qwen3.6-27B'],'qwen3.8-27b':['Qwen/Qwen3.8-27B','Qwen/Qwen3.8-27B-0814'],
}
os.makedirs('/tmp/ti/hf',exist_ok=True);res={}
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'x'}),timeout=30).read().decode()
for s,repos in C.items():
    res[s]={'tried':[]}
    for repo in repos:
        for fn in ['chat_template.jinja','tokenizer_config.json','chat_template.json']:
            u=f'https://huggingface.co/{repo}/resolve/main/{fn}'
            try:
                t=get(u)
                if fn!='chat_template.jinja':
                    t=json.loads(t).get('chat_template')
                    if isinstance(t,list): t={x['name']:x['template'] for x in t}.get('default')
                    if not t: res[s]['tried'].append(u+' (no chat_template)'); continue
                open(f'/tmp/ti/hf/{s}.jinja','w').write(t); res[s].update(src=u,repo=repo); break
            except Exception as e: res[s]['tried'].append(f'{u} ({e})')
        if 'src' in res[s]: break
    print(s,res[s].get('src') or res[s]['tried'][-1])
json.dump(res,open('/tmp/ti/hf.json','w'),indent=1)
