import json,struct,urllib.request,sys
info=json.load(open('/tmp/ti/ollama.json'))
class R:
    def __init__(s,url): s.url=url;s.buf=b'';s.pos=0
    def need(s,n):
        while len(s.buf)<s.pos+n:
            a=len(s.buf);req=urllib.request.Request(s.url,headers={'Range':f'bytes={a}-{a+max(8<<20,n)-1}'})
            s.buf+=urllib.request.urlopen(req,timeout=120).read()
    def rd(s,n): s.need(n);b=s.buf[s.pos:s.pos+n];s.pos+=n;return b
    def u(s,f): return struct.unpack('<'+f,s.rd(struct.calcsize(f)))[0]
    def st(s): return s.rd(s.u('Q')).decode('utf8','replace')
SZ={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def val(r,t):
    if t==8: return r.st()
    if t==9:
        et=r.u('I');n=r.u('Q')
        if et==8: [r.st() for _ in range(n)]; return f'<arr str {n}>'
        r.rd(n*struct.calcsize(SZ[et])); return f'<arr {n}>'
    return r.u(SZ[t])
for slug,i in info.items():
    repo,tag=(i['name'].split(':')+['latest'])[:2]
    r=R(f'https://registry.ollama.ai/v2/library/{repo}/blobs/{i["model_layer"]}')
    try:
        assert r.rd(4)==b'GGUF'; r.u('I'); r.u('Q'); n=r.u('Q'); meta={}
        for _ in range(n):
            k=r.st(); t=r.u('I'); v=val(r,t)
            if 'chat_template' in k or k in('general.name','general.architecture','general.basename','general.organization','general.base_model.0.repo_url','general.repo_url'): meta[k]=v
        json.dump(meta,open(f'/tmp/ti/gguf/{slug}.json','w'),indent=1)
        print(slug, len(r.buf)>>20,'MB', [k for k in meta])
    except Exception as e: print(slug,'ERR',e)
