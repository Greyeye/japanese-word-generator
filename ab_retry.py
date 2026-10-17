"""A/B: does giving the retry its own prior output kill the duplication?"""
import json, urllib.request, time, re, random

API="http://192.168.0.12:8889"; TOK="sk-unsloth-eb68e3f73915781686d00d2804cecbe8"
MODEL="lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF"
norm=lambda s: re.sub(r'[。．！？、…\s「」『』（）・]','',s)

GP="〜があります"; PAT="Place に Noun が あります"
def prompt(n, avoid=None):
    av = ""
    if avoid:
        av = "\nすでに次の例文を作りました。これらと同じ文は絶対に作らないでください:\n" + \
             "\n".join(f"  {s}" for s in avoid)
    return f"""日本語のN5例文を{n}個作ってください。
文法: {PAT}（{GP}）
です・ます体。20文字以内。1文のみ。
禁止: 文頭にこの/その/あの。「彼/彼女」。
{av}
出力: JSONのみ {{"sentences":["...","..."]}}"""

def call(p, temp=0.85):
    b={"model":MODEL,"max_tokens":3000,"messages":[{"role":"user","content":p}],
       "reasoning_effort":"none","temperature":temp}
    r=urllib.request.Request(API+"/v1/chat/completions",data=json.dumps(b).encode(),
      headers={"Authorization":"Bearer "+TOK,"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(r,timeout=300))
    txt=re.sub(r"```(?:json)?","",d["choices"][0]["message"].get("content") or "").strip()
    i,j=txt.find("{"),txt.rfind("}")
    try: got=json.loads(txt[i:j+1])["sentences"]
    except Exception: got=re.findall(r'"([^"]{2,40})"', txt[i:j+1])
    return time.time()-t0, [s for s in got if s.strip().lower()!="sentences"]

overlap_noc, overlap_ctx = [], []
for trial in range(3):
    random.seed(100+trial)
    # --- attempt 1: baseline batch
    _, first = call(prompt(15))
    fset = {norm(s) for s in first}

    # --- attempt 2 WITHOUT context (current pipeline behaviour)
    _, a2 = call(prompt(10))
    ov_n = len({norm(s) for s in a2} & fset)

    # --- attempt 2 WITH the prior output as an exclusion list
    _, a3 = call(prompt(10, avoid=first))
    ov_c = len({norm(s) for s in a3} & fset)

    overlap_noc.append((len(a2), ov_n))
    overlap_ctx.append((len(a3), ov_c))
    print(f"trial {trial+1}: attempt1={len(first)}  "
          f"retry_no_context={len(a2)} returned, {ov_n} were repeats  |  "
          f"retry_with_context={len(a3)} returned, {ov_c} were repeats", flush=True)

def summarise(name, rows):
    tot=sum(r[0] for r in rows); dup=sum(r[1] for r in rows)
    print(f"  {name:<22} {tot} sentences, {dup} repeats "
          f"({100*dup/max(tot,1):.0f}% waste)")
print()
print("=== retry duplication ===")
summarise("no context (current)", overlap_noc)
summarise("with context (fix)", overlap_ctx)
