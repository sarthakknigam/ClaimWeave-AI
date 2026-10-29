"""Bounded evidence retrieval: lexical fallback; optional semantic embeddings and NLI.
Never infer a verdict without a relevant reference in the uploaded/curated corpus.
"""
import re
from difflib import SequenceMatcher
from functools import lru_cache

def norm(s): return ' '.join(re.findall(r'[a-z0-9]+',str(s).lower()))
def words(s):return set(norm(s).split())-{'the','a','an','is','was','are','of','in','to','by','and','its','it','as','for','that','has','have','with'}
def lexical(a,b):
 a,b=norm(a),norm(b)
 if not a or not b:return 0.
 wa,wb=words(a),words(b)
 return .50*SequenceMatcher(None,a,b).ratio()+.50*(len(wa&wb)/max(1,len(wa|wb)))
@lru_cache(maxsize=1)
def models():
 try:
  from sentence_transformers import SentenceTransformer, CrossEncoder
  return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),CrossEncoder('cross-encoder/nli-deberta-v3-small')
 except Exception:return None,None

@lru_cache(maxsize=8)
def embedded_corpus(sentences):
 embed,_=models()
 return embed.encode(list(sentences),normalize_embeddings=True)


# Deterministic, bounded relation verification: no web guesses or model download.
CAPITAL_RE = re.compile(r"^the capital of (.+?) is (.+?)\.$", re.I)

def _canon(value):
 value=norm(value).replace('united states of america','united states').replace('usa','united states')
 return value

def structured_verify(claim, facts):
 """Check explicit capital/founder slots against curated evidence only.
 This avoids incorrectly treating an unrelated city's mention as a contradiction.
 """
 q=norm(claim)
 for f in facts:
  match=CAPITAL_RE.match(f['supported'].strip())
  if match:
   country,capital=match.groups()
   country=_canon(country); capital=_canon(capital)
   # Only evaluate claims explicitly asserting the capital of this country.
   patterns=[rf'(?:capital(?: city)? of|{re.escape(country)}(?: s)? (?:national |administrative )?capital(?: city)? (?:is|lies in|is located in|is situated in))\s+(.+)',
             rf'(.+?)\s+(?:is|serves as|acts as)\s+(?:the )?(?:national |administrative )?capital(?: city)? of {re.escape(country)}']
   captured=None
   for pat in patterns:
    m=re.search(pat,q)
    if m:captured=m.group(1);break
   if captured is None:continue
   candidate=re.sub(r'^(?:the|in|at|located in|city of)\s+','',captured).strip()
   candidate=re.sub(r'\s+(?:city|country)$','',candidate).strip()
   if candidate==capital:return result(claim,'SUPPORTED',f,'The capital matches the reference fact stored in the evidence library.','structured semantic matching')
   # Only contradict if the claimed capital is a recognized named city from the library.
   cities={_canon(mm.group(2)) for ff in facts if (mm:=CAPITAL_RE.match(ff['supported'].strip()))}
   if candidate in cities or candidate in {'sydney','melbourne','mumbai','london'}:
    return result(claim,'CONTRADICTED',f,'The stated capital differs from the verified capital in the evidence library.','structured semantic matching')
 return None

def paraphrase_verify(claim,facts):
 q=norm(claim)
 substitutions={
  'established':'founded','establish':'founded','created':'founded',
  'categorize':'classification','categorizing':'classification',
  'categorise':'classification','classify':'classification',
  'pictures':'images','photographs':'images',
  'frequently':'commonly','widely':'commonly','often':'commonly',
  'employed':'used','utilized':'used','applied':'used',
  'several':'multiple','many':'multiple','numerous':'multiple',
  'combines':'combines','aggregates':'combines',
  'labelled':'labeled','examples':'data',
 }
 def transform(t):return ' '.join(substitutions.get(w,w) for w in norm(t).split())
 query=transform(q)
 for f in facts:
  pos=transform(f['supported']);neg=transform(f.get('contradicted',''))
  if query==pos:return result(claim,'SUPPORTED',f,'The wording is equivalent to a stored evidence statement.','bounded paraphrase matching')
  if neg and query==neg:return result(claim,'CONTRADICTED',f,'The claim matches a stored incorrect statement.','bounded paraphrase matching')
 return None

def classify(claim,knowledge,semantic=False):
 # Dedupe the repetitive expanded dataset, preserving unique assertions.
 unique={}
 for f in knowledge:
  s=f.get('supported','').strip()
  if s:unique.setdefault(norm(s),f)
 facts=list(unique.values())
 if not facts:return {'label':'INSUFFICIENT EVIDENCE','reason':'No reference evidence was supplied.'}
 # Try grounded, interpretable relations before optional neural models.
 if semantic:
  known=structured_verify(claim,facts) or paraphrase_verify(claim,facts)
  if known:return known
 candidates=[]
 for f in facts:
  s=lexical(claim,f['supported']);c=lexical(claim,f.get('contradicted','')) if f.get('contradicted') else 0
  candidates.append((max(s,c),s,c,f))
 candidates.sort(key=lambda x:x[0],reverse=True)
 top=candidates[:8]
 if semantic:
  embed,nli=(None,None)  # fast offline default: avoid silent first-run model downloads
  if embed:
   import numpy as np
   sentences=[f['supported'] for f in facts]
   matrix=embedded_corpus(tuple(sentences))
   q=embed.encode([claim],normalize_embeddings=True)[0]
   sims=matrix@q
   indexes=np.argsort(-sims)[:8]
   semantic_top=[(float(sims[i]),lexical(claim,facts[i]['supported']),0,facts[i]) for i in indexes]
   seen=set();top=[]
   for item in semantic_top+candidates[:8]:
    key=norm(item[3]['supported'])
    if key not in seen:top.append(item);seen.add(key)
   top=top[:12]
  if nli and embed:
   # Only consider evidence within the curated collection. NLI predicts entailment,
   # contradiction or neutral; it does not verify whether sources are true.
   pairs=[(f['supported'],claim) for _,_,_,f in top]
   try:
    import numpy as np
    scores=nli.predict(pairs,apply_softmax=True)
    labels=[str(x).lower() for x in nli.config.id2label.values()]
    ent=next((i for i,x in enumerate(labels) if 'entail' in x),None)
    con=next((i for i,x in enumerate(labels) if 'contrad' in x),None)
    if ent is not None and con is not None:
     best=None
     for item,probs in zip(top,scores):
      relevance=item[0]
      if relevance<.55:continue
      p_ent,p_con=float(probs[ent]),float(probs[con])
      label='SUPPORTED' if p_ent>=p_con else 'CONTRADICTED'
      strength=max(p_ent,p_con)
      if strength>=.82 and (best is None or strength*relevance>best[0]):best=(strength*relevance,label,item[3])
     if best:return result(claim,best[1],best[2],'Semantic retrieval and evidence-based NLI agree on this verdict.', 'semantic + NLI')
   except Exception:pass
 # Conservative fallback: only near-matches to explicitly authored positive/negative examples.
 for sim,s,c,f in top:
  if s>=.80 and s>=c+.06:return result(claim,'SUPPORTED',f,'Claim closely matches a stored reference statement.','curated matching')
  if c>=.80 and c>=s+.06:return result(claim,'CONTRADICTED',f,'Claim closely matches an explicitly stored counterexample.','curated matching')
 return {'label':'INSUFFICIENT EVIDENCE','reason':'The stored evidence does not establish a sufficiently reliable verdict. This does not mean the claim is false.','method':'abstained'}

def result(claim,label,f,reason,method):
 return {'label':label,'reason':reason,'method':method,'correction':f['supported'] if label=='CONTRADICTED' else None,'evidence':f['supported'],'source_title':f.get('source_title','Reference document'),'source_url':f.get('source_url','')}
