from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().with_name('.env'))
from live_engine import run_live, VerificationError
import csv,io,json,re,os
from pathlib import Path
from semantic_engine import classify
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
P=Path(__file__).parent
SEEDS=json.loads((P/'seed_facts.json').read_text(encoding='utf-8'))
app=FastAPI(title='ClaimWeave AI API')
app.mount('/static',StaticFiles(directory=P/'static'),name='static')

def segment(t):
 return [x.strip(' \t\r\n•-') for x in re.split(r'(?<=[.!?])\s+|\n+',t) if len(x.strip())>7][:100]
def entries(extra):
 return SEEDS+extra
class Verify(BaseModel):
 text:str=Field(min_length=1,max_length=8000)
 evidence:list[dict]=Field(default_factory=list)
 semantic:bool=False
 mode:Literal["live","evidence","enhanced"]="live"
@app.get('/')
def home():return FileResponse(P/'static'/'index.html')
@app.get('/api/demo')
def demo():return {'count':len(SEEDS),'samples':['The capital of India is New Delhi.','The capital of Australia is Sydney.','The private laboratory on Mars was founded in 2041.']}
@app.post('/api/upload')
async def upload(file:UploadFile=File(...)):
 raw=await file.read()
 if len(raw)>8_000_000:raise HTTPException(413,'Maximum upload size is 8 MB')
 try:t=raw.decode('utf-8-sig')
 except UnicodeDecodeError:raise HTTPException(400,'Please upload UTF-8 CSV, TXT or JSON')
 ext=Path(file.filename or '').suffix.lower()
 out=[]
 if ext=='.csv':
  for row in csv.DictReader(io.StringIO(t)):
   supported=row.get('supported') or row.get('evidence') or row.get('statement') or row.get('claim') or ''
   if supported.strip():out.append({'supported':supported.strip(),'contradicted':row.get('contradicted',''),'source_title':row.get('source_title') or row.get('source') or file.filename,'source_url':row.get('source_url','')})
 elif ext=='.json':
  try:items=json.loads(t)
  except ValueError:raise HTTPException(400,'Invalid JSON')
  if not isinstance(items,list):raise HTTPException(400,'JSON must contain a list of records')
  for row in items:
   if isinstance(row,dict) and row.get('supported'):out.append(row)
 elif ext=='.txt':
  out=[{'supported':line.strip(),'contradicted':'','source_title':file.filename,'source_url':''} for line in t.splitlines() if len(line.strip())>12 and not line.lstrip().startswith(('#','Source:','URL:'))]
 else:raise HTTPException(400,'Supported formats: CSV, TXT, JSON')
 return {'entries':out[:2000],'count':min(len(out),2000),'filename':file.filename}
@app.post('/api/verify')
def verify(body:Verify):
 if body.mode=="live":
  try:return run_live(body.text)
  except ValueError as exc:
   raise HTTPException(400,str(exc))
  except VerificationError as exc:
   raise HTTPException(503,str(exc))
 claims=segment(body.text)
 if not claims:raise HTTPException(400,'Enter one or more factual claims')
 knowledge=entries(body.evidence)
 results=[]
 for claim in claims:
  outcome=classify(claim,knowledge,semantic=body.semantic or body.mode=="enhanced")
  results.append({'claim':claim,'confidence':None,**outcome})
 counts={x:sum(r['label']==x for r in results) for x in ['SUPPORTED','CONTRADICTED','INSUFFICIENT EVIDENCE']}
 return {'results':results,'counts':counts,'total':len(results),'disclaimer':'Prototype: verdicts are limited to supplied evidence. Semantic models are optional and may make mistakes; unsupported claims abstain. Sources are references, not independently audited.'}

@app.get('/api/status')
def status():
 return {'groq_configured':bool(os.getenv('GROQ_API_KEY','').strip()),
         'serper_configured':bool(os.getenv('SERPER_API_KEY','').strip()),
         'model':os.getenv('GROQ_MODEL','openai/gpt-oss-120b'),
         'default_mode':'live','note':'Configuration check only; API validity is checked during analysis.'}
