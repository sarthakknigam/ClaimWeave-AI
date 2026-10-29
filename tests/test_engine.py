import unittest,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from semantic_engine import classify
facts=json.loads((pathlib.Path(__file__).resolve().parents[1]/'seed_facts.json').read_text())
class Tests(unittest.TestCase):
 def test_capital_supported(self):
  self.assertEqual(classify('Canberra is the capital of Australia.',facts,True)['label'],'SUPPORTED')
 def test_capital_contradicted(self):
  self.assertEqual(classify('The capital of Australia is Sydney.',facts,True)['label'],'CONTRADICTED')
 def test_capital_paraphrase(self):
  self.assertEqual(classify('The capital city of India is New Delhi.',facts,True)['label'],'SUPPORTED')
 def test_unknown(self):
  self.assertEqual(classify('Secret model XQ-999 won a private award.',facts,True)['label'],'INSUFFICIENT EVIDENCE')
 def test_existing_fact(self):
  self.assertEqual(classify('The solar system has eight planets.',facts,True)['label'],'SUPPORTED')
if __name__=='__main__':unittest.main()
