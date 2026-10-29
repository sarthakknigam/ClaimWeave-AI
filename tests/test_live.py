"""Deterministic regressions; mocked providers are not a live accuracy evaluation."""
import os
import sys
import pathlib
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import live_engine as e
from fastapi.testclient import TestClient
from server import app

SOURCE = {'title': 'College contact', 'url': 'https://example.org/contact',
          'snippet': 'Pranveer Singh Institute of Technology, Kanpur, Uttar Pradesh.', 'kind': 'search_snippet'}

class LiveTests(unittest.TestCase):
    def test_default_api_is_live(self):
        with patch('server.run_live', return_value={'mode':'live'}) as run:
            r = TestClient(app).post('/api/verify', json={'text':'PSIT is in Lucknow.'})
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.json()['mode'],'live')
            run.assert_called_once()

    def test_default_html_is_live(self):
        html = TestClient(app).get('/').text
        self.assertIn('value="live" selected',html)

    def test_citation_copied_not_generated(self):
        reply={'label':'CONTRADICTED','reason':'The listed address is in Kanpur.',
               'correction':'PSIT is in Kanpur.','source_ids':[1], 'evidence':'An invented quote'}
        with patch.object(e,'_llm',return_value=reply):
            result=e.classify_live('PSIT is in Lucknow.',[SOURCE])
        self.assertEqual(result['label'],'CONTRADICTED')
        self.assertEqual(result['evidence'],SOURCE['snippet'])
        self.assertEqual(result['source_url'],SOURCE['url'])

    def test_supported(self):
        with patch.object(e,'_llm',return_value={'label':'SUPPORTED','reason':'Address confirms city.','source_ids':[1]}):
            self.assertEqual(e.classify_live('PSIT is in Kanpur.',[SOURCE])['label'],'SUPPORTED')

    def test_invalid_citation_repairs(self):
        replies=[{'label':'SUPPORTED','reason':'A','source_ids':[90]},
                 {'label':'SUPPORTED','reason':'B','source_ids':[1]}]
        with patch.object(e,'_llm',side_effect=replies) as llm:
            self.assertEqual(e.classify_live('PSIT is in Kanpur.',[SOURCE])['label'],'SUPPORTED')
            self.assertEqual(llm.call_count,2)

    def test_invalid_citation_not_insufficient(self):
        with patch.object(e,'_llm',return_value={'label':'SUPPORTED','reason':'A','source_ids':[99]}):
            with self.assertRaises(e.VerificationError):
                e.classify_live('PSIT is in Kanpur.',[SOURCE])

    def test_missing_correction_is_error(self):
        with patch.object(e,'_llm',return_value={'label':'CONTRADICTED','reason':'A','source_ids':[1]}):
            with self.assertRaises(e.VerificationError):
                e.classify_live('PSIT is in Lucknow.',[SOURCE])

    def test_genuine_unknown(self):
        with patch.object(e,'_llm',return_value={'label':'INSUFFICIENT EVIDENCE','reason':'No record of the private event.','source_ids':[]}):
            self.assertEqual(e.classify_live('My private model has 99% accuracy.',[SOURCE])['label'],'INSUFFICIENT EVIDENCE')

    def test_no_results(self):
        self.assertEqual(e.classify_live('A fact',[])['label'],'INSUFFICIENT EVIDENCE')

    def test_serper_answer_box_and_attributes(self):
        data={'answerBox':{'title':'Address','link':SOURCE['url'],'answer':'Kanpur'},
              'knowledgeGraph':{'title':'PSIT','website':SOURCE['url'],'attributes':{'Address':'Kanpur'}},
              'organic':[{'title':'bad','link':'javascript:alert(1)','snippet':'bad'},
                         {'title':'good','link':SOURCE['url'],'snippet':'Kanpur'}]}
        with patch.dict(os.environ,{'SERPER_API_KEY':'test'}),patch.object(e,'_post',return_value=data):
            result=e.search('PSIT city')
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]['snippet'],'Kanpur')
        self.assertIn('Address: Kanpur',result[-1]['snippet'])

    def test_errors_not_verdicts(self):
        with patch.object(e,'search',side_effect=e.VerificationError('Serper HTTP 401')):
            result=e._verify_one({'claim':'PSIT is in Lucknow.','queries':['PSIT city']})
        self.assertEqual(result['label'],'ERROR')

    def test_second_search_recovers(self):
        with patch.object(e,'search',side_effect=[e.VerificationError('timeout'),[SOURCE]]),patch.object(e,'classify_live',return_value={'label':'SUPPORTED'}):
            result=e._verify_one({'claim':'PSIT is in Kanpur.','queries':['PSIT address','PSIT city']})
        self.assertEqual(result['label'],'SUPPORTED')
        self.assertTrue(result['warnings'])

    def test_targeted_retry(self):
        replies=[{'label':'INSUFFICIENT EVIDENCE','followup_query':'PSIT official address'}, {'label':'CONTRADICTED'}]
        with patch.object(e,'search',return_value=[SOURCE]) as search,patch.object(e,'classify_live',side_effect=replies):
            result=e._verify_one({'claim':'PSIT is in Lucknow.','queries':['PSIT city']})
        self.assertEqual(result['label'],'CONTRADICTED')
        self.assertEqual(search.call_count,2)
        self.assertIn('PSIT official address',result['search_queries'])

    def test_incomplete_search_not_unknown(self):
        with patch.object(e,'search',side_effect=[[SOURCE],e.VerificationError('Rate limit')]),patch.object(e,'classify_live',return_value={'label':'INSUFFICIENT EVIDENCE','followup_query':'another query'}):
            result=e._verify_one({'claim':'Some fact','queries':['first query']})
        self.assertEqual(result['label'],'ERROR')

    def test_nonsense_not_invented(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'test','SERPER_API_KEY':'test'}),patch.object(e,'_llm',return_value={'claims':[]}):
            result=e.run_live('hello, how are you?')
        self.assertEqual(result['total'],0)
        self.assertIn('No checkable',result['message'])

    def test_overflow_not_silently_dropped(self):
        with patch.object(e,'_llm',return_value={'claims':[],'too_many':True}):
            with self.assertRaises(ValueError): e.extract_claims('Many claims')

    def test_input_validation(self):
        client=TestClient(app)
        self.assertEqual(client.post('/api/verify',json={'text':'x'*8001}).status_code,422)
        self.assertEqual(client.post('/api/verify',json={'text':'Claim','mode':'bad'}).status_code,422)

    def test_missing_key_clear_error(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':''}):
            r=TestClient(app).post('/api/verify',json={'text':'PSIT is in Kanpur.'})
        self.assertEqual(r.status_code,503)
        self.assertIn('GROQ_API_KEY',r.json()['detail'])

    def test_count_errors_separately(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'test','SERPER_API_KEY':'test'}),patch.object(e,'extract_claims',return_value=[{'claim':'A','queries':['A']}]),patch.object(e,'_verify_one',return_value={'label':'ERROR'}):
            result=e.run_live('A claim')
        self.assertEqual(result['counts']['ERROR'],1)
        self.assertEqual(result['counts']['INSUFFICIENT EVIDENCE'],0)

if __name__=='__main__': unittest.main()
