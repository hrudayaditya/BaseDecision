import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from basedecision import (Request, Option, Result, InputError, decide, load,
    CalibratedDecision, CalibrationError, calibration_profiles)
from basedecision.calibration import _DATA, _sha

class Fake:
    precision='bf16';device=SimpleNamespace(type='cuda')
    def __init__(self,path):self.model_id=str(path);self.calls=0
    def count_tokens(self,r):return len(r.context)+20
    def predict_batch(self,requests):
        self.calls+=1
        return [Result(r.options[0].id,r.options[0].id,r.options[0].text,
            {o.id:p for o,p in zip(r.options,[.9820137900379085,.01798620996209156])},
            {o.id:z for o,z in zip(r.options,[4.,0.])},self.count_tokens(r),self.model_id,r.kind) for r in requests]

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.raw=Fake(self.root)
        a=json.loads(_DATA.read_text());a['checkpoint']={}
        for name in ['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/tokenizer.json']:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}');a['checkpoint'][name]=_sha(p)
        a['scopes']['sgd_identifier'].update(min_tokens=1,max_tokens=1000,min_options=2,max_options=2)
        self.artifact=self.root/'cal.json';self.artifact.write_text(json.dumps(a))
        self.req=Request('state','Which?',(Option('a','A'),Option('b','B')))
    def tearDown(self):self.tmp.cleanup()
    def wrap(self):return CalibratedDecision(self.raw,profile='sgd_identifier',artifact=self.artifact)
    def test_apply_preserves_decision_and_raw(self):
        raw=self.raw.predict_batch([self.req])[0];cal=self.wrap();r=cal.apply(raw)
        self.assertEqual(r.answer,raw.answer);self.assertEqual(r.raw_logits,raw.raw_logits)
        self.assertEqual(r.raw_probabilities,raw.probabilities);self.assertLess(r.probabilities['a'],raw.probabilities['a'])
        self.assertAlmostEqual(sum(r.probabilities.values()),1);self.assertEqual(r.to_dict()['calibration']['profile'],'sgd_identifier')
        self.assertEqual(raw.calibration_status,'raw_softmax_uncalibrated')
    def test_reject_repeated_and_foreign_result(self):
        cal=self.wrap();r=cal.predict(self.req)
        with self.assertRaises(CalibrationError):cal.apply(r)
        with self.assertRaises(CalibrationError):cal.apply(replace(r,model_id='other'))
    def test_declined_and_unknown_profiles(self):
        for name in ('clinc','refund'):
            with self.assertRaises(CalibrationError):CalibratedDecision(self.raw,profile=name,artifact=self.artifact)
        self.assertFalse(calibration_profiles()['clinc']['available'])
    def test_binding_and_precision(self):
        (self.root/'model.safetensors').write_text('modified')
        with self.assertRaises(CalibrationError):self.wrap()
        (self.root/'model.safetensors').write_text('{}');self.raw.precision='fp32'
        with self.assertRaises(CalibrationError):self.wrap()
    def test_batch_prevalidates_every_item(self):
        cal=self.wrap()
        with self.assertRaises(CalibrationError):cal.predict_batch([self.req,replace(self.req,context='x'*1001)])
        self.assertEqual(self.raw.calls,0)
        result=list(cal.predict_iter([self.req]*3,buffer_size=2));self.assertEqual(len(result),3)
    def test_wrong_kind_rejected(self):
        cal=self.wrap()
        with self.assertRaises(CalibrationError):cal.check(context='s',question='q')
        with self.assertRaises(CalibrationError):cal.score(context='s',question='q',levels=['a','b'])
    def test_named_schema_validated_before_inference(self):
        with self.assertRaises(InputError):decide(self.raw,context='s',questions={'good':{'question':'q','options':['a','b']},'bad':{'question':'q','typo':1}})
        self.assertEqual(self.raw.calls,0)
        r=decide(self.raw,context='s',questions={'intent':{'question':'q','options':['a','b']},'active':{'kind':'noul','question':'active?'}})
        self.assertEqual(list(r),['intent','active'])
    def test_load_is_explicit(self):
        with patch('basedecision.api.BaseDecision.from_pretrained',return_value=self.raw) as factory:
            self.assertIs(load('/local',device='cpu',precision='fp32'),self.raw)
            factory.assert_called_once_with('/local',device='cpu',precision='fp32')
    def test_unavailable_profile_rejected_before_load(self):
        with patch('basedecision.api.BaseDecision.from_pretrained') as factory:
            with self.assertRaises(CalibrationError):load('/local',calibration_profile='clinc')
            factory.assert_not_called()
    def test_packaged_inference_contract_unchanged(self):
        a=json.loads(_DATA.read_text());root=Path(__import__('basedecision').__file__).parent
        self.assertTrue(all(_sha(root/k)==v for k,v in a['sdk_contract'].items()))
    def test_import_does_not_load_torch(self):
        import subprocess,sys
        subprocess.run([sys.executable,'-c','import basedecision,sys; assert "torch" not in sys.modules'],check=True)

if __name__=='__main__':unittest.main()
