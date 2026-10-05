import unittest
from unittest.mock import patch
from basedecision import load,InputError,CalibrationError,CPUResult
class CPUAPI(unittest.TestCase):
    def test_explicit_dispatch(self):
        with patch('basedecision.cpu.CPUFastDecision.from_pretrained',return_value='sentinel') as f:
            self.assertEqual(load('model',backend='cpu_fast'),'sentinel')
            f.assert_called_once_with('model',device='cpu',precision='bf16')
    def test_calibration_blocked_before_load(self):
        with patch('basedecision.cpu.CPUFastDecision.from_pretrained') as f:
            with self.assertRaises(CalibrationError):load('model',backend='cpu_fast',calibration_profile='sgd_identifier')
            f.assert_not_called()
    def test_unknown_backend(self):
        with self.assertRaises(InputError):load('model',backend='magic')
    def test_metadata(self):
        r=CPUResult('a','a','A',{'a':.5,'b':.5},{'a':0,'b':0},10,'model','choice')
        self.assertEqual(r.to_dict()['backend'],'cpu_fast_experimental')
        self.assertEqual(r.calibration_status,'raw_softmax_uncalibrated')
