import copy
import math
import unittest
from compare_kld_strict import compare


class Contracts(unittest.TestCase):
    def setUp(self):
        self.panel = [dict(id=0, kind='generic', text='A sufficiently long generic prompt.')]
        self.data = dict(model='fixture', k=20, items=[dict(id=0, kind='generic',
            prompt_lp=[None, {str(i): math.log(0.04) for i in range(20)}])])

    def test_identical_distributions(self):
        d = compare(self.panel, self.data, copy.deepcopy(self.data))
        self.assertAlmostEqual(d['mean_kl'], 0)
        self.assertEqual(d['teacher_forced_positions'], 1)
        self.assertFalse(d['numerical_quality_gate_applied'])

    def test_no_greedy_fallback_or_truncated_grid(self):
        for replacement in ([], [dict(id=0, kind='generic', gen_tokens=['x'])]):
            d = copy.deepcopy(self.data); d['items'] = replacement
            with self.assertRaises(ValueError): compare(self.panel, self.data, d)

    def test_no_zip_truncation(self):
        d = copy.deepcopy(self.data); d['items'][0]['prompt_lp'].append(d['items'][0]['prompt_lp'][1])
        with self.assertRaisesRegex(ValueError, 'length'): compare(self.panel, self.data, d)

    def test_invalid_support_and_identity(self):
        for change in ('nan', 'mass', 'missing', 'id', 'model'):
            d = copy.deepcopy(self.data)
            if change == 'nan': d['items'][0]['prompt_lp'][1]['0'] = float('nan')
            if change == 'mass': d['items'][0]['prompt_lp'][1] = {str(i): 0.0 for i in range(20)}
            if change == 'missing': del d['items'][0]['prompt_lp'][1]['0']
            if change == 'id': d['items'][0]['id'] = 1
            if change == 'model': d['model'] = 'other'
            with self.assertRaises(ValueError): compare(self.panel, self.data, d)


if __name__ == '__main__': unittest.main()
