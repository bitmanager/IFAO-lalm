import unittest
from lalm.annotation.inspection import agreement_summary

class AgreementTests(unittest.TestCase):
    def test_two_is_not_three(self):
        out = agreement_summary({'gemini_verbatim': 'Ёж, да!', 'gemini_coverage': 'еж да'})
        self.assertEqual(out['agreement_status'], '2 совпали')
        self.assertFalse(out['three_pass_agreement'])

    def test_any_disagreement_excludes_unanimity(self):
        out = agreement_summary({'gemini_verbatim': 'да', 'gemini_coverage': 'да',
                                 'gemini_literal_audit': 'нет'})
        self.assertEqual(out['agreement_status'], 'расхождение')

    def test_reference_is_not_blind(self):
        out = agreement_summary({'gemini_verbatim': 'да', 'gemini_coverage': 'Да!',
                                 'gemini_source_correction': 'да'})
        self.assertTrue(out['three_pass_agreement'])
        self.assertTrue(out['source_conditioned_agreement'])

    def test_empty_is_not_agreement(self):
        out = agreement_summary({'gemini_verbatim': '', 'gemini_coverage': None})
        self.assertEqual(out['agreement_status'], 'неполный ответ')
