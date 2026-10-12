import unittest
from lalm.annotation.source_identity import production_conversation


class SourceIdentityTests(unittest.TestCase):
    def test_channels_and_dates_preserve_same_explicit_session(self):
        session = '089e27e9-89b4-4200-9129-db0813970e2a'
        prefix = 'staging/bitmanager-sessions/hf-revision/sessions/'
        for day, channel in [('2026/04/07', 'agent'), ('2026/04/08', 'customer')]:
            self.assertEqual(production_conversation(f'{prefix}{day}/{session}/{channel}.opus'), session)
        for key in [f'{prefix}2026/04/07/{session}/other.opus',
                    f'staging/another/sessions/2026/04/07/{session}/agent.opus',
                    f'{prefix}2026/04/07/{"-"*36}/agent.opus',
                    f'staging/bitmanager-sessions/{session}.opus']:
            self.assertIsNone(production_conversation(key))


if __name__ == '__main__':
    unittest.main()
