import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_qrt_openai_server import HttpResponse, VerificationError, Verifier


class ToolRecoveryVerifierTest(unittest.TestCase):
    def pair(self, exhausted):
        progress = dict(history_signature_occurrences=2, history_no_progress_results=2,
                        history_no_progress_streak=2 if exhausted else 0,
                        exhausted_history_calls_suppressed=int(exhausted), no_progress=exhausted)
        call = dict(id='call_1', type='function', function=dict(name='get_weather', arguments='{"city":"Paris"}'))
        if exhausted:
            full = dict(error=dict(code='tool_call_no_progress'), qrt_tool_progress=progress)
            chunks = [copy.deepcopy(full)]
        else:
            full = dict(choices=[dict(finish_reason='tool_calls', message=dict(tool_calls=[call]))],
                        qrt_tool_progress=progress)
            chunks = [dict(choices=[dict(delta=dict(tool_calls=[dict(call, index=0)]), finish_reason=None)]),
                      dict(choices=[dict(delta={}, finish_reason='tool_calls')], qrt_tool_progress=progress)]
        return full, chunks

    def check(self, full, chunks, exhausted, *, status=None, done=True):
        status = (400 if exhausted else 200) if status is None else status
        body = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
        if done:
            body += 'data: [DONE]\n\n'
        verifier = Verifier('http://127.0.0.1:8000', 'test-model', None, 1, 262144)
        return verifier.check_tool_recovery_pair(
            HttpResponse(status, {}, json.dumps(full).encode()), HttpResponse(200, {}, body.encode()), exhausted)

    def test_exhaustion_requires_an_error_without_a_successful_finish(self):
        full, chunks = self.pair(True)
        self.assertTrue(self.check(full, chunks, True)['exhausted'])
        with self.assertRaises(VerificationError):
            self.check(full, chunks, True, status=200)
        with self.assertRaises(VerificationError):
            self.check(full, chunks, True, done=False)
        with self.assertRaises(VerificationError):
            self.check(full, chunks + [dict(choices=[dict(delta={}, finish_reason='stop')])], True)
        old = dict(choices=[dict(finish_reason='stop', message=dict(content='I will check.'))],
                   qrt_tool_progress=full['qrt_tool_progress'])
        with self.assertRaises(VerificationError):
            self.check(old, chunks, True)
        with self.assertRaises(VerificationError):
            self.check(full, [dict(qrt_tool_progress=full['qrt_tool_progress'])], True)

    def test_recovery_requires_original_call_and_retained_lifetime_counts(self):
        full, chunks = self.pair(False)
        self.assertFalse(self.check(full, chunks, False)['exhausted'])
        wrong = copy.deepcopy(full)
        wrong['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"city":"London"}'
        with self.assertRaises(VerificationError):
            self.check(wrong, chunks, False)
        for field, value in [('history_no_progress_results', 0), ('history_no_progress_streak', 2)]:
            with self.subTest(field=field):
                wrong_full, wrong_chunks = self.pair(False)
                wrong_full['qrt_tool_progress'][field] = value
                with self.assertRaises(VerificationError):
                    self.check(wrong_full, wrong_chunks, False)


if __name__ == '__main__':
    unittest.main()
