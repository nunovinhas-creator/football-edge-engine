"""
Testes unitários do rate limiter local (src/historical_dataset/rate_limiter.py).

Usa relógio e sleep falsos (injetados) para testar o comportamento de
forma determinística e instantânea, sem depender de tempo real.
"""

import unittest

from src.historical_dataset.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestRateLimiter(unittest.TestCase):

    def test_invalid_constructor_args_raise(self):
        with self.assertRaises(ValueError):
            RateLimiter(max_calls=0)
        with self.assertRaises(ValueError):
            RateLimiter(max_calls=5, period_seconds=0)

    def test_calls_within_limit_do_not_wait(self):
        clock = FakeClock()
        limiter = RateLimiter(max_calls=3, period_seconds=1.0, sleep_func=clock.sleep, time_func=clock.time)

        waited = [limiter.acquire() for _ in range(3)]

        self.assertEqual(waited, [0.0, 0.0, 0.0])
        self.assertEqual(clock.now, 0.0)

    def test_exceeding_limit_waits_for_window_to_free_up(self):
        clock = FakeClock()
        limiter = RateLimiter(max_calls=2, period_seconds=1.0, sleep_func=clock.sleep, time_func=clock.time)

        limiter.acquire()  # t=0
        limiter.acquire()  # t=0
        waited = limiter.acquire()  # deve esperar até t=1 (janela de 1s)

        self.assertAlmostEqual(waited, 1.0)
        self.assertAlmostEqual(clock.now, 1.0)

    def test_calls_spaced_out_naturally_never_wait(self):
        clock = FakeClock()
        limiter = RateLimiter(max_calls=2, period_seconds=1.0, sleep_func=clock.sleep, time_func=clock.time)

        limiter.acquire()
        clock.now += 1.0
        waited = limiter.acquire()

        self.assertEqual(waited, 0.0)

    def test_reset_clears_recorded_calls(self):
        clock = FakeClock()
        limiter = RateLimiter(max_calls=1, period_seconds=10.0, sleep_func=clock.sleep, time_func=clock.time)

        limiter.acquire()
        limiter.reset()
        waited = limiter.acquire()

        self.assertEqual(waited, 0.0)


if __name__ == "__main__":
    unittest.main()
