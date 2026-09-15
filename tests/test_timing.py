from lab.core.config import Timeframe
from lab.quant.timing import bar_times_for_session


def test_daily_session_times_follow_dst():
    winter = bar_times_for_session("XNYS", "2024-01-12", Timeframe.D1)[0]
    summer = bar_times_for_session("XNYS", "2024-07-12", Timeframe.D1)[0]
    assert winter[0].hour == 14
    assert summer[0].hour == 13


def test_early_close_and_short_final_hour_are_explicit():
    daily = bar_times_for_session("XNYS", "2024-11-29", Timeframe.D1)
    hourly = bar_times_for_session("XNYS", "2024-11-29", Timeframe.H1)
    assert (daily[0][1] - daily[0][0]).total_seconds() == 3.5 * 60 * 60
    assert hourly[-1][1] == daily[0][1]
    assert (hourly[-1][1] - hourly[-1][0]).total_seconds() == 30 * 60
