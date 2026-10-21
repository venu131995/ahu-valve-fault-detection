import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_all as R  # noqa: E402


def ahu(days=3, oa_frac=0.3, vlv_cmd=0.0, vlv_pos=None, coil_dT=-1.0, seed=0):
    """Minute data for an occupied AHU: outdoor 30 F, return 72 F, mixed air from the given outdoor-air fraction."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-08-06 06:00", periods=days * 24 * 60, freq="1min")
    n = len(t)
    oa, ra = np.full(n, 30.0), np.full(n, 72.0)
    ma = oa_frac * oa + (1 - oa_frac) * ra
    return pd.DataFrame(dict(SYS_CTL=1.0, SF_SPD_DM=1.0, OA_TEMP=oa, RA_TEMP=ra, MA_TEMP=ma + rng.normal(0, 0.01, n),
                             OA_DMPR_DM=0.4, SA_TEMP=ma - coil_dT, SA_TEMPSPT=ma - coil_dT, CHWC_VLV_DM=vlv_cmd,
                             CHWC_VLV=vlv_cmd if vlv_pos is None else vlv_pos), index=t)


def model(oa_frac=0.3):
    bins = np.linspace(0, 1, 21)
    return dict(bins=bins, frac_med=pd.Series(oa_frac, index=range(0, 23)), dT_closed_q99=1.1)


def test_mixing_balance_recovers_the_outdoor_air_fraction():
    x = R.features(ahu(days=1, oa_frac=0.35))
    assert np.allclose(x.oa_frac, 0.35, atol=1e-3)


def test_healthy_unit_raises_no_flags():
    f = R.daily_flags(ahu(), model())
    assert not f.any().any()


def test_damper_not_following_its_command_is_flagged():
    f = R.daily_flags(ahu(oa_frac=0.8), model(oa_frac=0.3))      # far more outdoor air than the command implies
    assert f.damper.all() and not f.leak.any()


def test_valve_that_does_not_close_shows_only_with_position_feedback():
    f = R.daily_flags(ahu(vlv_cmd=0.0, vlv_pos=0.10), model())     # how the data set models a 10 % leak
    assert f.position.all() and not f.leak.any()


def test_cooling_while_commanded_closed_is_flagged():
    f = R.daily_flags(ahu(vlv_cmd=0.0, coil_dT=4.0), model())     # 4 F of cooling with the valve told to close
    assert f.leak.all()
