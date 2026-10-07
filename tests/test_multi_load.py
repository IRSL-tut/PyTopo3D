import numpy as np
import pytest

from pytopo3d.core.optimizer import top3d, top3d_multi


@pytest.mark.parametrize("as_sequence", [False, True])
def test_duplicate_load_cases_double_compliance_and_preserve_design(as_sequence):
    nelx, nely, nelz = 4, 3, 2
    force_field = np.zeros((nely, nelx, nelz, 3))
    force_field[0, nelx - 1, :, 2] = -1.0
    force_fields = (
        [force_field, force_field]
        if as_sequence
        else np.stack((force_field, force_field))
    )
    single_design, single_history = top3d(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.35,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        force_field=force_field,
        tolx=0.0,
        maxloop=1,
        save_history=True,
        history_frequency=1,
        use_gpu=False,
    )
    multi_design, multi_history = top3d_multi(
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=0.35,
        penal=3.0,
        rmin=1.5,
        disp_thres=0.5,
        force_fields=force_fields,
        tolx=0.0,
        maxloop=1,
        save_history=True,
        history_frequency=1,
        use_gpu=False,
    )

    np.testing.assert_allclose(multi_design, single_design, rtol=1e-10, atol=1e-12)
    assert multi_history["compliance_history"][-1] == pytest.approx(
        2.0 * single_history["compliance_history"][-1], rel=1e-12
    )


@pytest.mark.parametrize(
    "force_fields",
    [
        np.zeros((0, 3, 4, 2, 3)),
        np.zeros((2, 3, 4, 2, 2)),
        np.full((2, 3, 4, 2, 3), np.nan),
    ],
)
def test_multi_load_rejects_invalid_force_fields(force_fields):
    with pytest.raises(ValueError, match="force_fields"):
        top3d_multi(
            nelx=4,
            nely=3,
            nelz=2,
            volfrac=0.3,
            penal=3.0,
            rmin=1.5,
            disp_thres=0.5,
            force_fields=force_fields,
            maxloop=0,
        )
