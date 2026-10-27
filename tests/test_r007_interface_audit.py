from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from aorta_drive.vision.r007_interface_audit import (
    FactorialCell,
    SequentialCut,
    audit_sequential_cuts,
    audit_factorial_cells,
    factorial_effects,
    require_exact_identity,
    validate_derangement,
    validate_factorial_cells,
)


def _cells(*, interaction: float = 0.0) -> dict[str, FactorialCell]:
    def output(native: float, dino: float) -> npt.NDArray[np.float64]:
        return np.asarray([2.0 * native + 3.0 * dino + interaction * native * dino])

    return {
        "shift_shift": FactorialCell(output(0.0, 0.0), "native-shift", "dino-shift"),
        "shift_clean": FactorialCell(output(0.0, 1.0), "native-shift", "dino-clean"),
        "clean_shift": FactorialCell(output(1.0, 0.0), "native-clean", "dino-shift"),
        "clean_clean": FactorialCell(output(1.0, 1.0), "native-clean", "dino-clean"),
    }


def test_additive_factorial_has_zero_interaction_and_exact_closure() -> None:
    effects = factorial_effects(_cells())
    np.testing.assert_array_equal(effects.dino_repair_at_shifted_native, [3.0])
    np.testing.assert_array_equal(effects.native_repair_at_shifted_dino, [2.0])
    np.testing.assert_array_equal(effects.joint_repair, [5.0])
    np.testing.assert_array_equal(effects.interaction, [0.0])
    np.testing.assert_array_equal(effects.closure_residual, [0.0])
    assert audit_factorial_cells(_cells()).closure_exact


def test_nonadditive_factorial_preserves_interaction_instead_of_hiding_it() -> None:
    effects = factorial_effects(_cells(interaction=7.0))
    np.testing.assert_array_equal(effects.interaction, [7.0])
    np.testing.assert_array_equal(effects.joint_repair, [12.0])
    np.testing.assert_array_equal(effects.closure_residual, [0.0])


def test_path_isolation_detects_native_path_leakage() -> None:
    cells = _cells()
    leaked = cells["shift_clean"]
    cells["shift_clean"] = FactorialCell(
        leaked.output, "native-shift-rebuilt", leaked.dino_fingerprint
    )
    with pytest.raises(ValueError, match="path isolation"):
        validate_factorial_cells(cells)


def test_path_isolation_detects_dino_path_leakage() -> None:
    cells = _cells()
    leaked = cells["clean_shift"]
    cells["clean_shift"] = FactorialCell(
        leaked.output, leaked.native_fingerprint, "dino-shift-rebuilt"
    )
    with pytest.raises(ValueError, match="path isolation"):
        validate_factorial_cells(cells)


def test_factorial_rejects_aliases_nonfinite_and_shape_mismatch() -> None:
    aliased = _cells()
    aliased["shift_clean"] = FactorialCell(
        aliased["shift_clean"].output, "native-shift", "dino-shift"
    )
    aliased["clean_clean"] = FactorialCell(
        aliased["clean_clean"].output, "native-clean", "dino-shift"
    )
    with pytest.raises(ValueError, match="DINO paths unexpectedly alias"):
        validate_factorial_cells(aliased)

    nonfinite = _cells()
    nonfinite["clean_clean"] = FactorialCell(
        np.asarray([np.nan]), "native-clean", "dino-clean"
    )
    with pytest.raises(ValueError, match="finite"):
        validate_factorial_cells(nonfinite)

    wrong_shape = _cells()
    wrong_shape["clean_clean"] = FactorialCell(
        np.asarray([1.0, 2.0]), "native-clean", "dino-clean"
    )
    with pytest.raises(ValueError, match="shapes disagree"):
        validate_factorial_cells(wrong_shape)


def test_derangement_and_identity_contracts_are_fail_closed() -> None:
    validate_derangement({"g0": "g1", "g1": "g2", "g2": "g0"})
    with pytest.raises(ValueError, match="fixed points"):
        validate_derangement({"g0": "g0", "g1": "g1"})
    with pytest.raises(ValueError, match="permutation"):
        validate_derangement({"g0": "g1", "g1": "g2"})

    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    require_exact_identity(reference, reference.copy())
    with pytest.raises(ValueError, match="not exact"):
        require_exact_identity(reference, reference + np.float32(1e-7))


def _sequential_cuts() -> dict[str, SequentialCut]:
    shifted = np.asarray([2.0, 3.0], dtype=np.float32)
    return {
        "raw_dino_tokens": SequentialCut(
            "raw_dino_tokens",
            "raw-clean",
            "raw-shift",
            shifted,
            shifted.copy(),
            np.asarray([1.0, 1.5], dtype=np.float32),
        ),
        "projected_dino_tokens": SequentialCut(
            "projected_dino_tokens",
            "projected-clean",
            "projected-shift",
            shifted,
            shifted.copy(),
            np.asarray([1.2, 1.7], dtype=np.float32),
        ),
        "compressed_dino_tokens": SequentialCut(
            "compressed_dino_tokens",
            "compressed-clean",
            "compressed-shift",
            shifted,
            shifted.copy(),
            np.asarray([1.4, 1.8], dtype=np.float32),
        ),
    }


def test_sequential_dino_only_chain_qualifies_complete_raw_cut() -> None:
    order = ("raw_dino_tokens", "projected_dino_tokens", "compressed_dino_tokens")
    audit = audit_sequential_cuts(
        _sequential_cuts(),
        expected_order=order,
        clean_output=np.asarray([1.0, 1.5], dtype=np.float32),
    )
    assert audit.stages == order
    assert audit.identity_exact and audit.full_cut_exact and audit.finite


def test_sequential_chain_rejects_wrong_order_alias_and_false_full_cut() -> None:
    cuts = _sequential_cuts()
    wrong_order = (
        "projected_dino_tokens",
        "raw_dino_tokens",
        "compressed_dino_tokens",
    )
    with pytest.raises(ValueError, match="insertion order"):
        audit_sequential_cuts(
            cuts,
            expected_order=wrong_order,
            clean_output=np.asarray([1.0, 1.5], dtype=np.float32),
        )

    cuts = _sequential_cuts()
    item = cuts["projected_dino_tokens"]
    cuts["projected_dino_tokens"] = SequentialCut(
        item.stage,
        item.shifted_fingerprint,
        item.shifted_fingerprint,
        item.shifted_output,
        item.identity_output,
        item.clean_patch_output,
    )
    order = ("raw_dino_tokens", "projected_dino_tokens", "compressed_dino_tokens")
    with pytest.raises(ValueError, match="alias"):
        audit_sequential_cuts(
            cuts,
            expected_order=order,
            clean_output=np.asarray([1.0, 1.5], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="not exact"):
        audit_sequential_cuts(
            _sequential_cuts(),
            expected_order=order,
            clean_output=np.asarray([9.0, 9.0], dtype=np.float32),
        )

