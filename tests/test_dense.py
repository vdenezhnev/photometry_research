"""Tests for dense fused pipeline (mocked pycolmap)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adaptive_sampling.sparse_eval.dense import dense_api_available, run_dense_fusion


def test_dense_api_available_without_pycolmap(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pycolmap":
            raise ImportError("no pycolmap")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert dense_api_available() is False


def test_run_dense_fusion_calls_pycolmap_stages(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "000.jpg").write_bytes(b"jpeg")
    sparse_model = tmp_path / "sparse" / "0"
    sparse_model.mkdir(parents=True)
    workspace = tmp_path / "workspace"

    mock_pm = MagicMock()
    mock_pm.filter = False

    mock_pycolmap = MagicMock()
    mock_pycolmap.PatchMatchOptions.return_value = mock_pm

    def _fusion(out_path, dense_dir, *, output_type):
        Path(out_path).write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n0 0 0\n1 1 1\n"
        )

    mock_pycolmap.undistort_images = MagicMock()
    mock_pycolmap.patch_match_stereo = MagicMock()
    mock_pycolmap.stereo_fusion = MagicMock(side_effect=_fusion)

    with patch.dict("sys.modules", {"pycolmap": mock_pycolmap}):
        fused, durations, count = run_dense_fusion(
            workspace=workspace,
            image_dir=image_dir,
            sparse_model_dir=sparse_model,
            dense_cfg={"fused_ply_filename": "fused.ply"},
            log=lambda _msg: None,
        )

    assert fused is not None
    assert fused.name == "fused.ply"
    assert durations["undistort"] >= 0
    assert durations["patch_match_stereo"] >= 0
    assert durations["stereo_fusion"] >= 0
    assert count == 2
    mock_pycolmap.undistort_images.assert_called_once()
    mock_pycolmap.patch_match_stereo.assert_called_once()
    mock_pycolmap.stereo_fusion.assert_called_once()
    assert mock_pm.filter is True


def test_run_dense_fusion_raises_without_cuda_api() -> None:
    mock_pycolmap = MagicMock(spec=[])

    with patch.dict("sys.modules", {"pycolmap": mock_pycolmap}):
        with pytest.raises(RuntimeError, match="CUDA"):
            run_dense_fusion(
                workspace=Path("w"),
                image_dir=Path("i"),
                sparse_model_dir=Path("s"),
                dense_cfg={},
                log=lambda _msg: None,
            )
