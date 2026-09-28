"""Acceptance controls preserve production dispatch and an exhaustive oracle."""
import pytest
import torch

from rankseg_benchmark.nnunet import screening_benchmark as benchmark

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dim", [1, 3, 31, 513, 4097])
@pytest.mark.parametrize("chunk", [1, 7, 128, 8192])
def test_bounded_oracle_matches_original(device, dtype, dim, chunk):
    p = torch.rand(2, 3, dim, dtype=dtype, generator=torch.Generator().manual_seed(771)).to(device)
    p[0, 0] = 0
    p[0, 1] = 1
    p[1, 0] = .5
    before = p.clone()
    actual = benchmark.bounded_binary_oracle(p, chunk)
    expected = benchmark.binary_oracle(p)
    torch.testing.assert_close(actual, expected, rtol=16 * torch.finfo(torch.float64).eps, atol=0)
    assert torch.equal(p, before)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_bounded_oracle_rejects_invalid_chunk(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.bounded_binary_oracle(torch.ones(1, 1, 3), chunk)


@pytest.mark.parametrize("mode", ["multiclass", "multilabel"])
def test_production_screening_does_not_change_dispatch(monkeypatch, mode):
    from rankseg import _rankseg_algo as algo
    values = {name: getattr(algo, name, None) for name in benchmark.CUTOFFS}
    dispatch = getattr(algo, "_rma_dice_use_screening", None)
    decoder = benchmark.ScreeningDecoder("screened", mode, production_dispatch=True)
    def predict(probs):
        assert {name: getattr(algo, name, None) for name in benchmark.CUTOFFS} == values
        assert getattr(algo, "_rma_dice_use_screening", None) is dispatch
        return probs
    monkeypatch.setattr(decoder.decoder, "predict", predict)
    monkeypatch.setattr(benchmark, "decoder_control", lambda *args: pytest.fail("forced dispatch"))
    p = torch.ones(1, 3, 7)
    assert decoder.predict(p) is p


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["multiclass", "multilabel"])
@pytest.mark.parametrize("screening_mode", [True, "auto"])
def test_production_decoder_matches_public_wrapper(device, mode, screening_mode):
    from rankseg import RankSEG
    p = torch.rand(2, 3, 37, generator=torch.Generator().manual_seed(173)).to(device)
    p[1] = 0
    decoder = benchmark.ScreeningDecoder("screened", mode, production_dispatch=True, screening_mode=screening_mode)
    expected = RankSEG(metric="dice", solver="RMA", smooth=0, pruning_prob=.5,
                       output_mode=mode, safe_screening=screening_mode).predict(p)
    assert torch.equal(decoder.predict(p), expected)
