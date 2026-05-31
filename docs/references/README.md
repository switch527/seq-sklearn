# References

Academic papers that ground the algorithms seq-sklearn implements.

## v1 (TFT)

- `Temporal Fusion Transformers.pdf` — Lim, Arik, Loeff, Pfister (2021).
  Temporal Fusion Transformers for Interpretable Multi-horizon Time
  Series Forecasting. arXiv:1912.09363. The encoder-only adaptation
  for classification and standard regression is documented in
  `docs/architecture.md` "v1 concrete: TFT".

## Planned additions

Per the v2 / v3 roadmap in `docs/requirements.md`:

- **v2 transformer family**: PatchTST (Nie et al., arXiv:2211.14730),
  TimesNet (Wu et al., arXiv:2210.02186), TST (Zerveas et al.,
  arXiv:2010.02803).
- **v3 recurrent family**: LSTM-FCN (Karim et al.,
  arXiv:1709.05206; ALSTM-FCN arXiv:1801.04503).

## File naming

Originals can keep their published filenames. If a paper is renamed
later for sorting consistency, do so via `git mv`.
