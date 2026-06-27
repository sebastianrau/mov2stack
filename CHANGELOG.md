# Changelog

## 0.1.0 - 2026-06-27

Initial release.

- Stack video frames with `mean`, `average`, `max`, `min`, and `median`.
- Add lightning-optimized two-pass stacking mode.
- Keep output at the source video resolution.
- Add optional motion compensation: `translation`, `phase`, `affine`, and `ecc`.
- Add frame progress counter and percent bar.
- Add parallel batch preprocessing.
- Add `--start` and `--stop` video trimming in seconds.
- Use `<input name>_stacked.png` as the default output path.
