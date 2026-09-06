# Micro-copilot publish-readiness report

Date: 2026-09-06  
Test platform: macOS, Python 3.10.20, PyTorch 2.1.2, CPU

## Status

The primary workflow is ready for a clean GitHub release: start `main.py`, drag in an image, run the bundled cell-segmentation model, and run lysosome/puncta analysis. Both supplied model files are packaged defaults and were verified from an installed wheel.

Before making the repository public, revoke and replace the Azure OpenAI credential and Google service-account credential that were previously stored in this copy. The credential files and hard-coded values have been removed, but deletion does not revoke a credential or remove it from another copy or existing Git history.

## Cleanup performed

- Removed one-off plotting and temporary analysis programs from the repository root, including `ferroptosis_rsl3_concat_framewise.py`, `gary_threechannel_co.py`, and the related ferroptosis/RSL3/VATPase plotting scripts. The required co-localization calculation from the Gary script was retained in the application.
- Removed generated figure directories, the 47 MB `draw_figures` tree, notebooks, debug images, logs, settings, caches, build products, and package metadata.
- Removed unused experimental training/test programs and obsolete package modules that were not imported by the application.
- Removed stale upstream Cellpose documentation and PyInstaller material that described or packaged a different application.
- Removed the embedded Google service-account JSON and all Google Cloud upload code.
- Replaced the embedded Azure OpenAI key and endpoint with optional environment/user configuration.
- Added a concise project README, packaging metadata, Git ignores, Git LFS tracking, a root `main.py`, and version `0.1.0`.

## Bundled models

- `cellquant/weights/model_0528_resize` is the default cell-segmentation model, displayed as **Micro-copilot (bundled)**.
- `cellquant/weights/model.pth` is the default puncta false-positive/ring classifier and **use DL** is enabled on a clean first launch.
- `CELLQUANT_CELLPOSE_MODEL` and `CELLQUANT_PUNCTA_MODEL` can override the defaults.
- The resulting wheel is 68 MB and contains both weights. Both models load successfully after installing that wheel into an empty temporary directory.
- SHA-256: `model_0528_resize` = `3cdbd0a810f2311953a239b8b77b320af1e2500be55523d309cd3df1a1f5b0f1`; `model.pth` = `3dd4d1d6c6e1b077b93f739246f9fc4cd01e02e1228d336e1df3e278b8505af1`.

## Confirmed fixes

- Added the missing root launcher, so `python main.py` works.
- Fixed default/custom model selection, including an off-by-one model-menu lookup.
- Renamed user-facing product text and the bundled model entry to **Micro-copilot** while retaining the `cellquant` Python package for compatibility.
- The bundled segmentation model now loads its saved 170.89-pixel diameter on a clean launch.
- Analysis Run now restores and displays the cell-segmentation mask after lysosome detection instead of leaving a puncta-only label layer.
- Analysis Run now saves a cell-segmentation overlay PNG beside every input image, including when the GUI supplied the mask.
- Two-channel analysis now records both directional co-localization measurements in a per-image `*_colocalization.csv` summary and annotates each source-channel punctum table.
- Replaced a developer-only Windows classifier path with the packaged model path.
- Prevented unnecessary ImageNet weight downloads when loading the puncta classifier and made CPU loading work for GPU-saved checkpoints.
- Prevented empty detection results, blank channels, and a result-queue race from crashing the GUI.
- Prevented offline Acquire/Z-position buttons from raising errors when Micro-Manager is absent.
- Added bounded shutdown so background processes cannot keep `main.py` hanging indefinitely.
- Moved user settings outside the source tree and stopped saving LLM settings merely because analysis was run.
- Prevented batch outputs from overwriting one another by including the input-image stem in output names.
- Corrected the per-cell CSV pixel-count column and made saved segmentation images OpenCV-safe.
- Removed a missing `mask2former` training dependency by using the equivalent local PyTorch scheduler.
- Limited wavelet depth for small images and replaced a deprecated morphology call.

## Verification results

| Check | Result |
| --- | --- |
| Python compilation of `main.py`, package, and tests | Passed |
| `python main.py --version` | Passed; reports 0.1.0 |
| Drag a synthetic 512×512 three-channel TIFF into the GUI | Passed |
| Display/channel/brush/mask controls | Passed |
| **Run Segmentation** with bundled `model_0528_resize` | Passed |
| **Run** lysosome analysis with bundled `model.pth` | Passed; per-cell CSV created |
| Two-channel co-localization export | Passed; both directional comparisons written beside input image |
| Cell-segmentation overlay export during Analysis Run | Passed; PNG written beside input image |
| Acquire and Z controls without Micro-Manager | Passed; clear warning, no crash |
| Window close after analysis | Passed; bounded shutdown |
| Focused local tests | 8 passed, 1 skipped |
| New co-localization and overlay regression tests | 3 passed, 0 failed |
| Full test collection | 12 passed, 11 skipped, 0 failed |
| Clean wheel build and temporary install | Passed; 68 MB |
| Installed-wheel loading of both bundled weights | Passed |
| Wheel content audit | Passed; no keys, settings, caches, or removed diagnostics |
| Source credential/local-path scan | Passed; no embedded credential or developer filesystem path found |

Ten of the full-suite skips require the external Cellpose reference-data archive. The old URLs return 404 and the current official OSF archive returned HTTP 403 during this run. One CUDA-only test was skipped because no CUDA device was available. A manually downloaded reference-data directory can be supplied with `CELLQUANT_TEST_DATA_DIR`.

## Remaining risks and limitations

1. **Medium — GUI responsiveness on large images.** Segmentation and analysis wait synchronously in the Qt UI path. A large image can make the window appear frozen until processing completes. Correcting this properly requires a larger asynchronous UI refactor and was kept out of this conservative cleanup.
2. **Medium — optional hardware workflow not validated.** The disconnected state is safe, but real Micro-Manager acquisition, autofocus, abort, and repeated acquisition were not tested without microscope hardware.
3. **Medium — optional language mode executes generated analysis code.** Use it only with a trusted endpoint and trusted instructions. The built-in lysosome workflow does not require language mode.
4. **Low — plot-process shutdown.** The plot worker did not always exit within five seconds in the headless test and was terminated by the new shutdown guard. No result data was lost in the tested workflow.
5. **Release operations.** Git LFS is not installed on the test machine, but `.gitattributes` is configured for both weights. Install Git LFS before committing them. Confirm that the project has permission to redistribute both checkpoints and rotate the deleted credentials before publication.

## Recommended release command sequence

```bash
git lfs install
python -m pip install ".[gui]"
python main.py
python -m pytest -q
```
