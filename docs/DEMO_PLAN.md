# Demo Plan

No public-ready real robot demo media was found in the Git-tracked public path. Raw videos under local dataset/output folders should stay out of Git until privacy, size, and lab-background review.

## Current Media Audit

| File | Content | Resolution | File size | Suitable for README |
| --- | --- | --- | ---: | --- |
| `diffusion_policy-main/diffusion_policy/env/kitchen/.../*.png` | Upstream kitchen/franka textures | Not inspected | 0.3-4.7 MB | No. Upstream texture assets, not ARX5 demo. |

Dataset videos may exist under ignored local folders such as `diffusion_policy-main/data_local/`, but they are not public release assets.

## Demo A: Base Diffusion Policy / ARX5 Grasp

1. Video content: ARX5 performs a complete glue-stick grasp using the strongest DP-EEF checkpoint.
2. Suggested length: 10-20 seconds raw, 5-8 seconds README clip.
3. Recommended camera: third-person side/front view showing robot, object, and table.
4. Screen recording: optional.
5. Show camera input: optional picture-in-picture if readable.
6. Show robot state: no for README, yes for technical video.
7. Show predicted action: no for README, optional overlay for technical video.
8. Show expert activation: no.
9. README crop: approach -> grasp -> lift.
10. Format: GIF for README, MP4 for release.
11. Suggested filename: `assets/demos/arx5_dp_grasp_main.gif`.

## Demo B: Action Chunk / DP-CFG Comparison

1. Video content: side-by-side DP baseline and DP-CFG execution on the same object pose, only if the comparison is representative.
2. Suggested length: 10-15 seconds per method.
3. Recommended camera: fixed third-person camera.
4. Screen recording: optional.
5. Show camera input: optional.
6. Show robot state: optional.
7. Show predicted action: useful for technical appendix, not README top.
8. Show expert activation: no.
9. README crop: 5-10 second comparison only if visually clear.
10. Format: MP4 link preferred; GIF only if small.
11. Suggested filename: `assets/demos/arx5_dp_cfg_chunk_comparison.mp4`.

## Demo C: Observation Visualization

1. Video content: one frame/timestep showing camera_0, camera_1, camera_2 and low-dimensional robot state text.
2. Suggested length: static figure or 3-5 second clip.
3. Recommended camera: generated from dataset frames, not an external camera.
4. Screen recording: not needed.
5. Show camera input: yes.
6. Show robot state: yes.
7. Show predicted action: optional.
8. Show expert activation: no.
9. README crop: one static figure under architecture or data section.
10. Format: PNG.
11. Suggested filename: `assets/figures/arx5_observation_example.png`.

## Demo D: Multi-task / MoE

1. Video content: future multi-task or expert-routing behavior.
2. Suggested length: TBD.
3. Recommended camera: fixed third-person camera plus optional observation grid.
4. Screen recording: optional.
5. Show camera input: yes if explaining multimodal behavior.
6. Show robot state: optional.
7. Show predicted action: optional.
8. Show expert activation: only after the method is approved for public release.
9. README crop: not for v1.
10. Format: MP4 external link.
11. Suggested filename: `assets/demos/arx5_multitask_moe_ongoing.mp4`.

## README Recommendation

Use exactly one primary demo near the README top. Put additional demos under a `## Demos` section or a GitHub Release.

