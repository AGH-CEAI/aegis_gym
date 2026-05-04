# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [PR-93](https://github.com/AGH-CEAI/aegis_gym/pull/93) - Added feature to reset last layers of RL/BC algorithms.
- [PR-92](https://github.com/AGH-CEAI/aegis_gym/pull/92) - Added multi-checkpoint evaluation.
- [PR-82](https://github.com/AGH-CEAI/aegis_gym/pull/82) - Added utility scripts to manage ClearML tasks: `clearml_enquee_tasks.py`, `clearml_exp_plotter.py` & `clearml_summarizer.py`.
- [PR-86](https://github.com/AGH-CEAI/aegis_gym/pull/86) - Added attention-based fusion modules.
- [PR-80](https://github.com/AGH-CEAI/aegis_gym/pull/80) - Added support for autoencoder.
- [PR-83](https://github.com/AGH-CEAI/aegis_gym/pull/83) - Added lighting.
- [PR-75](https://github.com/AGH-CEAI/aegis_gym/pull/75) - Added ability to Load trained models from ClearML via new CLI arguments: `--load-rl-task-id` or `--load-rl-model-id`.
- [PR-77](https://github.com/AGH-CEAI/aegis_gym/pull/77) - Added template script for hyperparameter optimization.

### Changed

- [PR-95](https://github.com/AGH-CEAI/aegis_gym/pull/95) - Changed URDF IDs.
- [PR-95](https://github.com/AGH-CEAI/aegis_gym/pull/95) - Changed table height and object color.
- [PR-95](https://github.com/AGH-CEAI/aegis_gym/pull/95) - Changed workbench height and scene camera offset matrix to support updated robot model.
- [PR-82](https://github.com/AGH-CEAI/aegis_gym/pull/82) - Using `uv run --script` to run utility scripts (updated `upload_urdf_to_clearml.py`).
- [PR-86](https://github.com/AGH-CEAI/aegis_gym/pull/86) - Set rasterizer and cell visualization as defaults.
- [PR-86](https://github.com/AGH-CEAI/aegis_gym/pull/86) - Moved pooling and flattening from vision encoders to fusion modules; encoders return feature maps.
- [PR-85](https://github.com/AGH-CEAI/aegis_gym/pull/85) - Made teacher action mixing optional.
- [PR-83](https://github.com/AGH-CEAI/aegis_gym/pull/83) - Changed table color.
- [PR-76](https://github.com/AGH-CEAI/aegis_gym/pull/76) - Increased genesis logging level from "warning" to "info".



### Deprecated
### Removed
### Fixed

- [PR-95](https://github.com/AGH-CEAI/aegis_gym/pull/95) - Fixed missing pool size in config and arguments.
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed wrong model path for teacher in BC training
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed ClearML task overwriting on repeated runs.
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed BC evaluation not logging to ClearML.

### Security

## [v202603091815]

### Added

- [PR-72](https://github.com/AGH-CEAI/aegis_gym/pull/72) - Added flag to disable visual observations in Grasp ROS environment.
- [PR-65](https://github.com/AGH-CEAI/aegis_gym/pull/65) - Added support for both shared convolutional encoder and per-camera convolutional encoders.
- [PR-65](https://github.com/AGH-CEAI/aegis_gym/pull/65) - Introduced modular vision encoder interface for behavior cloning policies.
- [PR-67](https://github.com/AGH-CEAI/aegis_gym/pull/67) - Grasp Env: added support for new gRPC client joints enum.
- [PR-67](https://github.com/AGH-CEAI/aegis_gym/pull/67) - Grasp Env: Added `gripper_width` properties to the `manipulator` abstractions.
- [PR-67](https://github.com/AGH-CEAI/aegis_gym/pull/67) - Grasp Env: Added `--plotjuggler` flag to the eval script.
- [PR-64](https://github.com/AGH-CEAI/aegis_gym/pull/64) - Grasp Env: Added flag `--visualize-camera` to train, for preview of the simulated cameras.
- [PR-63](https://github.com/AGH-CEAI/aegis_gym/pull/63) - Added success checking for Grasp environment.
- [PR-61](https://github.com/AGH-CEAI/aegis_gym/pull/61) - Added reading visual observations from the gRPC server.
- [PR-60](https://github.com/AGH-CEAI/aegis_gym/pull/60) - Added option for calibration movement.
- [PR-56](https://github.com/AGH-CEAI/aegis_gym/pull/56) - Added Grasp Env logging to PlotJuggler (via UDP server).
- [PR-54](https://github.com/AGH-CEAI/aegis_gym/pull/53) - Added ClearML logging to behavioral cloning.
- [PR-49](https://github.com/AGH-CEAI/aegis_gym/pull/49) - Utility script for uploading the URDF assets to the ClearML server as a dataset.
- [PR-48](https://github.com/AGH-CEAI/aegis_gym/pull/48) - Added real robot control via gRPC in RSL-RL Grasp env.
- [PR-46](https://github.com/AGH-CEAI/aegis_gym/pull/46) - Added TCP-to-object Grasp environment.
- [PR-40](https://github.com/AGH-CEAI/aegis_gym/pull/40) - Added scene and tool cameras setup for Grasp environment.
- [PR-39](https://github.com/AGH-CEAI/aegis_gym/pull/39) - Added Grasp environment compatible with RSL-RL.
- [PR-32](https://github.com/AGH-CEAI/aegis_gym/pull/32) - `ROSRoboticCommander`: Added servoing (twist/jog) with MoveIt2 Servo.
- [PR-31](https://github.com/AGH-CEAI/aegis_gym/pull/31) - Introduced control frequency parameter to decouple policy updates from physics steps in Genesis.
- [PR-28](https://github.com/AGH-CEAI/aegis_gym/pull/28) - `AegisReacher`: Observation normalization for TCP and Target positions.
- [PR-26](https://github.com/AGH-CEAI/aegis_gym/pull/26) - Added support for multimodal observations in environments.
- [PR-26](https://github.com/AGH-CEAI/aegis_gym/pull/26) - Added a scene camera to Genesis simulation for visual observations.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Added Cartesian control for Genesis robot commander.
- [PR-19](https://github.com/AGH-CEAI/aegis_gym/pull/19) - Added Cartesian control for ROS robot commander.
- [PR-17](https://github.com/AGH-CEAI/aegis_gym/pull/17) - Added Cartesian control for Reacher environment.
- [PR-11,12,13,14,15](https://github.com/AGH-CEAI/aegis_gym/pull/5) - Added Genesis simulator.
- [PR-2](https://github.com/AGH-CEAI/aegis_gym/pull/2) - Added reinforcement learning Reacher environment for the Aegis robot station.
- [PR-1](https://github.com/AGH-CEAI/aegis_gym/pull/1) - Initial package with boilerplate and gymnasium API.

### Changed

- [PR-72](https://github.com/AGH-CEAI/aegis_gym/pull/72) - Enforced single environment for Grasp ROS training.
- [PR-69](https://github.com/AGH-CEAI/aegis_gym/pull/69) - Migrated from `poetry` to `uv` python package manager.
- [PR-59](https://github.com/AGH-CEAI/aegis_gym/pull/59) - Changed control type in simulated GraspEnv to match the real one (velocity control).
- [PR-58](https://github.com/AGH-CEAI/aegis_gym/pull/58) - Major development towards sim2real calibration.
- [PR-49](https://github.com/AGH-CEAI/aegis_gym/pull/49) - The `AegisGrasp`'s rsl_rl robot config accepts ID to download URDF dataset from ClearML (see [aegis_ros PR-95](https://github.com/AGH-CEAI/aegis_ros/pull/95)).
- [PR-42](https://github.com/AGH-CEAI/aegis_gym/pull/42) - Extracted Grasp environment configs to a new file.
- [PR-42](https://github.com/AGH-CEAI/aegis_gym/pull/42) - Ported Grasp environment to use `rsl-rl-lib==3.3.0`.
- [PR-38](https://github.com/AGH-CEAI/aegis_gym/pull/38) - Changed `ur_base` frame to the `world` frame.
- [PR-36](https://github.com/AGH-CEAI/aegis_gym/pull/36) - Changed target spawn ranges in Reacher environment.
- [PR-31](https://github.com/AGH-CEAI/aegis_gym/pull/31) - Changed physics timesteps and substeps.
- [PR-31](https://github.com/AGH-CEAI/aegis_gym/pull/31) - Changed robot joint PD gains in Genesis.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Changed robot joint PD gains in Genesis.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Differentiated between synchronous and asynchronous control methods.
- [PR-11,12,13,14,15](https://github.com/AGH-CEAI/aegis_gym/pull/5) - REFACTOR: Abstract interfaces for Genesis sim and ROS control.
- [PR-4](https://github.com/AGH-CEAI/aegis_gym/pull/4) - Updated & fixed dependencies versions.

### Deprecated

- [PR-66](https://github.com/AGH-CEAI/aegis_gym/pull/66) - Grasp Env: Policy position control from Genesis.
- [PR-3](https://github.com/AGH-CEAI/aegis_gym/pull/3) - Automatic change to ROSInterfaceMock for PyTest environment.

### Removed

- [PR-66](https://github.com/AGH-CEAI/aegis_gym/pull/66) - Grasp Env: Reverted PD gains from [PR-58](https://github.com/AGH-CEAI/aegis_gym/pull/58).
- [PR-49](https://github.com/AGH-CEAI/aegis_gym/pull/49) - Removed automatic URDF generation with `xacro`.

### Fixed

- [PR-67](https://github.com/AGH-CEAI/aegis_gym/pull/67) - Grasp Env: Fixed real `grasp_and_lift_demo()`.
- [PR-67](https://github.com/AGH-CEAI/aegis_gym/pull/67) - Grasp Env: Fixed joints getters.
- [PR-63](https://github.com/AGH-CEAI/aegis_gym/pull/63) - Fixed conditional checks for action commands.
- [PR-46](https://github.com/AGH-CEAI/aegis_gym/pull/46) - Fixed model sorting and typos.
- [PR-26](https://github.com/AGH-CEAI/aegis_gym/pull/26) - Fixed pose retrieval for Genesis entities.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Fixed TCP link handling in Genesis.
