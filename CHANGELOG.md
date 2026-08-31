# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [PR-148](https://github.com/AGH-CEAI/aegis_gym/pull/148) - Added the Push-T task.
- [PR-148](https://github.com/AGH-CEAI/aegis_gym/pull/148) - Added a `--env` flag and an environment registry to select between the environments by name.
- [PR-148](https://github.com/AGH-CEAI/aegis_gym/pull/148) - Added `MESH` and `URDF` object types to the scene abstraction.
- [PR-137](https://github.com/AGH-CEAI/aegis_gym/pull/137) - AegisGrasp cleanup p.6: Introdcued `BaseObjectsFactory` with custom implementations for `RosGrpcObjectsFactory` and `GenesisFactory`.
- [PR-138](https://github.com/AGH-CEAI/aegis_gym/pull/138) - Added the scripts to autobuild the development toolbox.
- [PR-136](https://github.com/AGH-CEAI/aegis_gym/pull/136) - AegisGrasp cleanup p.5: introduced `BasePolicyRunner` abstraction for wrapping rsl_rl's `OnPolicyRunner` and the `BehaviorCloningRunner`.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - Introducing observation cache in the `BaseEnv`class, which allows to query the same observation multiple times without re-calculating the observation.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - Added `Modality` enum for indicating all available and default modalities in `BaseEnv` specializations.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - Added `RandomizationTypes` enum for flagging `BaseScene` specialization available domain randomization capabilities.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - AegisGrasp cleanup p.4: Added `BaseEnvWrapper` with an implementation for visual observations `VisualAugEnvWrapper` and for debugging preview & recording `ObsPreviewEnvWrapper`.
- [PR-132](https://github.com/AGH-CEAI/aegis_gym/pull/132) - AegisGrasp cleanup p.3: Added `CamerasSetup` enum in the config.
- [PR-123](https://github.com/AGH-CEAI/aegis_gym/pull/123) - AegisGrasp cleanup p.2: Added `BaseEnv`, `BaseManipulator` and `BaseScene` abstractions for use in AegisGrasp.

### Changed

- [PR-148](https://github.com/AGH-CEAI/aegis_gym/pull/148) - Grouped environment config fields into shared and task-specific sections for clarity.
- [PR-153](https://github.com/AGH-CEAI/aegis_gym/pull/153) - Moved Reacher's hardcoded spawn box ranges into config fields.
- [PR-147](https://github.com/AGH-CEAI/aegis_gym/pull/147) - Replace print statements with logging. Divide aux section ingo sub-packages (e.g. logging)
- [PR-145](https://github.com/AGH-CEAI/aegis_gym/pull/145) - AegisGrasp cleanup p.7: Moved files out from the `rsl` folder. Reforatted codebase according to the ruff v0.16.0. Updated main README.md.
- [PR-144](https://github.com/AGH-CEAI/aegis_gym/pull/144) - Using new strenum literals from `aegis_grpc` client.
- [PR-137](https://github.com/AGH-CEAI/aegis_gym/pull/137) - AegisGrasp cleanup p.6: `GraspEnv` was reworked into `ReacherEnv` using all previous abstractions.
- [PR-136](https://github.com/AGH-CEAI/aegis_gym/pull/136) - AegisGrasp cleanup p.5: Sliced `BehaviorCloning` into smaller files.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - AegisGrasp cleanup p.4: Moved all debug preview code from the `grasp_env.py` into a new env-wrapper: `ObsPreviewEnvWrapper`.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - AegisGrasp cleanup p.4: Moved all visual augumentation from the domain randomization (PR #116) from the `grasp_env.py` into a new env-wrapper: `VisualAugEnvWrapper`.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - AegisGrasp cleanup p.4: Totally makeover of the `get_observations()` and `get_observations_vis()` API into `get_modality_observations()`, `_build_agent_observations()` and `_format_rslrl_observations()`.
- [PR-132](https://github.com/AGH-CEAI/aegis_gym/pull/132) - AegisGrasp cleanup p.3: Extracted objects into `BaseObject` interface.
- [PR-129](https://github.com/AGH-CEAI/aegis_gym/pull/129) - AegisGrasp cleanup p.2: Config total makeover into `ConfigManager` and `LaunchArgs` classes. Added strcit config creation control (via frozen dataclasses).
- [PR-123](https://github.com/AGH-CEAI/aegis_gym/pull/123) - AegisGrasp cleanup p.1: Created `BaseEnv` interface and loosely used it for both of the grasp env implementations.
- [PR-123](https://github.com/AGH-CEAI/aegis_gym/pull/123) - AegisGrasp cleanup p.1: Refactored `Manipulator` interfaces into `BaseManipulator` abstraction and its `RosGrpcManipulator` and `GenesisManipualtor` implementations.


### Deprecated

- [PR-137](https://github.com/AGH-CEAI/aegis_gym/pull/137) - AegisGrasp cleanup p.6: removed the BC checkpoints sweep feature.

### Removed

- [PR-146](https://github.com/AGH-CEAI/aegis_gym/pull/146) - Moved `utils` ClearML scripts to [clearml_utils](https://github.com/AGH-CEAI/clearml_utils).
- [PR-117](https://github.com/AGH-CEAI/aegis_gym/pull/113) - AegisGrasp cleanup p.4: Removed flag `--debug-swap-tool-cameras` for swapping RGB tool cameras sides (left <-> right). The new `Modality` module ensures the correct order of the cameras.

### Fixed
- [PR-148](https://github.com/AGH-CEAI/aegis_gym/pull/148) - Fixed simulation-only runs crashing on startup because of an unconditional import of the real-hardware gRPC manipulator client, which fails in containers with an incompatible `protobuf` version. The import is now optional.
- [PR-150](https://github.com/AGH-CEAI/aegis_gym/pull/150) - Fixed RL training crashing on GPU resource limits at high environment counts by conditionally skipping camera and renderer setup.
- [PR-149](https://github.com/AGH-CEAI/aegis_gym/pull/149) - Fixed spawning the box with a hardcoded size instead of reading it from the config in Reacher task.
- [PR-142](https://github.com/AGH-CEAI/aegis_gym/pull/142) - Fixed `ConfigManager` silently dropping the default `max_iterations` and the `--debug-record-vis-obs` flag, which forced `--max-iterations` on every run and disabled visual observation recording.
- [PR-133](https://github.com/AGH-CEAI/aegis_gym/pull/133) - Fixed a typo in `genesis_manipulator.py` in `ctrl_gripper_X()` methods which was blocking more than 2 envs from running.


### Security

## [v202606181210]

### Added

- [PR-113](https://github.com/AGH-CEAI/aegis_gym/pull/113) - Added flag `---debug-enable-vis-preview` for showing cameras preview.
- [PR-113](https://github.com/AGH-CEAI/aegis_gym/pull/113) - Added flag `--debug-record-vis-obs` and `--debug-record-dir` arg for recording cameras preview into a given directory.
- [PR-113](https://github.com/AGH-CEAI/aegis_gym/pull/113) - Added flag `--debug-swap-tool-cameras` for swapping RGB tool cameras sides (left <-> right).
- [PR-113](https://github.com/AGH-CEAI/aegis_gym/pull/113) - Added flag `--episode-length-s` for overwriting the default episode time.
- [PR-116](https://github.com/AGH-CEAI/aegis_gym/pull/103) - Added domain randomization for Genesis environment.
- [PR-103](https://github.com/AGH-CEAI/aegis_gym/pull/103) - Added new `ConcatenatedCNNEncoder` encoder class.
- [PR-112](https://github.com/AGH-CEAI/aegis_gym/pull/112) - Added script to batch-enqueue ClearML eval tasks from training runs.
- [PR-94](https://github.com/AGH-CEAI/aegis_gym/pull/94) - Added ClearML config connection to the `grasp_cfg.py` config getters.
- [PR-104](https://github.com/AGH-CEAI/aegis_gym/pull/104) - Added best model checkpoint saving.
- [PR-106](https://github.com/AGH-CEAI/aegis_gym/pull/106) - Added rsl-rl option `detach_actor_grad` to stop propagation of policy gradients.

### Changed

- [PR-103](https://github.com/AGH-CEAI/aegis_gym/pull/103) - Parametrized the decoder builder in `AutoencoderCNNEncoder` (resolved issue #79).
- [PR-103](https://github.com/AGH-CEAI/aegis_gym/pull/103) - Tensor operation improvements in `grasp_env.py`, `grasp_env_ros.py`, `PerCameraCNNEncoder`, `SharedCNNEncoder` and `AutoencoderCNNEncoder`.
- [PR-109](https://github.com/AGH-CEAI/aegis_gym/pull/109) - Made pose head optional.
- [PR-97](https://github.com/AGH-CEAI/aegis_gym/pull/97) - Changed BC logging to every iteration (removed `log_freq` parameter from bc_config).
- [PR-96](https://github.com/AGH-CEAI/aegis_gym/pull/96) - `utils/clearml_enquee_tasks.py`: Improved information for the user.

### Fixed

- [PR-115](https://github.com/AGH-CEAI/aegis_gym/pull/115) - Fixed device mismatch in encoder shape inference.
- [PR-114](https://github.com/AGH-CEAI/aegis_gym/pull/114) - Fixed bug with config overwrite by args.
- [PR-96](https://github.com/AGH-CEAI/aegis_gym/pull/96) - Fixed bug with `matplotlib` dependency in  `utils/clearml_enquee_tasks.py`.

## [v202605061407]

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

### Fixed

- [PR-95](https://github.com/AGH-CEAI/aegis_gym/pull/95) - Fixed missing pool size in config and arguments.
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed wrong model path for teacher in BC training
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed ClearML task overwriting on repeated runs.
- [PR-88](https://github.com/AGH-CEAI/aegis_gym/pull/88) - Fixed BC evaluation not logging to ClearML.

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
