# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- [PR-63](https://github.com/AGH-CEAI/aegis_gym/pull/63) - Fixed conditional checks for action commands.
- [PR-46](https://github.com/AGH-CEAI/aegis_gym/pull/46) - Fixed model sorting and typos.
- [PR-26](https://github.com/AGH-CEAI/aegis_gym/pull/26) - Fixed pose retrieval for Genesis entities.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Fixed TCP link handling in Genesis.

### Security
