# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [PR-40](https://github.com/AGH-CEAI/pull/40) - Added scene and tool cameras setup for Grasp environment.
- [PR-39](https://github.com/AGH-CEAI/pull/39) - Added Grasp environment compatible with RSL-RL.
- [PR-32](https://github.com/AGH-CEAI/pull/32) - `ROSRoboticCommander`: Added servoing (twist/jog) with MoveIt2 Servo.
- [PR-31](https://github.com/AGH-CEAI/pull/31) - Introduced control frequency parameter to decouple policy updates from physics steps in Genesis.
- [PR-28](https://github.com/AGH-CEAI/pull/28) - `AegisReacher`: Observation normalization for TCP and Target positions.
- [PR-26](https://github.com/AGH-CEAI/pull/26) - Added support for multimodal observations in environments.
- [PR-26](https://github.com/AGH-CEAI/pull/26) - Added a scene camera to Genesis simulation for visual observations.
- [PR-21](https://github.com/AGH-CEAI/pull/21) - Added Cartesian control for Genesis robot commander.
- [PR-19](https://github.com/AGH-CEAI/pull/19) - Added Cartesian control for ROS robot commander.
- [PR-17](https://github.com/AGH-CEAI/pull/17) - Added Cartesian control for Reacher environment.
- [PR-11,12,13,14,15](https://github.com/AGH-CEAI/aegis_gym/pull/5) - Added Genesis simulator.
- [PR-2](https://github.com/AGH-CEAI/aegis_gym/pull/2) - Added reinforcement learning Reacher environment for the Aegis robot station.
- [PR-1](https://github.com/AGH-CEAI/aegis_gym/pull/1) - Initial package with boilerplate and gymnasium API.

### Changed

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

- [PR-3](https://github.com/AGH-CEAI/aegis_gym/pull/3) - Automatic change to ROSInterfaceMock for PyTest environment.

### Removed

### Fixed

- [PR-26](https://github.com/AGH-CEAI/pull/26) - Fixed pose retrieval for Genesis entities.
- [PR-21](https://github.com/AGH-CEAI/aegis_gym/pull/21) - Fixed TCP link handling in Genesis.

### Security
