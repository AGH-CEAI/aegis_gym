"""
Spring-mounted press plate: a rigid box on three slide joints with stiffness and damping.

Stands in for a compliant surface (a sponge, a foam pad) with a stiffness set in N/m, independent
of the arm, which Genesis' soft contact cannot give: its softness is scaled by the effective mass.
The contact itself stays rigid-rigid, so `get_contacts()` and the simulated F/T sensor see it as
any other contact. The normal axis has a hard stop after `travel`, like the table under a sponge;
the two shear axes reproduce the sideways compliance (friction keeps the fingers stuck to the top).
"""

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch as th

_MJCF = """<mujoco model="spring_plate">
  <worldbody>
    <body name="plate" pos="{x} {y} {z}">
      <joint name="shear_x" type="slide" axis="1 0 0" stiffness="{ksx}" damping="{ds}"/>
      <joint name="shear_y" type="slide" axis="0 1 0" stiffness="{ksy}" damping="{ds}"/>
      <joint name="normal" type="slide" axis="0 0 1" stiffness="{kn}" damping="{dn}"
             limited="true" range="{low} {high}"/>
      <geom type="box" pos="0 0 {gz}" size="{hx} {hy} {hz}" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class SpringPlateParams:
    """Defaults: the kitchen sponge, fitted to the robot's press (a1_press.csv, 2026-11-04)."""

    size: tuple[float, float, float] = (0.15, 0.15, 0.01)  # [m] full box size
    normal_stiffness: float = 1370.0  # [N/m]
    normal_damping: float = 20.0  # [N*s/m]
    # Softer in y: the robot's y ripple sits near the free admittance mode (1.6 Hz).
    shear_stiffness: tuple[float, float] = (1650.0, 300.0)  # [N/m] world x, y
    shear_damping: float = 20.0  # [N*s/m]
    travel: float = 0.03  # [m] compression before the hard stop
    mass: float = 0.5  # [kg]; with k it sets the plate's own frequency (>> the arm's)
    friction: float = 0.6
    # Random sideways force on the plate per call of `apply_surface_noise()`, standing in for what
    # keeps the real contact ringing (uneven sponge, micro stick-slip); reaches the gripper only
    # through the contact.
    surface_noise_std: tuple[float, float] = (0.35, 0.14)  # [N] world x, y
    surface_noise_cutoff_hz: float = 3.0  # first-order low-pass of the noise


class SpringPlate:
    """Joint order of `deflection()` and the setters: shear x, shear y, normal (world axes)."""

    def __init__(
        self,
        gs_scene: gs.Scene,
        top_center: tuple[float, float, float],
        params: SpringPlateParams | None = None,
    ):
        self._gs_scene = gs_scene
        self.top_center = top_center
        self.params = params or SpringPlateParams()
        self._entity = None
        self._noise: th.Tensor | None = None

    def create(self) -> None:
        p = self.params
        hx, hy, hz = (0.5 * s for s in p.size)
        mjcf = _MJCF.format(
            x=self.top_center[0],
            y=self.top_center[1],
            z=self.top_center[2],
            ksx=p.shear_stiffness[0],
            ksy=p.shear_stiffness[1],
            ds=p.shear_damping,
            kn=p.normal_stiffness,
            dn=p.normal_damping,
            low=-p.travel,
            high=p.travel,
            gz=-hz,
            hx=hx,
            hy=hy,
            hz=hz,
            mass=p.mass,
            mu=p.friction,
        )
        # The MJCF morph loads from a file; the body origin is the top face, so the joint
        # positions are the deflection of the surface itself.
        path = Path(tempfile.mkdtemp(prefix="aegis_spring_plate_")) / "spring_plate.xml"
        path.write_text(mjcf)
        self._entity = self._gs_scene.add_entity(
            gs.morphs.MJCF(file=str(path)),
            material=gs.materials.Rigid(
                gravity_compensation=1.0,
                friction=p.friction,
                coup_friction=p.friction,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.8, 0.2))
            ),
        )

    @property
    def entity(self):
        return self._entity

    @property
    def link(self):
        """The moving plate; the MJCF loader adds a fixed `world` link in front of it."""
        return self._entity.get_link("plate")

    def deflection(self) -> th.Tensor:
        """[num_envs, 3] displacement of the top face from rest [m]."""
        return self._entity.get_dofs_position()

    def apply_surface_noise(self, dt: float) -> None:
        """Pushes the plate sideways for the next `scene.step()` of length `dt`.

        Low-passed Gaussian noise (first order, AR(1) with a = exp(-2 pi f_c dt)), scaled so its
        stationary std is `surface_noise_std`: white noise would mostly excite the plate's own
        shear resonance instead of the arm's contact mode.
        """
        p = self.params
        std = th.tensor(p.surface_noise_std, device=gs.device)
        if not bool((std > 0.0).any()):
            return
        n = max(self._gs_scene.n_envs, 1)
        a = math.exp(-2.0 * math.pi * p.surface_noise_cutoff_hz * dt)
        fresh = th.randn(n, 2, device=gs.device) * std
        if self._noise is None:
            self._noise = fresh
        else:
            self._noise = a * self._noise + math.sqrt(1.0 - a * a) * fresh
        force = th.zeros(n, 1, 3, device=gs.device)
        force[:, 0, :2] = self._noise
        self._gs_scene.sim.rigid_solver.apply_links_external_wrench(
            force=force, links_idx=[self.link.idx]
        )

    def set_springs(
        self,
        normal_stiffness: th.Tensor | float | None = None,
        shear_stiffness: th.Tensor | float | None = None,
        normal_damping: th.Tensor | float | None = None,
        shear_damping: th.Tensor | float | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        """Per-env values ([len(envs_idx)]; shear also [len(envs_idx), 2] for world x, y) or one
        value for all; None keeps the current one."""
        n = self._gs_scene.n_envs if envs_idx is None else len(envs_idx)
        device = gs.device

        def merge(current: th.Tensor, normal, shear) -> th.Tensor:
            out = current.clone()
            if shear is not None:
                shear = th.as_tensor(shear, dtype=out.dtype, device=device)
                out[:, :2] = (
                    shear.reshape(-1, 2)
                    if shear.numel() == 2 * n
                    else shear.reshape(-1, 1)
                )
            if normal is not None:
                out[:, 2] = th.as_tensor(
                    normal, dtype=out.dtype, device=device
                ).reshape(-1)
            return out

        stiffness = self._entity.get_dofs_stiffness(envs_idx=envs_idx).reshape(n, 3)
        damping = self._entity.get_dofs_damping(envs_idx=envs_idx).reshape(n, 3)
        self._entity.set_dofs_stiffness(
            merge(stiffness, normal_stiffness, shear_stiffness), envs_idx=envs_idx
        )
        self._entity.set_dofs_damping(
            merge(damping, normal_damping, shear_damping), envs_idx=envs_idx
        )
