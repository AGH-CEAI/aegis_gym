"""
Spring-mounted press plate: a rigid box on three slide joints with stiffness and damping.

Stands in for a compliant surface (a sponge, a foam pad) with a stiffness set in N/m, independent
of the arm, which Genesis' soft contact cannot give: its softness is scaled by the effective mass.
The contact itself stays rigid-rigid, so `get_contacts()` and the simulated F/T sensor see it as
any other contact. The normal axis has a hard stop after `travel`, like the table under a sponge;
the two shear axes reproduce the sideways compliance (friction keeps the fingers stuck to the top).
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch as th

_MJCF = """<mujoco model="spring_plate">
  <worldbody>
    <body name="plate" pos="{x} {y} {z}">
      <joint name="shear_x" type="slide" axis="1 0 0" stiffness="{ks}" damping="{ds}"/>
      <joint name="shear_y" type="slide" axis="0 1 0" stiffness="{ks}" damping="{ds}"/>
      <joint name="normal" type="slide" axis="0 0 1" stiffness="{kn}" damping="{dn}"
             limited="true" range="{low} {high}"/>
      <geom type="box" pos="0 0 {gz}" size="{hx} {hy} {hz}" mass="{mass}" friction="{mu} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass(frozen=True)
class SpringPlateParams:
    size: tuple[float, float, float] = (0.15, 0.15, 0.01)  # [m] full box size
    normal_stiffness: float = 1200.0  # [N/m]
    normal_damping: float = 20.0  # [N*s/m]
    shear_stiffness: float = 3000.0  # [N/m]
    shear_damping: float = 10.0  # [N*s/m]
    travel: float = 0.03  # [m] compression before the hard stop
    mass: float = 0.5  # [kg]; with k it sets the plate's own frequency (>> the arm's)
    friction: float = 0.6


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

    def create(self) -> None:
        p = self.params
        hx, hy, hz = (0.5 * s for s in p.size)
        mjcf = _MJCF.format(
            x=self.top_center[0],
            y=self.top_center[1],
            z=self.top_center[2],
            ks=p.shear_stiffness,
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

    def set_springs(
        self,
        normal_stiffness: th.Tensor | float | None = None,
        shear_stiffness: th.Tensor | float | None = None,
        normal_damping: th.Tensor | float | None = None,
        shear_damping: th.Tensor | float | None = None,
        envs_idx: th.Tensor | None = None,
    ) -> None:
        """Per-env values ([len(envs_idx)]) or one value for all; None keeps the current one."""
        n = self._gs_scene.n_envs if envs_idx is None else len(envs_idx)
        device = gs.device

        def merge(current: th.Tensor, normal, shear) -> th.Tensor:
            out = current.clone()
            if shear is not None:
                out[:, :2] = th.as_tensor(
                    shear, dtype=out.dtype, device=device
                ).reshape(-1, 1)
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
