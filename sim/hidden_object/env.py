"""ManiSkill3 scene for cross-episode hidden-object localisation.

The scene holds a table inside a textured room, a parked Panda robot, three inverted
YCB mugs as movable covers, a yellow target ball and one or two distractor objects.
Covers, ball and distractors are kinematic: an invisible external actor moves them
by setting poses, so no physics is stepped. The layout of a trial is passed through
``env.reset(options={"reconfigure": True, "trial": trial_dict})``; recording is
driven from ``trial.py`` via ``place``/``set_camera``/``capture``.
"""
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, quat2mat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .textures import texture_file

FLIP = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz: 180 deg about x, turns a mug upside down


def default_trial() -> dict:
    """Minimal layout used when the env is created without a trial (e.g. by gym.make)."""
    return {
        "scene": {"scene_id": 0, "texture_seed": 100, "distractors": []},
        "cover_colors": [[0.5, 0.5, 0.5]] * 3,
        "cover_yaw": 0.0,
        "distractors": [],
    }


@register_env("HiddenObjectShell-v0", max_episode_steps=10_000, asset_download_ids=["ycb"])
class HiddenObjectShellEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    COVER_MODEL = "025_mug"
    COVER_SCALE = 1.3
    BALL_RADIUS = 0.02
    BALL_COLOR = [1.0, 0.82, 0.1, 1.0]
    IMG_W, IMG_H, FOVY_DEG = 640, 480, 50.0
    ROOM_HALF, ROOM_HEIGHT = 1.8, 2.6

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.trial = default_trial()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ config
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.8, 0.0, 0.6], target=[0.0, 0.0, 0.03])
        return [CameraConfig("rec_camera", pose, self.IMG_W, self.IMG_H, np.deg2rad(self.FOVY_DEG), 0.01, 20)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.8, 0.0, 0.6], target=[0.0, 0.0, 0.03])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 20)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_lighting(self, options: dict):
        # One shadow-casting key light plus a weak fill light: shadows give VGGT and the
        # feature matcher additional cues compared with flat, shadowless lighting.
        self.scene.set_ambient_light([0.25, 0.25, 0.25])
        self.scene.add_directional_light(
            [0.6, 0.4, -1.0], [1.6, 1.6, 1.6], shadow=self.enable_shadow, shadow_scale=3, shadow_map_size=4096
        )
        self.scene.add_directional_light([-0.5, -0.3, -1.0], [0.35, 0.35, 0.35])

    # ------------------------------------------------------------------- scene
    def _load_scene(self, options: dict):
        self.trial = options.get("trial") or default_trial()
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        self._build_room(self.trial["scene"]["texture_seed"])
        self.covers = [self._build_cover(i, rgb) for i, rgb in enumerate(self.trial["cover_colors"])]
        self.ball = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=self.BALL_COLOR,
            name="target_ball",
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )
        self.distractors = [
            self._build_ycb(d["model_id"], f"distractor_{j}") for j, d in enumerate(self.trial["distractors"])
        ]

    def _build_room(self, texture_seed: int):
        floor_z = -self.table_scene.table_height
        r, h = self.ROOM_HALF, self.ROOM_HEIGHT
        panels = [  # (centre, half size, texture style)
            ([-r, 0, floor_z + h / 2], [0.01, r, h / 2], "wall"),
            ([r, 0, floor_z + h / 2], [0.01, r, h / 2], "wall"),
            ([0, -r, floor_z + h / 2], [r, 0.01, h / 2], "wall"),
            ([0, r, floor_z + h / 2], [r, 0.01, h / 2], "wall"),
            ([0, 0, floor_z + 0.005], [r, r, 0.005], "floor"),
        ]
        for k, (centre, half, style) in enumerate(panels):
            material = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1], roughness=0.9, metallic=0.0)
            material.base_color_texture = sapien.render.RenderTexture2D(texture_file(texture_seed * 10 + k, style))
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(half_size=half, material=material)
            builder.initial_pose = sapien.Pose(p=centre)
            builder.build_static(name=f"room_{k}")

    def _build_cover(self, idx: int, rgb) -> Actor:
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.COVER_MODEL}")
        material = sapien.render.RenderMaterial(base_color=[*rgb, 1.0], roughness=0.65, metallic=0.0)
        for record in builder.collision_records:
            record.scale = (np.asarray(record.scale, dtype=np.float32) * self.COVER_SCALE).tolist()
        for record in builder.visual_records:
            record.scale = (np.asarray(record.scale, dtype=np.float32) * self.COVER_SCALE).tolist()
            record.material = material  # same mesh for all covers, only the colour differs
        builder.initial_pose = sapien.Pose(p=[0, 0, 2.0 + 0.3 * idx])
        return builder.build_kinematic(name=f"cover_{idx}")

    def _build_ycb(self, model_id: str, name: str):
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 3.0])
        return builder.build_kinematic(name=name)

    def _after_reconfigure(self, options: dict):
        # Resting heights: distance from the actor origin to the lowest point of its
        # collision mesh in the orientation it will have on the table.
        def rest_height(actor, quat):
            verts = np.asarray(actor.get_first_collision_mesh(to_world_frame=False).vertices)
            return float(-(verts @ quat2mat(quat).T)[:, 2].min())

        self.cover_rest_z = rest_height(self.covers[0], FLIP)
        self.distractor_rest_z = [rest_height(d, np.array([1.0, 0, 0, 0])) for d in self.distractors]

    # Parked arm folded back and up so that it does not hang over the cover area.
    PARKED_QPOS = [0.0, -0.9, 0.0, -2.8, 0.0, 2.2, 0.785, 0.04, 0.04]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        self.agent.reset(torch.as_tensor(np.asarray([self.PARKED_QPOS], dtype=np.float32)))

    # ------------------------------------------------------- task interface
    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}

    # ----------------------------------------------------- recording helpers
    def place(self, actor, xyz, yaw: float = 0.0, flip: bool = False):
        q = euler2quat(0.0, 0.0, yaw)
        if flip:
            q = qmult(q, FLIP)
        p = torch.as_tensor(np.asarray([xyz], dtype=np.float32))
        actor.set_pose(Pose.create_from_pq(p=p, q=torch.as_tensor(np.asarray([q], dtype=np.float32))))

    def place_cover(self, idx: int, xy, lift: float = 0.0):
        self.place(self.covers[idx], [xy[0], xy[1], self.cover_rest_z + lift], yaw=self.trial["cover_yaw"], flip=True)

    def place_ball(self, xy):
        self.place(self.ball, [xy[0], xy[1], self.BALL_RADIUS])

    def place_distractors(self):
        for actor, cfg, z in zip(self.distractors, self.trial["distractors"], self.distractor_rest_z):
            self.place(actor, [cfg["xy"][0], cfg["xy"][1], z], yaw=cfg["yaw"])

    def set_camera(self, eye, target):
        self._sensors["rec_camera"].camera.set_local_pose(sapien_utils.look_at(eye=eye, target=target).sp)

    def capture(self) -> dict:
        """Render the recording camera; returns rgb, depth (m), segmentation ids, K and extrinsic."""
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        sensor = self._sensors["rec_camera"]
        sensor.capture()
        obs = sensor.get_obs(rgb=True, depth=True, position=False, segmentation=True)
        depth = obs["depth"][0, ..., 0].cpu().numpy()
        if np.issubdtype(depth.dtype, np.integer):  # ManiSkill returns depth in millimetres
            depth = depth.astype(np.float32) / 1000.0
        return {
            "rgb": obs["rgb"][0].cpu().numpy().astype(np.uint8),
            "depth": depth.astype(np.float32),
            "seg": obs["segmentation"][0, ..., 0].cpu().numpy().astype(np.int32),
            "K": sensor.camera.get_intrinsic_matrix()[0].cpu().numpy(),
            "extrinsic_cv": sensor.camera.get_extrinsic_matrix()[0].cpu().numpy(),
        }

    def segmentation_ids(self) -> dict:
        ids = {f"cover_{i}": int(c.per_scene_id[0]) for i, c in enumerate(self.covers)}
        ids["target_ball"] = int(self.ball.per_scene_id[0])
        ids.update({f"distractor_{j}": int(d.per_scene_id[0]) for j, d in enumerate(self.distractors)})
        return ids
