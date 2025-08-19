from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0
import numpy as np
from etils import epath


class HmiElbowBase(PoseEnvV0):

    def __init__(self, perturb_scale, obsd_model_path=None, seed=None):

        self.perturb_scale = perturb_scale
        model_path= (epath.resource_path('myosuite')/'envs'/'myo'/'assets'/'elbow'/'myoelbow_1dof6muscles.xml').as_posix()
        # breakpoint()
        kwargs={
            "target_jnt_range": {
                "r_elbow_flex": (0, 2.27),
            },
            "viz_site_targets": ("wrist",),
            "normalize_act": True,
            "pose_thd": 0.175,
            "reset_type": "random",
            }
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, **kwargs)

    def reset(self, **kwargs):
        perturb_force = self.perturb_scale * np.random.randn()
        self.sim.data.qfrc_applied = perturb_force
        return super().reset(**kwargs)