import numpy as np

class ROSInterfaceMock:
    def __init__(self):
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.dof_home = {name: 0.0 for name in self.joint_names}

    def get_joint_positions(self):
        return np.zeros(len(self.joint_names), dtype=np.float32)

    def get_joint_velocities(self):
        return np.zeros(len(self.joint_names), dtype=np.float32)

    def get_tcp_position(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def control_dofs_position(self, target_pos, max_vel=0.3, max_accel=0.3):
        pass

    def move_to_home(self):
        pass

    def publish_target_pos(self, pos):
        pass

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()
