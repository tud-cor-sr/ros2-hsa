"""
Postprocess the experimental data recorded in rosbag2 files and save the results into a npz file.
Assumes the planar case.
"""
import dill
import nml_bag
import numpy as np
from pathlib import Path
from tqdm import tqdm


experiment_name = "20230925_142209"
ROSBAG2_PATH = Path(
    f"/home/mstoelzle/phd/rosbags/rosbag2_{experiment_name}/rosbag2_{experiment_name}_0.db3"
)


def main():
    reader = nml_bag.Reader(str(ROSBAG2_PATH), storage_id="sqlite3")

    print("Available topics:\n", reader.topics)

    ts_q_ls, q_ls = [], []
    ts_q_d_ls, q_d_ls = [], []
    ts_chiee_ls, chiee_ls = [], []

    controller_info_ts = {
        "ts": [],
        "chiee_des": [],
        "q_des": [],
        "phi_ss": [],
        "optimality_error": [],
        "chiee": [],
        "chiee_d": [],
        "q": [],
        "q_d": [],
        "varphi_des": [],
        "varphi": [],
        "e_int": [],
        "phi_des_unsat": [],
        "phi_des_sat": [],
        "motor_goal_angles": [],
    }

    for msg in tqdm(reader):
        time_ns = msg["time_ns"]
        time = time_ns / 1e9
        topic = msg["topic"]

        if topic == "/configuration":
            q = np.array([msg["kappa_b"], msg["sigma_sh"], msg["sigma_a"]])
            ts_q_ls.append(time)
            q_ls.append(q)
        elif topic == "/configuration_velocity":
            ts_q_d_ls.append(time)
            q_d_ls.append(np.array([msg["kappa_b"], msg["sigma_sh"], msg["sigma_a"]]))
        elif topic == "/end_effector_pose":
            pose_msg = msg["pose"]
            chiee = np.array([pose_msg["x"], pose_msg["y"], pose_msg["theta"]])
            ts_chiee_ls.append(time)
            chiee_ls.append(chiee)
        elif topic == "/controller_info":
            controller_info_ts["ts"].append(time)

            if msg["planar_setpoint"] is not None:
                planar_setpoint = msg["planar_setpoint"]
                if planar_setpoint["chiee_des"] is not None:
                    controller_info_ts["chiee_des"].append(
                        np.array(
                            [
                                planar_setpoint["chiee_des"]["x"],
                                planar_setpoint["chiee_des"]["y"],
                                planar_setpoint["chiee_des"]["theta"],
                            ]
                        )
                    )
                if planar_setpoint["q_des"] is not None:
                    controller_info_ts["q_des"].append(
                        np.array(
                            [
                                planar_setpoint["q_des"]["kappa_b"],
                                planar_setpoint["q_des"]["sigma_sh"],
                                planar_setpoint["q_des"]["sigma_a"],
                            ]
                        )
                    )
                if planar_setpoint["phi_ss"] is not None:
                    controller_info_ts["phi_ss"].append(planar_setpoint["phi_ss"])
                if planar_setpoint["optimality_error"] is not None:
                    controller_info_ts["optimality_error"].append(
                        planar_setpoint["optimality_error"]
                    )

            if msg["chiee"] is not None:
                pose_msg = msg["chiee"]["pose"]
                controller_info_ts["chiee"].append(
                    np.array([pose_msg["x"], pose_msg["y"], pose_msg["theta"]])
                )
            if msg["chiee_d"] is not None:
                pose_msg = msg["chiee_d"]["pose"]
                controller_info_ts["chiee_d"].append(
                    np.array([pose_msg["x"], pose_msg["y"], pose_msg["theta"]])
                )

            if msg["q"] is not None:
                controller_info_ts["q"].append(
                    np.array(
                        [msg["q"]["kappa_b"], msg["q"]["sigma_sh"], msg["q"]["sigma_a"]]
                    )
                )
            if msg["q_d"] is not None:
                controller_info_ts["q_d"].append(
                    np.array(
                        [
                            msg["q_d"]["kappa_b"],
                            msg["q_d"]["sigma_sh"],
                            msg["q_d"]["sigma_a"],
                        ]
                    )
                )

            if msg["e_int"] is not None:
                controller_info_ts["e_int"].append(np.array(msg["e_int"]))

            if msg["varphi_des"] is not None:
                controller_info_ts["varphi_des"].append(np.array(msg["varphi_des"]))
            if msg["varphi"] is not None:
                controller_info_ts["varphi"].append(np.array(msg["varphi"]))

            if msg["phi_des_unsat"] is not None:
                controller_info_ts["phi_des_unsat"].append(
                    np.array(msg["phi_des_unsat"])
                )
            if msg["phi_des_sat"] is not None:
                controller_info_ts["phi_des_sat"].append(np.array(msg["phi_des_sat"]))

            if msg["motor_goal_angles"] is not None:
                controller_info_ts["motor_goal_angles"].append(
                    np.array(msg["motor_goal_angles"])
                )

    for key, value in controller_info_ts.items():
        if len(value) == 0:
            controller_info_ts.pop(key)
        controller_info_ts[key] = np.stack(value)

    data_ts = {
        "ts_q": np.array(ts_q_ls),
        "ts_q_d": np.array(ts_q_d_ls),
        "ts_chiee": np.array(ts_chiee_ls),
        "q_ts": np.stack(q_ls),
        "q_d_ts": np.stack(q_d_ls),
        "chiee_ts": np.stack(chiee_ls),
        "controller_info_ts": controller_info_ts,
    }

    # save data
    with open(str(ROSBAG2_PATH.with_suffix(".dill")), "wb") as f:
        dill.dump(data_ts, f)


if __name__ == "__main__":
    main()
