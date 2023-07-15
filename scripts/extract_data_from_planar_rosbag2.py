"""
Postprocess the experimental data recorded in rosbag2 files and save the results into a npz file.
Assumes the planar case.
"""

import nml_bag
import numpy as np
from pathlib import Path
from tqdm import tqdm


ROSBAG2_PATH = Path("/home/mstoelzle/phd/rosbags/rosbag2_20230714_135545/rosbag2_20230714_135545_0.mcap")

def main():
    reader = nml_bag.Reader(
        str(ROSBAG2_PATH),
        storage_id="mcap"
    )

    print("Available topics:\n", reader.topics)
    
    ts_q_ls, q_ls = [], []
    ts_q_d_ls, q_d_ls = [], []
    ts_chiee_ls, chiee_ls = [], []
    ts_phi_ls, phi_ls = [], []
    
    ts_q_des_ls, q_des_ls = [], []
    ts_chiee_des_ls, chiee_des_ls = [], []
    ts_phi_ss_ls, phi_ss_ls = [], []
    
    ts_phi_des_ls, phi_des_ls = [], []
    ts_phi_sat_ls, phi_sat_ls = [], []

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
            q_d_ls.append(np.array(msg["data"]))
        elif topic == "/end_effector_pose":
            chiee = np.array([msg["x"], msg["y"], msg["theta"]])
            ts_chiee_ls.append(time)
            chiee_ls.append(chiee)
        elif topic == "/actuation_coordinates":
            ts_phi_ls.append(time)
            phi_ls.append(np.array(msg["data"]))
        elif topic == "/setpoint_in_control_loop":
            ts_chiee_des_ls.append(time)
            chiee_des_ls.append(np.array([msg["chiee_des"]["x"], msg["chiee_des"]["y"], msg["chiee_des"]["theta"]]))
            ts_q_des_ls.append(time)
            q_des_ls.append(np.array([msg["q_des"]["kappa_b"], msg["q_des"]["sigma_sh"], msg["q_des"]["sigma_a"]]))
            ts_phi_ss_ls.append(time)
            phi_ss_ls.append(np.array(msg["phi_ss"]))
        elif topic == "/unsaturated_control_input":
            phi_des = np.array(msg["data"])
            ts_phi_des_ls.append(time)
            phi_des_ls.append(phi_des)
        elif topic == "/saturated_control_input":
            phi_sat = np.array(msg["data"])
            ts_phi_sat_ls.append(time)
            phi_sat_ls.append(phi_sat)

    data_ts = {
        "ts_q": np.array(ts_q_ls),
        "ts_q_d": np.array(ts_q_d_ls),
        "ts_chiee": np.array(ts_chiee_ls),
        "ts_chiee_des": np.array(ts_chiee_des_ls),
        "ts_q_des": np.array(ts_q_des_ls),
        "ts_phi_ss": np.array(ts_phi_ss_ls),
        "ts_phi_des": np.array(ts_phi_des_ls),
        "ts_phi_sat": np.array(ts_phi_sat_ls),
        "q_ts": np.stack(q_ls),
        "q_d_ts": np.stack(q_d_ls),
        "chiee_ts": np.stack(chiee_ls),
        "chiee_des_ts": np.stack(chiee_des_ls),
        "q_des_ts": np.stack(q_des_ls),
        "phi_ss_ts": np.stack(phi_ss_ls),
        "phi_des_ts": np.stack(phi_des_ls),
        "phi_sat_ts": np.stack(phi_sat_ls),
    }
    if len(ts_phi_ls) > 0:
        data_ts["ts_phi"] = np.array(ts_phi_ls)
        data_ts["phi_ts"] = np.stack(phi_ls)

    # save data
    np.savez(str(ROSBAG2_PATH.with_suffix(".npz")), **data_ts)
        

if __name__ == "__main__":
    main()
