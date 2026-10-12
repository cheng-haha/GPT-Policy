"""Nominal simulated fingertip reference derived from the pinned URDF/STL."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .io import sha256
from .transforms import to_link6, to_tool, transform, xyzw_pose


def calibration(urdf):
    urdf = Path(urdf)
    tree = ET.parse(urdf).getroot()
    fingers = []
    for joint_name in ("joint7", "joint8"):
        joint = next(j for j in tree.findall("joint") if j.get("name") == joint_name)
        if joint.find("parent").get("link") != "link6":
            raise ValueError("Unexpected finger chain")
        origin = joint.find("origin")
        if not np.allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0):
            raise ValueError("Unexpected finger axis convention")
        child = joint.find("child").get("link")
        link = next(x for x in tree.findall("link") if x.get("name") == child)
        mesh = urdf.parent / link.find("collision/geometry/mesh").get("filename")
        data = mesh.read_bytes()
        count = int.from_bytes(data[80:84], "little")
        if len(data) != 84 + 50 * count:
            raise ValueError("Expected binary STL")
        vertices = np.frombuffer(
            data[84:],
            dtype=np.dtype([("n", "<f4", 3), ("p", "<f4", (3, 3)), ("a", "<u2")]),
        )["p"].reshape(-1, 3)
        tip = vertices[vertices[:, 0] > vertices[:, 0].max() - 1e-6]
        center = (tip.min(0) + tip.max(0)) / 2
        fingers.append(
            {
                "link": child,
                "origin": np.fromstring(origin.get("xyz"), sep=" ").tolist(),
                "axis": np.fromstring(joint.find("axis").get("xyz"), sep=" ").tolist(),
                "distal_center": center.tolist(),
                "mesh_sha256": sha256(mesh),
            }
        )
    matrix = np.eye(4)
    matrix[:3, 3] = np.mean(
        [np.array(x["origin"]) + x["distal_center"] for x in fingers], axis=0
    )
    return {
        "link6_from_tool": matrix.tolist(),
        "fingers": fingers,
        "urdf_sha256": sha256(urdf),
        "site": "nominal symmetric-gripper distal-face midpoint",
        "orientation": "same axes as link6",
        "physical_inner_contact_point_certified": False,
    }


def verify_native(kinematics, calibrations):
    result = {}
    for arm, robot in kinematics.robots.items():
        manager = kinematics.manager
        link = manager.get_real_endpose(robot)[0]
        link_matrix = transform(link[:3], link[3:])
        joints = manager.get_end_effector_real_val(robot)[0]
        errors = []
        for index, finger in enumerate(calibrations[arm]["fingers"]):
            native = manager.get_link_pose(robot, finger["link"], is_relative=True)[0]
            local = np.array(finger["distal_center"])
            measured = transform(native[:3], native[3:]) @ np.r_[local, 1]
            predicted = (
                link_matrix
                @ np.r_[
                    np.array(finger["origin"])
                    + np.array(finger["axis"]) * joints[index]
                    + local,
                    1,
                ]
            )
            errors.append(float(np.linalg.norm(measured[:3] - predicted[:3])))
        tool = np.array(calibrations[arm]["link6_from_tool"])
        base = kinematics.root(arm)
        tcp = to_tool(link_matrix, base, tool)
        roundtrip = float(np.max(np.abs(to_link6(tcp, base, tool) - link_matrix)))
        if max(errors) > 0.002 or roundtrip > 1e-9:
            raise ValueError("Simulation tool reference verification failed")
        result[arm] = dict(
            finger_fk_errors_m=errors,
            roundtrip_max_error=roundtrip,
            tool_pose_in_arm_base=xyzw_pose(tcp),
            **calibrations[arm],
        )
    return result
