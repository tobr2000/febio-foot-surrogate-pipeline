from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "templates" / "simplefoot_stance_ligamented_base.feb"
DEFAULT_RUNS = ROOT / "runs"
DEFAULT_FEBIO = r"C:\Program Files\FEBioStudio\bin\febio4.exe"


MATERIAL_NAMES = {
    "E_flesh": "Flesh_soft_tissue",
    "E_bone": "Cortical_bone_stiff",
    "E_joint": "Ankle_joint_pad",
    "E_heel": "Heel_pad_soft",
    "E_forefoot": "Forefoot_pad_soft",
    "E_collar": "Ankle_ligament_collar",
    "E_plantar": "Plantar_fascia_band",
    "E_achilles": "Achilles_like_band",
}


@dataclass(frozen=True)
class Sample:
    sample_id: int
    seed: int
    params: dict[str, float]

    @property
    def name(self) -> str:
        return f"sample_{self.sample_id:06d}"


def load_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                Sample(
                    sample_id=int(row["sample_id"]),
                    seed=int(row["seed"]),
                    params={k: float(v) for k, v in row["params"].items()},
                )
            )
    return samples


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def set_text(elem: ET.Element | None, value: float | str) -> None:
    if elem is None:
        raise ValueError("Cannot set missing XML element")
    elem.text = str(value)


def find_material(root: ET.Element, name: str) -> ET.Element:
    for material in root.findall("./Material/material"):
        if material.attrib.get("name") == name:
            return material
    raise KeyError(f"Material not found: {name}")


def find_contact(root: ET.Element) -> ET.Element:
    contact = root.find("./Contact/contact")
    if contact is None:
        raise KeyError("No Contact/contact element found")
    return contact


def find_load_controller(root: ET.Element, controller_id: str) -> ET.Element:
    for controller in root.findall("./LoadData/load_controller"):
        if controller.attrib.get("id") == controller_id:
            return controller
    raise KeyError(f"Load controller not found: {controller_id}")


def scale_deformable_nodes(root: ET.Element, sx: float, sy: float, sz: float) -> None:
    """Scale non-floor nodes while keeping the sole plane at z=0."""
    floor_ids = set()
    for elem_block in root.findall("./Mesh/Elements"):
        if elem_block.attrib.get("name") != "Floor":
            continue
        for elem in elem_block.findall("elem"):
            floor_ids.update(int(v) for v in elem.text.split(","))

    all_deformable = []
    for node in root.findall("./Mesh/Nodes/node"):
        node_id = int(node.attrib["id"])
        if node_id in floor_ids:
            continue
        x, y, z = (float(v) for v in node.text.split(","))
        all_deformable.append((node, x, y, z))

    if not all_deformable:
        return

    x_center = sum(x for _, x, _, _ in all_deformable) / len(all_deformable)
    for node, x, y, z in all_deformable:
        x2 = x_center + (x - x_center) * sx
        y2 = y * sy
        z2 = z * sz
        node.text = f"{x2:.10g},{y2:.10g},{z2:.10g}"


def deformable_node_records(root: ET.Element) -> list[tuple[ET.Element, float, float, float]]:
    floor_ids = set()
    for elem_block in root.findall("./Mesh/Elements"):
        if elem_block.attrib.get("name") != "Floor":
            continue
        for elem in elem_block.findall("elem"):
            floor_ids.update(int(v) for v in elem.text.split(","))

    records = []
    for node in root.findall("./Mesh/Nodes/node"):
        node_id = int(node.attrib["id"])
        if node_id in floor_ids:
            continue
        x, y, z = (float(v) for v in node.text.split(","))
        records.append((node, x, y, z))
    return records


def apply_shape_family(root: ET.Element, params: dict[str, float], include_base_profile: bool = True) -> None:
    """Apply deterministic base-family and sample-level geometric variation."""
    records = deformable_node_records(root)
    if not records:
        return

    xs = [x for _, x, _, _ in records]
    ys = [y for _, _, y, _ in records]
    zs = [z for _, _, _, z in records]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)
    z_span = max(1e-9, z_max - z_min)

    base_foot_length = params.get("base_foot_length", 1.0) if include_base_profile else 1.0
    base_foot_width = params.get("base_foot_width", 1.0) if include_base_profile else 1.0
    base_leg_length = params.get("base_leg_length", 1.0) if include_base_profile else 1.0
    base_arch_lift = params.get("base_arch_lift", 0.0) if include_base_profile else 0.0
    base_toe_splay = params.get("base_toe_splay", 0.0) if include_base_profile else 0.0
    arch_lift = params.get("arch_lift", 0.0)
    heel_toe_roll = params.get("heel_toe_roll", 0.0)
    toe_off_bias = params.get("toe_off_bias", 0.0)

    for node, x, y, z in records:
        nx = (x - x_mid) / x_span
        ny = (y - y_mid) / y_span
        nz = (z - z_min) / z_span
        foot_weight = 1.0 - min(1.0, max(0.0, (z - 0.18) / 0.42))
        leg_weight = 1.0 - foot_weight

        x2 = x_mid + (x - x_mid) * (1.0 + foot_weight * (base_foot_length - 1.0))
        y2 = y_mid + (y - y_mid) * (1.0 + foot_weight * (base_foot_width - 1.0))
        z2 = z_min + (z - z_min) * (1.0 + leg_weight * (base_leg_length - 1.0))

        arch_profile = math.sin(math.pi * min(1.0, max(0.0, (x - x_min) / x_span)))
        arch_profile *= max(0.0, 1.0 - abs(2.0 * (y - y_mid) / y_span))
        z2 += foot_weight * (base_arch_lift + arch_lift) * arch_profile

        toe_profile = max(0.0, nx)
        y2 += foot_weight * base_toe_splay * toe_profile * (1.0 if y >= y_mid else -1.0)
        z2 += foot_weight * toe_off_bias * toe_profile
        z2 += foot_weight * heel_toe_roll * nx

        # A slight coupled ankle bend differentiates base families without changing topology.
        bend = params.get("base_ankle_bend", 0.0) if include_base_profile else 0.0
        x2 += leg_weight * bend * nz * (z - z_min)

        node.text = f"{x2:.10g},{y2:.10g},{z2:.10g}"


def ensure_lateral_drive(root: ET.Element, lateral_disp: float) -> None:
    """Replace the top y-fixity with a prescribed lateral displacement controller."""
    boundary = root.find("./Boundary")
    if boundary is None:
        return
    for bc in list(boundary.findall("bc")):
        if bc.attrib.get("name") == "TibiaNoLateralDrift":
            # Once y is prescribed by TibiaLateralDrive, this old helper BC would
            # have no active DOFs. FEBio treats that as an initialization error.
            boundary.remove(bc)

    exists = any(bc.attrib.get("name") == "TibiaLateralDrive" for bc in boundary.findall("bc"))
    if not exists:
        bc = ET.SubElement(
            boundary,
            "bc",
            {
                "name": "TibiaLateralDrive",
                "node_set": "@surface:TibiaTopDriveSurface",
                "type": "prescribed displacement",
            },
        )
        ET.SubElement(bc, "dof").text = "y"
        ET.SubElement(bc, "value", {"lc": "3"}).text = "1"
        ET.SubElement(bc, "relative").text = "1"

    load_data = root.find("./LoadData")
    if load_data is None:
        load_data = ET.SubElement(root, "LoadData")
    controller = None
    for item in load_data.findall("load_controller"):
        if item.attrib.get("id") == "3":
            controller = item
            break
    if controller is None:
        controller = ET.SubElement(
            load_data,
            "load_controller",
            {"id": "3", "name": "lateral_sway", "type": "loadcurve"},
        )
        ET.SubElement(controller, "interpolate").text = "LINEAR"
        ET.SubElement(controller, "extend").text = "CONSTANT"
        ET.SubElement(controller, "points")
    points = controller.find("points")
    if points is None:
        points = ET.SubElement(controller, "points")
    for pt in list(points):
        points.remove(pt)
    ET.SubElement(points, "pt").text = "0,0"
    ET.SubElement(points, "pt").text = f"0.55,{lateral_disp:.10g}"
    ET.SubElement(points, "pt").text = f"1,{0.35 * lateral_disp:.10g}"


def add_output_loggers(root: ET.Element, sample_name: str) -> None:
    output = root.find("./Output")
    if output is None:
        output = ET.SubElement(root, "Output")

    for old in output.findall("logfile"):
        output.remove(old)

    logfile = ET.SubElement(output, "logfile")

    ET.SubElement(
        logfile,
        "node_data",
        {
            "data": "x;y;z;ux;uy;uz;Rx;Ry;Rz",
            "name": "node_state",
            "file": f"{sample_name}_nodes.csv",
            "delim": ",",
        },
    )
    ET.SubElement(
        logfile,
        "element_data",
        {
            "data": "x;y;z;sx;sy;sz;sxy;syz;sxz",
            "name": "element_stress",
            "file": f"{sample_name}_elements.csv",
            "delim": ",",
        },
    )
    ET.SubElement(
        logfile,
        "face_data",
        {
            "data": "contact gap;contact pressure",
            "name": "sole_contact",
            "surface": "FootSoleContact",
            "file": f"{sample_name}_contact.csv",
            "delim": ",",
        },
    )


def validate_boundary_conditions(root: ET.Element) -> None:
    boundary = root.find("./Boundary")
    if boundary is None:
        return
    for bc in boundary.findall("bc"):
        if bc.attrib.get("type") != "zero displacement":
            continue
        dof_values = []
        for tag in ("x_dof", "y_dof", "z_dof"):
            item = bc.find(tag)
            if item is not None:
                dof_values.append(str(item.text).strip() in {"1", "true", "True"})
        if dof_values and not any(dof_values):
            name = bc.attrib.get("name", "<unnamed>")
            raise ValueError(f"Zero displacement BC {name!r} has no active DOFs")


def render_feb(
    template: Path,
    sample: Sample,
    out_feb: Path,
    include_base_profile: bool = True,
) -> None:
    tree = ET.parse(template)
    root = tree.getroot()
    params = sample.params

    for key, material_name in MATERIAL_NAMES.items():
        material = find_material(root, material_name)
        set_text(material.find("E"), params[key])

    contact = find_contact(root)
    set_text(contact.find("fric_coeff"), params["friction"])

    forward = find_load_controller(root, "1")
    forward_points = forward.find("points")
    if forward_points is None:
        raise KeyError("Forward load controller has no points")
    for pt in list(forward_points):
        forward_points.remove(pt)
    ET.SubElement(forward_points, "pt").text = "0,0"
    ET.SubElement(forward_points, "pt").text = f"1,{params['forward_disp']:.10g}"

    vertical = find_load_controller(root, "2")
    vertical_points = vertical.find("points")
    if vertical_points is None:
        raise KeyError("Vertical load controller has no points")
    for pt in list(vertical_points):
        vertical_points.remove(pt)
    ET.SubElement(vertical_points, "pt").text = "0,0"
    ET.SubElement(vertical_points, "pt").text = f"0.2,{params['early_down_disp']:.10g}"
    ET.SubElement(vertical_points, "pt").text = f"{params['peak_time']:.10g},{params['peak_down_disp']:.10g}"
    ET.SubElement(vertical_points, "pt").text = f"1,{params['final_down_disp']:.10g}"

    scale_deformable_nodes(
        root,
        sx=params["scale_x"],
        sy=params["scale_y"],
        sz=params["scale_z"],
    )
    apply_shape_family(root, params, include_base_profile=include_base_profile)
    ensure_lateral_drive(root, params.get("lateral_disp", 0.0))
    add_output_loggers(root, sample.name)
    validate_boundary_conditions(root)

    out_feb.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="\t")
    tree.write(out_feb, encoding="ISO-8859-1", xml_declaration=True)


def febio_executable(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    from_path = shutil.which("febio4")
    if from_path:
        return from_path
    if Path(DEFAULT_FEBIO).exists():
        return DEFAULT_FEBIO
    raise FileNotFoundError(
        "Could not find febio4. Pass --febio or add febio4 to PATH."
    )


def run_febio(febio: str, feb_file: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [febio, "-i", str(feb_file)],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def normal_termination(log_text: str) -> bool:
    return "N O R M A L   T E R M I N A T I O N" in log_text


def von_mises(sx: float, sy: float, sz: float, sxy: float, syz: float, sxz: float) -> float:
    return math.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )
