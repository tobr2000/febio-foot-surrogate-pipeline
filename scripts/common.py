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

ANATOMIC_MATERIAL_NAMES = {
    "E_flesh": ["Continuous_skin_fascia_envelope", "Skeletal_muscle_bulk", "Intrinsic_foot_muscle"],
    "E_bone": ["Cortical_trabecular_bone"],
    "E_joint": ["Articular_cartilage"],
    "E_heel": ["Subcutaneous_fat_and_heel_pad"],
    "E_forefoot": ["Intrinsic_foot_muscle", "Metatarsal_fat_pad"],
    "E_collar": ["Ligament_capsule_bands"],
    "E_plantar": ["Plantar_fascia_aponeurosis"],
    "E_achilles": ["Tendon_collagen_bands"],
}

SOLE_SURFACE_CANDIDATES = ("FootSoleContact", "AnatomicSoleContact")
TOP_SURFACE_CANDIDATES = ("TibiaTopDriveSurface", "KneeCutDriveSurface")
LATERAL_FIXITY_BC_CANDIDATES = ("TibiaNoLateralDrift", "KneeMedialLateralStability")


@dataclass(frozen=True)
class Sample:
    sample_id: int
    seed: int
    params: dict[str, float]
    dataset_id: str = "default"

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
            dataset_id = str(row.get("dataset_id") or row.get("params", {}).get("dataset_id", "default"))
            params = {
                k: float(v)
                for k, v in row["params"].items()
                if k != "dataset_id"
            }
            samples.append(
                Sample(
                    sample_id=int(row["sample_id"]),
                    seed=int(row["seed"]),
                    params=params,
                    dataset_id=dataset_id,
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


def material_lookup(root: ET.Element) -> dict[str, ET.Element]:
    return {
        material.attrib["name"]: material
        for material in root.findall("./Material/material")
        if "name" in material.attrib
    }


def surface_exists(root: ET.Element, name: str) -> bool:
    return root.find(f"./Mesh/Surface[@name='{name}']") is not None


def first_existing_surface(root: ET.Element, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if surface_exists(root, name):
            return name
    raise KeyError(f"None of these surfaces exist: {', '.join(candidates)}")


def set_template_materials(root: ET.Element, params: dict[str, float]) -> None:
    materials = material_lookup(root)
    if "Flesh_soft_tissue" in materials:
        mapping = {key: [value] for key, value in MATERIAL_NAMES.items()}
    elif "Continuous_skin_fascia_envelope" in materials:
        mapping = ANATOMIC_MATERIAL_NAMES
    else:
        raise KeyError("Could not identify template material family")

    for key, names in mapping.items():
        if key not in params:
            continue
        for name in names:
            material = materials.get(name)
            if material is not None:
                set_text(material.find("E"), params[key])


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


def set_time_control(root: ET.Element, time_steps: int | None, step_size: float | None) -> None:
    if time_steps is None and step_size is None:
        return
    control = root.find("./Control")
    if control is None:
        raise KeyError("No Control section found")
    if time_steps is not None:
        set_text(control.find("time_steps"), int(time_steps))
    if step_size is not None:
        set_text(control.find("step_size"), f"{float(step_size):.10g}")


def node_set_ids(root: ET.Element, name: str) -> set[int]:
    node_set = root.find(f"./Mesh/NodeSet[@name='{name}']")
    if node_set is None or not node_set.text:
        return set()
    return {int(value.strip()) for value in node_set.text.split(",") if value.strip()}


def apply_structured_insole(root: ET.Element, params: dict[str, float]) -> None:
    """Perturb v10 insole patch heights to create local contact switching."""
    tile_sets = [
        node_set
        for node_set in root.findall("./Mesh/NodeSet")
        if node_set.attrib.get("name", "").startswith("InsoleTile_")
    ]
    if not tile_sets:
        return

    nodes_by_id = {
        int(node.attrib["id"]): node
        for node in root.findall("./Mesh/Nodes/node")
    }
    medial_bias = float(params.get("insole_medial_bias", 0.0))
    lateral_bias = float(params.get("insole_lateral_bias", 0.0))
    heel_lift = float(params.get("insole_heel_lift", 0.0))
    forefoot_lift = float(params.get("insole_forefoot_lift", 0.0))
    ridge_amp = float(params.get("insole_ridge_amp", 0.0))
    pocket_amp = float(params.get("insole_pocket_amp", 0.0))

    for node_set in tile_sets:
        name = node_set.attrib["name"]
        _, ix_text, iy_text = name.split("_")
        ix = int(ix_text)
        iy = int(iy_text)
        region_weight = ix / 5.0
        side_weight = iy - 1.0
        key = f"insole_h_{ix:02d}_{iy:02d}"
        local = float(params.get(key, 0.0))
        height = (
            0.001
            + local
            + (1.0 - region_weight) * heel_lift
            + region_weight * forefoot_lift
            + max(0.0, side_weight) * medial_bias
            + max(0.0, -side_weight) * lateral_bias
            + ridge_amp * math.sin((ix + 1) * math.pi / 6.0) * (1.0 if iy == 1 else 0.35)
            - pocket_amp * (1.0 if ix in {3, 4} and iy == 1 else 0.0)
        )
        height = min(0.0055, max(-0.0015, height))
        for node_id in node_set_ids(root, name):
            node = nodes_by_id.get(node_id)
            if node is None or node.text is None:
                continue
            x, y, z = (float(v) for v in node.text.split(","))
            z2 = height if z > -0.004 else height - 0.010
            node.text = f"{x:.10g},{y:.10g},{z2:.10g}"


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
    if abs(float(lateral_disp)) < 1e-12:
        return
    boundary = root.find("./Boundary")
    if boundary is None:
        return
    top_surface = first_existing_surface(root, TOP_SURFACE_CANDIDATES)
    for bc in list(boundary.findall("bc")):
        if bc.attrib.get("name") in LATERAL_FIXITY_BC_CANDIDATES:
            # Once y is prescribed by TibiaLateralDrive, this old helper BC would
            # have no active DOFs. FEBio treats that as an initialization error.
            boundary.remove(bc)

    exists = any(bc.attrib.get("name") == "TopLateralDrive" for bc in boundary.findall("bc"))
    if not exists:
        bc = ET.SubElement(
            boundary,
            "bc",
            {
                "name": "TopLateralDrive",
                "node_set": f"@surface:{top_surface}",
                "type": "prescribed displacement",
            },
        )
        ET.SubElement(bc, "dof").text = "y"
        ET.SubElement(bc, "value", {"lc": "99"}).text = "1"
        ET.SubElement(bc, "relative").text = "1"

    load_data = root.find("./LoadData")
    if load_data is None:
        load_data = ET.SubElement(root, "LoadData")
    controller = None
    for item in load_data.findall("load_controller"):
        if item.attrib.get("id") == "99":
            controller = item
            break
    if controller is None:
        controller = ET.SubElement(
            load_data,
            "load_controller",
            {"id": "99", "name": "lateral_sway", "type": "loadcurve"},
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
    sole_surface = first_existing_surface(root, SOLE_SURFACE_CANDIDATES)

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
            "surface": sole_surface,
            "file": f"{sample_name}_contact.csv",
            "delim": ",",
        },
    )


def scale_anatomic_pressure_loads(root: ET.Element, params: dict[str, float]) -> None:
    load_scales = {
        "HeelBehaviorGRFPressure": params.get("heel_pressure_scale", 1.0),
        "MidfootBehaviorGRFPressure": params.get("midfoot_pressure_scale", 1.0),
        "ForefootBehaviorGRFPressure": params.get("forefoot_pressure_scale", 1.0),
        "ToeOffBehaviorGRFPressure": params.get("toe_pressure_scale", 1.0),
    }
    for load in root.findall("./Loads/surface_load"):
        name = load.attrib.get("name", "")
        if name not in load_scales:
            continue
        pressure = load.find("pressure")
        if pressure is None or pressure.text is None:
            continue
        pressure.text = f"{float(pressure.text) * float(load_scales[name]):.10g}"


def _set_loadcurve_points(root: ET.Element, controller_id: str, points: list[tuple[float, float]]) -> None:
    controller = find_load_controller(root, controller_id)
    point_parent = controller.find("points")
    if point_parent is None:
        raise KeyError(f"Load controller {controller_id} has no points")
    for pt in list(point_parent):
        point_parent.remove(pt)
    for time_value, value in points:
        ET.SubElement(point_parent, "pt").text = f"{time_value:.10g},{value:.10g}"


def tune_anatomic_gait_curves(root: ET.Element, params: dict[str, float]) -> None:
    """Shift legacy regional gait pressure timing when those controllers exist."""
    if not any(key in params for key in ("heel_peak_time", "midfoot_peak_time", "forefoot_peak_time", "toe_peak_time")):
        return
    if not all(root.find(f"./LoadData/load_controller[@id='{controller_id}']") is not None for controller_id in ("3", "4", "5", "6")):
        return

    def triangular_points(peak_key: str, width_key: str, tail: float = 0.0) -> list[tuple[float, float]]:
        peak = min(0.98, max(0.02, float(params.get(peak_key, 0.5))))
        width = min(0.45, max(0.03, float(params.get(width_key, 0.18))))
        start = max(0.0, peak - width)
        end = min(1.0, peak + width)
        shoulder_left = max(start, peak - 0.45 * width)
        shoulder_right = min(end, peak + 0.45 * width)
        raw_points = [
            (0.0, 0.0),
            (start, 0.0),
            (shoulder_left, 0.35),
            (peak, 1.0),
            (shoulder_right, 0.55),
            (end, tail),
            (1.0, tail),
        ]
        points: list[tuple[float, float]] = []
        for time_value, value in raw_points:
            if points and abs(time_value - points[-1][0]) < 1e-9:
                points[-1] = (time_value, value)
            else:
                points.append((time_value, value))
        return points

    _set_loadcurve_points(root, "3", triangular_points("heel_peak_time", "heel_width", tail=0.0))
    _set_loadcurve_points(root, "4", triangular_points("midfoot_peak_time", "midfoot_width", tail=0.0))
    _set_loadcurve_points(root, "5", triangular_points("forefoot_peak_time", "forefoot_width", tail=0.12))
    _set_loadcurve_points(root, "6", triangular_points("toe_peak_time", "toe_width", tail=0.10))


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
    time_steps: int | None = None,
    step_size: float | None = None,
) -> None:
    tree = ET.parse(template)
    root = tree.getroot()
    params = sample.params

    set_time_control(root, time_steps=time_steps, step_size=step_size)
    set_template_materials(root, params)

    contact = find_contact(root)
    set_text(contact.find("fric_coeff"), params["friction"])

    force_loads = {load.attrib.get("name"): load for load in root.findall("./Loads/nodal_load")}
    if "TopForwardForcePerNode" in force_loads or "TopDownForcePerNode" in force_loads:
        forward_force = force_loads.get("TopForwardForcePerNode")
        if forward_force is not None:
            scale = forward_force.find("scale")
            if scale is not None:
                scale.text = f"{params.get('forward_force_per_node', 0.000000008):.10g}"
        down_force = force_loads.get("TopDownForcePerNode")
        if down_force is not None:
            scale = down_force.find("scale")
            if scale is not None:
                scale.text = f"{-abs(params.get('down_force_per_node', 0.000000024)):.10g}"
        mid = float(params.get("force_ramp_mid", 0.34))
        final = float(params.get("force_ramp_final", 0.48))
        _set_loadcurve_points(root, "1", [(0.0, 0.0), (0.20, 0.01), (0.45, 0.15), (0.70, mid), (1.0, final)])
    elif "KneeForwardArcForce" in force_loads or "KneeDownArcForce" in force_loads:
        forward_force = force_loads.get("KneeForwardArcForce")
        if forward_force is not None:
            scale = forward_force.find("scale")
            if scale is not None:
                scale.text = f"{params.get('forward_force_per_node', 0.00012):.10g}"
        down_force = force_loads.get("KneeDownArcForce")
        if down_force is not None:
            scale = down_force.find("scale")
            if scale is not None:
                scale.text = f"{-abs(params.get('down_force_per_node', 0.00035)):.10g}"
        _set_loadcurve_points(root, "1", [(0.0, 0.0), (0.20, 0.0), (0.50, 0.30), (0.78, 0.85), (1.0, 1.0)])
        _set_loadcurve_points(root, "2", [(0.0, 0.0), (0.12, 0.18), (0.42, 0.75), (0.72, 1.0), (1.0, 0.60)])
    else:
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
    apply_structured_insole(root, params)
    ensure_lateral_drive(root, params.get("lateral_disp", 0.0))
    scale_anatomic_pressure_loads(root, params)
    tune_anatomic_gait_curves(root, params)
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
