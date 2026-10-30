import os

import meshio
import numpy as np
import pytest

from .conftest import RTOL, run_by_name, run_with_reference

base_folder = "immersed_fem"
fields_fluid = ["Pressure", "Velocity"]
fields_solid = ["Displacement", "Velocity", "VonMises_stress"]

# Keep the existing reference tolerances. Independently check fresh serial
# and MPI solutions with the stricter shared RTOL values below.
ifem_rtol = {
    "Pressure": 1.0e-5,
    "Velocity": 2.0e-5,
    "Displacement": 2.0e-9,
    "VonMises_stress": 2.0e-3,
}


@pytest.fixture(scope="module")
def serial_results():
    return {}


def _run_square_tube(test_folder, n_proc):
    folder = os.path.join("cases", base_folder, test_folder)
    fluid = run_by_name(
        folder, "solver.xml", 3, n_proc=n_proc, result_prefix="result_fluid_"
    )
    solid = meshio.read(os.path.join(folder, f"{n_proc}-procs", "result_solid_003.vtu"))
    return fluid, solid


def _check_square_tube(test_folder, n_proc, serial_results):
    if test_folder not in serial_results:
        serial_results[test_folder] = _run_square_tube(test_folder, 1)
    serial = serial_results[test_folder]
    results = serial if n_proc == 1 else _run_square_tube(test_folder, n_proc)
    for (part, fields), result, baseline in zip(
        (("fluid", fields_fluid), ("solid", fields_solid)), results, serial
    ):
        run_with_reference(
            base_folder, test_folder, fields, n_proc, 3,
            name_ref=f"result_{part}_003.vtu", field_rtol=ifem_rtol,
            result=result,
        )
        if part == "solid":
            # Element output must contain actual stress values, not just an
            # empty array or the metadata columns of the gathered buffer.
            element_stress = result.cell_data["E_VonMises"]
            serial_stress = baseline.cell_data["E_VonMises"]
            assert len(element_stress) == len(result.cells)
            for block, stress, serial_values in zip(
                result.cells, element_stress, serial_stress
            ):
                assert stress.shape == (len(block.data),)
                assert np.all(np.isfinite(stress))
                assert np.any(stress > 0.0)
                # This loaded block has nonuniform stress; constant domain or
                # partition metadata must not be mistaken for stress values.
                assert np.ptp(stress) > 0.0
                np.testing.assert_allclose(
                    stress, serial_values, rtol=RTOL["VonMises_stress"],
                    atol=RTOL["VonMises_stress"], equal_nan=False,
                )
        # Check ordering explicitly before comparing nodal fields.
        np.testing.assert_array_equal(result.points, baseline.points)
        for field in fields:
            np.testing.assert_allclose(
                result.point_data[field], baseline.point_data[field],
                rtol=RTOL[field], atol=RTOL[field], equal_nan=False,
                err_msg=f"{test_folder}: {part} {field}, {n_proc} ranks vs serial",
            )


def test_square_tube_2d(n_proc, serial_results):
    _check_square_tube("square_tube_2d", n_proc, serial_results)


def test_square_tube_3d(n_proc, serial_results):
    _check_square_tube("square_tube_3d", n_proc, serial_results)
