import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PYTHON_SIM_PATH = os.path.join(REPO_ROOT, "python-sim")
if PYTHON_SIM_PATH not in sys.path:
    sys.path.append(PYTHON_SIM_PATH)

from sim.inverse import solver as solver_module


class DummyCPP:
    def __init__(self):
        self.current_displacement = np.zeros(12)
        self.set_displacement_calls = []
        self.last_modulus = None
        self.state_marker = 1

    def set_current_displacement(self, displacement):
        self.current_displacement = np.asarray(displacement, dtype=float)
        self.set_displacement_calls.append(self.current_displacement.copy())

    def gen_unit_geometric_forces(self):
        if self.state_marker <= 1:
            return np.ones((12, 1))
        return np.full((12, 1), float(self.state_marker))

    def set_element_modulus(self, modulus):
        self.last_modulus = np.asarray(modulus, dtype=float)


class DummyInitializer:
    def __init__(self, output_dir):
        self.nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.cells = np.array([[0, 1, 2, 3]], dtype=int)
        self.num_nodes = len(self.nodes)
        self.num_cells = len(self.cells)
        self.E_base = 10000.0
        self.model_name = "joint_unit_test"
        self.output_dir = output_dir
        self.cpp_backend = DummyCPP()
        self.E_field = np.full(self.num_cells, self.E_base)
        self.commit_calls = 0
        self.bc_history = []

    def commit_to_cpp(self):
        self.commit_calls += 1
        self.cpp_backend.set_element_modulus(self.E_field)


class FakeForwardSolver:
    def __init__(self, initializer):
        self.init = initializer
        self.u = np.zeros(initializer.num_nodes * 3)

    def set_dirichlet_bc(self, fixed_nodes_indices):
        fixed_nodes = np.asarray(fixed_nodes_indices, dtype=int)
        self.fixed_nodes = fixed_nodes
        self.init.bc_history.append(fixed_nodes.copy())

    def solve_static_step(self, force_input, step_index=0, tol=1e-4, max_iter=20):
        marker = int(np.asarray(force_input).reshape(-1)[0])
        self.init.cpp_backend.state_marker = marker
        self.u = np.full_like(self.u, float(marker))


class TestJointInverseSolver(unittest.TestCase):
    def _expected_z_displacement(self, value):
        displacement = np.zeros(self.initializer.num_nodes * 3)
        displacement[2::3] = value
        return displacement

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.initializer = DummyInitializer(self.output_dir)
        self._write_state(step_idx=1, displacement_value=1.0, force_value=1.0)
        self._write_state(step_idx=5, displacement_value=2.0, force_value=2.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_state(self, step_idx, displacement_value, force_value):
        nodes_obs = self.initializer.nodes.copy()
        nodes_obs[:, 2] += displacement_value
        nodes_obs = nodes_obs.flatten()
        f_ext = np.full(self.initializer.num_nodes * 3, force_value)
        np.savetxt(os.path.join(self.output_dir, f"pnt{step_idx}.txt"), nodes_obs)
        np.savetxt(os.path.join(self.output_dir, f"force{step_idx}.txt"), f_ext)

    def test_init_loads_multiple_states_and_preserves_first_state_fields(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])

        self.assertEqual(solver.obs_step_list, [1, 5])
        self.assertEqual(solver.num_states, 2)
        self.assertEqual(len(solver.u_meas_list), 2)
        self.assertEqual(len(solver.f_ext_list), 2)
        np.testing.assert_allclose(solver.u_meas_list[0], self._expected_z_displacement(1.0))
        np.testing.assert_allclose(solver.u_meas_list[1], self._expected_z_displacement(2.0))
        np.testing.assert_allclose(solver.u_meas_full, solver.u_meas_list[0])
        np.testing.assert_allclose(solver.f_ext, solver.f_ext_list[0])

    def test_joint_solve_accumulates_states_and_uses_state_specific_boundary_conditions(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])
        captured = {}

        def fake_minimize(fun, x0, jac, bounds, method, options):
            captured["x0"] = np.asarray(x0, dtype=float)
            captured["bounds"] = bounds
            captured["method"] = method
            captured["options"] = options
            captured["objective_at_2"] = fun(np.array([2.0]))
            captured["gradient_at_2"] = jac(np.array([2.0]))
            return mock.Mock(success=True, x=np.array([3000.0]))

        with mock.patch.object(solver_module, "NHookeanForwardSolver", FakeForwardSolver), \
             mock.patch.object(solver_module.opt, "minimize", side_effect=fake_minimize):
            result = solver.solve_alternating(
                lambda_reg=0.0,
                max_iter=1,
                ignore_nodes_list=[np.array([0]), np.array([1])],
                observed_nodes_list=[np.array([2]), np.array([2, 3])],
                alpha=0.0,
            )

        self.assertEqual(captured["method"], "L-BFGS-B")
        self.assertEqual(captured["options"], {"ftol": 1e-9, "gtol": 1e-5})
        np.testing.assert_allclose(captured["x0"], np.array([10000.0]), atol=1e-12)
        self.assertEqual(
            captured["bounds"],
            [(solver_module.PHYSICAL_MIN_E, solver_module.PHYSICAL_MAX_E)],
        )
        np.testing.assert_allclose(captured["objective_at_2"], 11.0, atol=1e-12)
        np.testing.assert_allclose(captured["gradient_at_2"], np.array([6.5]), atol=1e-12)
        np.testing.assert_array_equal(self.initializer.bc_history[0], np.array([0, 2]))
        np.testing.assert_array_equal(self.initializer.bc_history[1], np.array([1, 2, 3]))
        np.testing.assert_allclose(result, np.array([3000.0]))
        self.assertEqual(self.initializer.commit_calls, 1)

    def test_ignore_nodes_list_length_must_match_state_count(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])

        with self.assertRaises(ValueError):
            solver.solve_alternating(max_iter=1, ignore_nodes_list=[np.array([0])])

    def test_observed_nodes_list_length_must_match_state_count(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])

        with self.assertRaises(ValueError):
            solver.solve_alternating(max_iter=1, observed_nodes_list=[np.array([0])])


if __name__ == "__main__":
    unittest.main()
