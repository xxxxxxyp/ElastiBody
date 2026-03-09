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

    def set_current_displacement(self, displacement):
        self.current_displacement = np.asarray(displacement, dtype=float)
        self.set_displacement_calls.append(self.current_displacement.copy())

    def gen_unit_geometric_forces(self):
        marker = int(round(float(np.mean(self.current_displacement))))
        if marker <= 1:
            return np.ones((12, 1))
        return np.full((12, 1), float(marker))

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
        self.model_name = "joint-unit-test"
        self.output_dir = output_dir
        self.cpp_backend = DummyCPP()
        self.E_field = np.full(self.num_cells, self.E_base)
        self.commit_calls = 0

    def commit_to_cpp(self):
        self.commit_calls += 1
        self.cpp_backend.set_element_modulus(self.E_field)


class FakeForwardSolver:
    bc_history = []

    def __init__(self, initializer):
        self.init = initializer
        self.u = np.zeros(initializer.num_nodes * 3)

    def set_dirichlet_bc(self, fixed_nodes_indices):
        fixed_nodes = np.asarray(fixed_nodes_indices, dtype=int)
        self.fixed_nodes = fixed_nodes
        FakeForwardSolver.bc_history.append(fixed_nodes.copy())

    def solve_static_step(self, force_input, step_index=0, tol=1e-4, max_iter=20):
        marker = int(np.asarray(force_input).reshape(-1)[0])
        self.u = np.full_like(self.u, float(marker))


class JointInverseSolverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.initializer = DummyInitializer(self.output_dir)
        self._write_state(step_idx=1, displacement_value=1.0, force_value=1.0)
        self._write_state(step_idx=5, displacement_value=2.0, force_value=2.0)

    def tearDown(self):
        FakeForwardSolver.bc_history = []
        self.temp_dir.cleanup()

    def _write_state(self, step_idx, displacement_value, force_value):
        nodes_obs = self.initializer.nodes.flatten() + displacement_value
        f_ext = np.full(self.initializer.num_nodes * 3, force_value)
        np.savetxt(os.path.join(self.output_dir, f"pnt{step_idx}.txt"), nodes_obs)
        np.savetxt(os.path.join(self.output_dir, f"force{step_idx}.txt"), f_ext)

    def test_init_loads_multiple_states_and_preserves_first_state_fields(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])

        self.assertEqual(solver.obs_step_list, [1, 5])
        self.assertEqual(solver.num_states, 2)
        self.assertEqual(len(solver.u_meas_list), 2)
        self.assertEqual(len(solver.f_ext_list), 2)
        np.testing.assert_allclose(solver.u_meas_list[0], np.ones(12))
        np.testing.assert_allclose(solver.u_meas_list[1], np.full(12, 2.0))
        np.testing.assert_allclose(solver.u_meas_full, solver.u_meas_list[0])
        np.testing.assert_allclose(solver.f_ext, solver.f_ext_list[0])

    def test_joint_solve_accumulates_states_and_uses_state_specific_boundary_conditions(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])
        captured = {}

        def fake_spsolve(matrix, rhs):
            captured["A"] = matrix.toarray()
            captured["b"] = np.asarray(rhs, dtype=float)
            return np.array([3000.0])

        with mock.patch.object(solver_module, "NHookeanForwardSolver", FakeForwardSolver), \
             mock.patch.object(solver_module.spla, "spsolve", side_effect=fake_spsolve):
            result = solver.solve_alternating(
                lambda_reg=0.0,
                max_iter=1,
                ignore_nodes_list=[np.array([0]), np.array([1])],
                alpha=0.0,
            )

        np.testing.assert_allclose(captured["A"], np.array([[2.0]]), atol=1e-12)
        np.testing.assert_allclose(captured["b"], np.array([-9.0]), atol=1e-12)
        np.testing.assert_array_equal(FakeForwardSolver.bc_history[0], np.array([0]))
        np.testing.assert_array_equal(FakeForwardSolver.bc_history[1], np.array([1]))
        np.testing.assert_allclose(result, np.array([3000.0]))
        self.assertEqual(self.initializer.commit_calls, 1)

    def test_ignore_nodes_list_length_must_match_state_count(self):
        solver = solver_module.InverseSolver(self.initializer, obs_step_list=[1, 5])

        with self.assertRaises(ValueError):
            solver.solve_alternating(max_iter=1, ignore_nodes_list=[np.array([0])])


if __name__ == "__main__":
    unittest.main()
