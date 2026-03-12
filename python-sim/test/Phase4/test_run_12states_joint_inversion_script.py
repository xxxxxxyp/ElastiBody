import ast
import os
import unittest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_12states_joint_inversion.py",
)


class TestRun12StatesJointInversionScript(unittest.TestCase):
    def setUp(self):
        with open(SCRIPT_PATH, "r", encoding="utf-8") as script_file:
            self.source = script_file.read()
        self.tree = ast.parse(self.source, filename=SCRIPT_PATH)

    def test_script_is_valid_python(self):
        compile(self.source, SCRIPT_PATH, "exec")

    def test_load_physics_masks_function_exists(self):
        function_node = next(
            (
                node
                for node in self.tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "load_physics_masks"
            ),
            None,
        )
        self.assertIsNotNone(function_node)

        loadtxt_calls = [
            node
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loadtxt"
        ]
        self.assertTrue(loadtxt_calls)
        self.assertIn('f"A-{state_idx}.txt"', self.source)
        self.assertIn('f"B-{state_idx}.txt"', self.source)
        self.assertIn("if not os.path.exists(filepath):", self.source)
        self.assertIn("if data.ndim == 1 and len(data) >= 2:", self.source)
        self.assertIn("dofs = data[:, 0].astype(int)", self.source)
        self.assertIn("return np.unique(dofs // 3)", self.source)
        self.assertIn("return fixed_nodes, observed_nodes", self.source)

    def test_joint_solver_configuration_is_present(self):
        self.assertIn("obs_steps = list(range(1, 13))", self.source)
        self.assertIn("solver = InverseSolver(init, obs_step_list=obs_steps)", self.source)
        self.assertIn("ignore_nodes_list=ignore_nodes_list", self.source)
        self.assertIn("fixed_nodes, observed_nodes = load_physics_masks(state_idx, data_dir)", self.source)
        self.assertIn("ignore_nodes_list.append(fixed_nodes)", self.source)
        self.assertIn('f"[BC] State {state_idx}: detected {len(fixed_nodes)} fixed nodes, "', self.source)
        self.assertIn('f"{len(observed_nodes)} observed nodes."', self.source)
        self.assertIn("lambda_reg=1e-15", self.source)
        self.assertIn("max_iter=20", self.source)
        self.assertIn("alpha=0.6", self.source)

    def test_output_files_are_exported(self):
        self.assertIn('"joint_inversion_12states.txt"', self.source)
        self.assertIn('"joint_inversion_12states.vtk"', self.source)
        self.assertIn('"E_Recon": [E_recon_safe]', self.source)
        self.assertIn('"E_Ref": [E_ref_safe]', self.source)
        self.assertIn('"E_Diff": [E_recon_safe - E_ref_safe]', self.source)


if __name__ == "__main__":
    unittest.main()
