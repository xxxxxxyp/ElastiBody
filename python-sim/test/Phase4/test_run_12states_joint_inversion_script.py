import ast
import unittest


SCRIPT_PATH = "/home/runner/work/ElastiBody/ElastiBody/python-sim/test/Phase4/run_12states_joint_inversion.py"


class TestRun12StatesJointInversionScript(unittest.TestCase):
    def setUp(self):
        with open(SCRIPT_PATH, "r", encoding="utf-8") as script_file:
            self.source = script_file.read()
        self.tree = ast.parse(self.source, filename=SCRIPT_PATH)

    def test_script_is_valid_python(self):
        compile(self.source, SCRIPT_PATH, "exec")

    def test_extract_fixed_nodes_function_exists(self):
        function_names = [
            node.name for node in self.tree.body if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn("extract_fixed_nodes", function_names)
        self.assertIn('np.loadtxt(os.path.join(data_dir, f"pnt{state_idx}.txt"))', self.source)
        self.assertIn("fix_threshold = np.max(u_mag) * 0.05", self.source)
        self.assertIn("return np.where(u_mag < fix_threshold)[0]", self.source)

    def test_joint_solver_configuration_is_present(self):
        self.assertIn("obs_steps = list(range(1, 13))", self.source)
        self.assertIn("solver = InverseSolver(init, obs_step_list=obs_steps)", self.source)
        self.assertIn("ignore_nodes_list=ignore_nodes_list", self.source)
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
