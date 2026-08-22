from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def test_submit_script_maps_each_array_task_to_its_own_seeded_sample(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    source_script = repository / "submit_simgen.sh"
    assert source_script.is_file(), "the SimGen SLURM submission script must exist"

    project = tmp_path / "simgen"
    examples = project / "examples"
    examples.mkdir(parents=True)
    script = project / "submit_simgen.sh"
    shutil.copy2(source_script, script)
    (examples / "panda_ball_can.yaml").write_text(
        "seed: 42\noutputs:\n  keep_simulation: true\n"
    )
    (project / "simgen.sif").touch()

    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    invocation_path = tmp_path / "singularity-invocation.txt"
    singularity = binary_dir / "singularity"
    singularity.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "pathlib.Path(os.environ['SIMGEN_TEST_INVOCATION']).write_text('\\n'.join(sys.argv[1:]))\n"
    )
    singularity.chmod(0o755)
    environment = os.environ | {
        "PATH": f"{binary_dir}{os.pathsep}{os.environ['PATH']}",
        "SLURM_ARRAY_TASK_ID": "17",
        "SIMGEN_TEST_INVOCATION": str(invocation_path),
    }

    result = subprocess.run(
        ["bash", str(script)], cwd=project, env=environment, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    invocation = invocation_path.read_text().splitlines()
    assert "--nv" in invocation
    assert f"{tmp_path}:/workspace" in invocation
    assert str(project / "simgen.sif") in invocation
    assert "/workspace/simgen/examples/.panda_ball_can_seed_17" in "\n".join(invocation)
    assert "runs/panda_ball_can/sample_17" in invocation
    assert not list(examples.glob(".panda_ball_can_seed_17.*"))
