import plenoirf
import json_line_logger
import io
import numpy as np
import json_utils


def test_seed_section_context():
    stream = io.StringIO()

    logger = json_line_logger.LoggerStream(stream=stream)

    with plenoirf.seeding.SeedSection(
        run_id=137, module=str, logger=logger
    ) as sec:
        assert sec.run_id == 137
        assert sec.block_id == 0
        _seed_used_in_section = sec.seed
        assert sec.name == "str"

        # the work in this section
        prng = np.random.Generator(np.random.PCG64(sec.seed))
        random_string_A_to_Z = (
            prng.integers(low=65, high=90, size=1024)
            .astype(np.uint8)
            .tobytes()
            .decode()
        )
        pos = sec.module.find(random_string_A_to_Z, "A")

        # "always the same result given the same run_id and module."
        assert pos == 6

    stream.seek(0)

    logs = []
    with json_utils.lines.Reader(file=stream) as json_lines_reader:
        for log_entry in json_lines_reader:
            logs.append(
                plenoirf.seeding.SeedSection.parse_json_lines_log_entry(
                    log_entry
                )
            )

    assert len(logs) == 2
    assert logs[0]["m"]["name"] == "str"
    assert logs[0]["m"]["run_id"] == 137
    assert logs[0]["m"]["block_id"] == 0
    assert logs[0]["m"]["seed"] == _seed_used_in_section
    assert logs[0]["m"]["status"] == "enter"

    assert logs[1]["m"]["name"] == logs[0]["m"]["name"]
    assert logs[1]["m"]["run_id"] == logs[0]["m"]["run_id"]
    assert logs[1]["m"]["block_id"] == logs[0]["m"]["block_id"]
    assert logs[1]["m"]["seed"] == logs[0]["m"]["seed"]
    assert logs[1]["m"]["status"] == "exit"

    assert logs[1]["t"] >= logs[0]["t"]
