import git
import os
import rename_after_writing as rnw


def init(plenoirf_dir):
    with rnw.open(os.path.join(plenoirf_dir, ".gitignore"), "wt") as f:
        f.write(_make_plenoirf_dir_gitignore())
    repo = git.Repo.init(plenoirf_dir)
    repo.git.branch(M="main")  # change name of branch to 'main'
    repo.git.add(".")
    repo.git.commit(m="init")


def is_dirty(plenoirf_dir):
    repo = git.Repo(plenoirf_dir)
    if repo.active_branch.name == "main":
        return repo.is_dirty()
    else:
        return True


def is_clean(plenoirf_dir):
    return not is_dirty(plenoirf_dir=plenoirf_dir)


def _make_plenoirf_dir_gitignore():
    txt = ""
    txt += "magnetic_deflection/*/*/store\n"
    txt += "magnetic_deflection/*/*/production\n"
    txt += "magnetic_deflection/plots\n"
    txt += "plenoptics/*\n"
    txt += "!plenoptics/config\n"
    txt += "response\n"
    txt += "trigger_geometry\n"
    txt += "provenance\n"
    txt += "debug\n"
    return txt
