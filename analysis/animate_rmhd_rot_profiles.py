"""Animate the saved rotating 1D RMHD neutron-star wind profiles."""

from animate_rmhd_profiles import REPO_DIR, animate_rmhd_profiles


DEFAULT_RUN_DIR = REPO_DIR / "problems" / "rmhd_rot_mpi"


def main() -> None:
    animate_rmhd_profiles(
        default_run_dir=DEFAULT_RUN_DIR,
        default_output_name="wind_evolution_rmhd_rot.gif",
    )


if __name__ == "__main__":
    main()
