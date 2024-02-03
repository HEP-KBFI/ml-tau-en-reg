import os
import string
import random


def generate_run_id(run_id_len=10):
    """ Creates a random alphanumeric string with a length of run_id_len

    Args:
        run_id_len : int
            [default: 10] Length of the alphanumeric string

    Returns:
        random_string : str
            The randomly generated alphanumeric string
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=run_id_len))


def create_tmp_run_dir():
    """ Creates the temporary directory where the slurm job, log and error files are saved for each job

    Args:
        None

    Returns:
        tmp_dir : str
            Path to the temporary directory
    """
    run_id = generate_run_id()
    tmp_dir = os.path.join(os.path.expandvars("$HOME"), "tmp", run_id)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir
