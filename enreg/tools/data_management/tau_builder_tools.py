import os
import enreg
# import subprocess
import numpy as np
import awkward as ak
from textwrap import dedent
from enreg.tools import slurm_tools as st
import pdb


def process_single_file(input_path: str, builder, output_path: str) -> None:
    if not os.path.exists(output_path):
        print("Opening file %s" % input_path)
        jets = ak.from_parquet(input_path)
        print("Processing jets...")
        pjets = builder.process_jets(jets)
        # pdb.set_trace()
        # assert(len(jets["gen_tau_label"]) == len(pjets["tau_dm"]))
        print("...done, writing output file %s" % output_path)
        merged_info = {field: jets[field] for field in jets.fields}
        merged_info.update(pjets)
        ak.to_parquet(ak.Record(merged_info), output_path)
    else:
        print(f"File [{output_path}] already processed ... Skipping")


def multipath_slurm_tau_builder(input_paths, output_paths, batch_size=7):
    output_dir = st.create_tmp_run_dir()
    print(f"Temporary directory created to {output_dir}")
    print(f"Run `bash enreg/scripts/submit_builder_batchJobs.sh {output_dir}/executables/`")
    number_batches = int(len(input_paths) / batch_size) + 1
    input_path_chunks = list(np.array_split(input_paths, number_batches))
    output_path_chunks = list(np.array_split(output_paths, number_batches))
    input_file_paths, output_file_paths = create_job_input_list(input_path_chunks, output_path_chunks, output_dir)
    for job_idx, (input_file_path, output_file_path) in enumerate(zip(input_file_paths, output_file_paths)):
        job_file = prepare_job_file(input_file_path, output_file_path, job_idx, output_dir)
        # subprocess.call(["sbatch", job_file])
    # wait_iteration()


def create_job_input_list(input_path_chunks, output_path_chunks, output_dir):
    input_file_paths = []
    output_file_paths = []
    for i, (in_chunk, out_chunk) in enumerate(zip(input_path_chunks, output_path_chunks)):
        input_file_path = os.path.join(output_dir, f"input_paths_{i}.txt")
        output_file_path = os.path.join(output_dir, f"output_paths_{i}.txt")
        with open(input_file_path, 'wt') as outFile:
            for path in in_chunk:
                outFile.write(path + "\n")
        with open(output_file_path, 'wt') as outFile:
            for path in out_chunk:
                outFile.write(path + "\n")
        input_file_paths.append(input_file_path)
        output_file_paths.append(output_file_path)
    return input_file_paths, output_file_paths


def prepare_job_file(
        input_file,
        output_file,
        job_idx,
        output_dir,
):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    input_file : str
        Path to the file containing the list of .parquet files to be processed
    output_file : str
        Location where the processed .parquet file will be saved
    job_idx : int
        Number of the job.
    output_dir : str
        Directory where the temporary output will be written

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    job_dir = os.path.join(output_dir, "executables")
    os.makedirs(job_dir, exist_ok=True)
    error_dir = os.path.join(output_dir, "error_files")
    os.makedirs(error_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log_files")
    os.makedirs(log_dir, exist_ok=True)
    job_file = os.path.join(job_dir, 'execute' + str(job_idx) + '.sh')
    error_file = os.path.join(error_dir, 'error' + str(job_idx))
    log_file = os.path.join(log_dir, 'output' + str(job_idx))
    run_script = os.path.join(os.path.dirname(enreg.__file__), "scripts", "runBuilder.py")
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            f"""
                #!/bin/bash
                #SBATCH --job-name=tauBuilder
                #SBATCH --ntasks=1
                #SBATCH --partition=short
                #SBATCH --time=00:15:00
                #SBATCH --cpus-per-task=2
                #SBATCH -e {error_file}
                #SBATCH -o {log_file}
                env
                date
                ./run.sh python {run_script} slurm_run=True +input_file={input_file} +output_file={output_file}
            """).strip('\n'))
    return job_file
